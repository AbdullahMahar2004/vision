import binascii
import datetime
import signal
import socket
import struct
import time

import cv2
import numpy as np
import rclpy
import tensorrt as trt
from cuda import cudart
from rclpy.node import Node


def cuda_check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f'CUDA error: {err}')


class GimbalController:
    """
    Minimal SIYI SDK client (UDP control port, separate from the RTSP video
    stream) for a one-shot relative move at startup. Reuses the same commands
    and P-controller gain calibrated against the real camera in
    gimbal_hold_up.py -- see that file for the full protocol writeup and how
    the sign/reference conventions were derived.

    Known calibrated on this unit: positive pitch_rate moves the camera UP
    and the raw pitch reading DECREASES as it moves up (so "down" = negative
    rate, raw reading increases). Yaw sign for "left" is NOT independently
    verified -- best-effort guess (negative yaw_rate = left), flagged in the
    printed output so it's obvious to correct if backwards.
    """
    CMD_ACQUIRE_ATTITUDE = 0x0D
    CMD_GIMBAL_ROTATION = 0x07
    P_GAIN = 1.5
    RATE_MAX_DEG = 90.0

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.3)
        self._seq = 0

    def _build_packet(self, cmd_id, data=b''):
        body = struct.pack('<BHHB', 1, len(data), self._seq, cmd_id) + data
        self._seq = (self._seq + 1) & 0xFFFF
        header = b'\x55\x66' + body
        crc = binascii.crc_hqx(header, 0)
        return header + struct.pack('<H', crc)

    def _send(self, cmd_id, data=b''):
        self.sock.sendto(self._build_packet(cmd_id, data), (self.ip, self.port))
        try:
            resp, _ = self.sock.recvfrom(1024)
            return resp
        except socket.timeout:
            return None

    def get_attitude(self):
        resp = self._send(self.CMD_ACQUIRE_ATTITUDE)
        if resp is None or len(resp) < 21:
            return None
        yaw, pitch, roll = struct.unpack('<hhh', resp[8:14])
        return yaw / 10.0, pitch / 10.0, roll / 10.0

    def set_rate(self, yaw_rate, pitch_rate):
        yaw_rate = max(-100, min(100, int(round(yaw_rate))))
        pitch_rate = max(-100, min(100, int(round(pitch_rate))))
        self._send(self.CMD_GIMBAL_ROTATION, struct.pack('<bb', yaw_rate, pitch_rate))

    def _rate_for_error(self, error_deg):
        return max(-100.0, min(100.0, error_deg * 100.0 * self.P_GAIN / self.RATE_MAX_DEG))

    def move_relative(self, yaw_left_deg, pitch_down_deg, tolerance_deg=2.0,
                       timeout_sec=8.0, rate_hz=10.0):
        """Best-effort one-shot move; never raises -- logs and gives up if the
        control port is unreachable, so a gimbal problem can't take the
        vision pipeline down with it."""
        start = self.get_attitude()
        if start is None:
            print('[GIMBAL] no response from control port, skipping startup move',
                  flush=True)
            return
        yaw0, pitch0, _ = start
        target_yaw = yaw0 - yaw_left_deg     # best-effort sign, see class docstring
        target_pitch = pitch0 + pitch_down_deg  # calibrated: down = raw pitch increases
        print(f'[GIMBAL] moving {yaw_left_deg:.0f} deg left, {pitch_down_deg:.0f} deg down '
              f'from yaw={yaw0:.1f} pitch={pitch0:.1f}', flush=True)

        period = 1.0 / rate_hz
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            t0 = time.time()
            attitude = self.get_attitude()
            if attitude is None:
                time.sleep(period)
                continue
            cur_yaw, cur_pitch, _ = attitude
            yaw_err = target_yaw - cur_yaw
            pitch_err = cur_pitch - target_pitch
            if abs(yaw_err) <= tolerance_deg and abs(pitch_err) <= tolerance_deg:
                self.set_rate(0, 0)
                print(f'[GIMBAL] reached yaw={cur_yaw:.1f} pitch={cur_pitch:.1f}', flush=True)
                return
            self.set_rate(self._rate_for_error(yaw_err), self._rate_for_error(pitch_err))
            time.sleep(max(0.0, period - (time.time() - t0)))

        self.set_rate(0, 0)
        print('[GIMBAL] timed out before reaching target, stopped where it is', flush=True)


class TiledYoloTRT:
    """
    Wraps a fixed-batch YOLO11 TensorRT engine (input NCHW, output (B, 4+num_classes, N))
    and runs it over a full frame by slicing the frame into a batch-sized grid of tiles.

    Written against tensorrt + cuda-python only (no pycuda, no torch -- neither is
    installed on this Jetson image).
    """

    def __init__(self, engine_path, n_cols, n_rows, conf_threshold, nms_iou):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))   # (B, 3, tile, tile)
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))  # (B, 4+C, N)

        self.batch = self.in_shape[0]
        self.tile = self.in_shape[2]
        self.num_classes = self.out_shape[1] - 4
        if self.batch != n_cols * n_rows:
            raise ValueError(
                f'engine batch {self.batch} does not match tiling grid {n_cols}x{n_rows}')
        # A 1x1 grid means "not actually tiled" -- resize the whole frame into
        # the single tile instead of cropping a tile-sized corner out of it.
        self.resize_whole_frame = (n_cols == 1 and n_rows == 1)

        in_nbytes = int(np.prod(self.in_shape)) * 4
        out_nbytes = int(np.prod(self.out_shape)) * 4
        err, self.d_in = cudart.cudaMalloc(in_nbytes)
        cuda_check(err)
        err, self.d_out = cudart.cudaMalloc(out_nbytes)
        cuda_check(err)
        err, self.stream = cudart.cudaStreamCreate()
        cuda_check(err)

        self.context.set_tensor_address(self.in_name, self.d_in)
        self.context.set_tensor_address(self.out_name, self.d_out)

        self.h_in = np.empty(self.in_shape, dtype=np.float32)
        self.h_out = np.empty(self.out_shape, dtype=np.float32)

        self.in_nbytes = in_nbytes
        self.out_nbytes = out_nbytes

    def tile_origins(self, frame_w, frame_h):
        """Evenly spaced tile top-left corners covering the frame edge to edge."""
        def positions(dim, n):
            if n == 1:
                return [0]
            step = (dim - self.tile) / (n - 1)
            return [max(0, min(dim - self.tile, round(i * step))) for i in range(n)]

        xs = positions(frame_w, self.n_cols)
        ys = positions(frame_h, self.n_rows)
        return [(x0, y0) for y0 in ys for x0 in xs]

    def infer(self, frame):
        """Run tiled inference on a full BGR frame. Returns list of (x1,y1,x2,y2,conf,cls)."""
        h, w = frame.shape[:2]

        if self.resize_whole_frame:
            origins = [(0, 0)]
            scale_x, scale_y = w / self.tile, h / self.tile
            resized = cv2.resize(frame, (self.tile, self.tile))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.h_in[0] = rgb.transpose(2, 0, 1)
        else:
            origins = self.tile_origins(w, h)
            scale_x, scale_y = 1.0, 1.0
            for i, (x0, y0) in enumerate(origins):
                tile = frame[y0:y0 + self.tile, x0:x0 + self.tile]
                rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                self.h_in[i] = rgb.transpose(2, 0, 1)

        cuda_check(cudart.cudaMemcpyAsync(
            self.d_in, self.h_in.ctypes.data, self.in_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)[0])
        self.context.execute_async_v3(stream_handle=self.stream)
        cuda_check(cudart.cudaMemcpyAsync(
            self.h_out.ctypes.data, self.d_out, self.out_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)[0])
        cuda_check(cudart.cudaStreamSynchronize(self.stream)[0])

        all_boxes_xywh = []
        all_scores = []
        all_cls = []
        for i, (x0, y0) in enumerate(origins):
            pred = self.h_out[i]  # (4+C, N)
            boxes = pred[0:4, :].T  # (N,4) cx,cy,w,h in tile-local pixels
            class_scores = pred[4:, :].T  # (N,C)
            cls_ids = np.argmax(class_scores, axis=1)
            confs = class_scores[np.arange(class_scores.shape[0]), cls_ids]

            keep = confs > self.conf_threshold
            if not np.any(keep):
                continue
            boxes, confs, cls_ids = boxes[keep], confs[keep], cls_ids[keep]

            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx, bw = cx * scale_x, bw * scale_x
            cy, bh = cy * scale_y, bh * scale_y
            x1 = cx - bw / 2 + x0
            y1 = cy - bh / 2 + y0

            for j in range(len(confs)):
                all_boxes_xywh.append([float(x1[j]), float(y1[j]), float(bw[j]), float(bh[j])])
                all_scores.append(float(confs[j]))
                all_cls.append(int(cls_ids[j]))

        if not all_boxes_xywh:
            return []

        # Global NMS merges duplicate detections of the same object seen in
        # more than one tile (the tile grid overlaps by construction).
        idxs = cv2.dnn.NMSBoxes(all_boxes_xywh, all_scores, self.conf_threshold, self.nms_iou)
        detections = []
        for i in np.array(idxs).flatten():
            x, y, bw, bh = all_boxes_xywh[i]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            x1 = max(0.0, min(x1, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w - 1))
            y2 = max(0.0, min(y2, h - 1))
            detections.append((x1, y1, x2, y2, all_scores[i], all_cls[i]))
        return detections

    def destroy(self):
        cudart.cudaFree(self.d_in)
        cudart.cudaFree(self.d_out)
        cudart.cudaStreamDestroy(self.stream)


class VisionDetectNode(Node):

    def __init__(self):
        super().__init__('vision_detect_node')

        self.declare_parameter('rtsp_url', 'rtsp://192.168.144.25:8554/main.264')
        self.declare_parameter(
            'engine_path', '/home/sky/vision/models/suas_v2_yolo11n_640_b1_fp16.engine')
        self.declare_parameter('class_names', ['person'])
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('nms_iou', 0.45)
        self.declare_parameter('tile_cols', 1)
        self.declare_parameter('tile_rows', 1)
        self.declare_parameter('output_video_path', '')
        self.declare_parameter('output_video_clean_path', '')
        self.declare_parameter('disconnect_timeout_sec', 5.0)
        self.declare_parameter('max_duration_sec', 300.0)
        self.declare_parameter('move_gimbal_at_start', True)
        self.declare_parameter('gimbal_ip', '192.168.144.25')
        self.declare_parameter('gimbal_port', 37260)
        self.declare_parameter('gimbal_yaw_left_deg', 25.0)
        self.declare_parameter('gimbal_pitch_down_deg', 45.0)

        rtsp_url = self.get_parameter('rtsp_url').value
        engine_path = self.get_parameter('engine_path').value
        self.class_names = self.get_parameter('class_names').value
        conf_threshold = self.get_parameter('conf_threshold').value
        nms_iou = self.get_parameter('nms_iou').value
        tile_cols = self.get_parameter('tile_cols').value
        tile_rows = self.get_parameter('tile_rows').value
        output_video_path = self.get_parameter('output_video_path').value
        output_video_clean_path = self.get_parameter('output_video_clean_path').value
        self.disconnect_timeout_sec = self.get_parameter('disconnect_timeout_sec').value
        self.max_duration_sec = self.get_parameter('max_duration_sec').value
        move_gimbal_at_start = self.get_parameter('move_gimbal_at_start').value
        gimbal_ip = self.get_parameter('gimbal_ip').value
        gimbal_port = self.get_parameter('gimbal_port').value
        gimbal_yaw_left_deg = self.get_parameter('gimbal_yaw_left_deg').value
        gimbal_pitch_down_deg = self.get_parameter('gimbal_pitch_down_deg').value

        if move_gimbal_at_start:
            GimbalController(gimbal_ip, gimbal_port).move_relative(
                gimbal_yaw_left_deg, gimbal_pitch_down_deg)

        self.get_logger().info(f'Loading engine: {engine_path}')
        self.model = TiledYoloTRT(engine_path, tile_cols, tile_rows, conf_threshold, nms_iou)
        self.get_logger().info(
            f'Engine loaded: batch={self.model.batch} tile={self.model.tile} '
            f'classes={self.model.num_classes} grid={tile_cols}x{tile_rows}')

        capture_pipeline = (
            f'rtspsrc location={rtsp_url} latency=100 protocols=tcp ! '
            'rtph264depay ! h264parse ! nvv4l2decoder ! '
            'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! '
            'appsink drop=true max-buffers=1 sync=false'
        )
        self.cap = cv2.VideoCapture(capture_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f'Failed to open RTSP capture pipeline for {rtsp_url}')

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out_fps = fps if fps and fps > 1.0 else 30.0
        self.get_logger().info(
            f'Capture opened: {self.frame_w}x{self.frame_h} @ {self.out_fps:.1f}fps (nominal)')

        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if not output_video_path:
            output_video_path = f'/home/sky/vision/output/annotated_{stamp}.mp4'
        if not output_video_clean_path:
            output_video_clean_path = f'/home/sky/vision/output/clean_{stamp}.mp4'
        self.output_video_path = output_video_path
        self.output_video_clean_path = output_video_clean_path

        def make_writer(path):
            pipeline = (
                'appsrc ! videoconvert ! video/x-raw,format=I420 ! '
                'x264enc speed-preset=ultrafast tune=zerolatency bitrate=6000 key-int-max=30 ! '
                f'h264parse ! mp4mux ! filesink location={path}'
            )
            writer = cv2.VideoWriter(
                pipeline, cv2.CAP_GSTREAMER, 0, self.out_fps,
                (self.frame_w, self.frame_h), True)
            if not writer.isOpened():
                raise RuntimeError(f'Failed to open video writer pipeline for {path}')
            return writer

        self.writer = make_writer(output_video_path)
        self.writer_clean = make_writer(output_video_clean_path)
        self.get_logger().info(f'Recording annotated video to {output_video_path}')
        self.get_logger().info(f'Recording clean video to {output_video_clean_path}')

        self.frame_count = 0
        self.detection_count = 0
        self.running = True

    def report_detections(self, detections):
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            self.detection_count += 1
            # Plain print(), not get_logger(): ROS 2 logging also publishes to
            # /rosout over DDS, which hits the same ~1s-per-call stall as the
            # image publisher on this Jetson's RMW -- unusable on a per-detection
            # hot path. get_logger() is reserved for infrequent lifecycle events.
            print(
                f'[DETECTION] frame={self.frame_count} class={name} conf={conf:.2f} '
                f'bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})', flush=True)

    def draw_detections(self, frame, detections):
        """Draws boxes + class/conf labels in place, in the same pixel space
        infer() returns (already scaled to the full frame), before the frame
        goes to the writer."""
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            label = f'{name} {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(pt1[1], th + 4)
            cv2.rectangle(frame, (pt1[0], label_y - th - 4), (pt1[0] + tw + 2, label_y),
                          (0, 255, 0), -1)
            cv2.putText(frame, label, (pt1[0] + 1, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def request_stop(self, signum=None, frame=None):
        # Shared by SIGINT and SIGTERM so an external `kill` also drains to a
        # clean writer.release() (EOS) instead of a truncated mp4.
        self.get_logger().info(f'Stop requested (signal={signum}), finishing cleanly...')
        self.running = False

    def spin_capture_loop(self):
        t_start = time.time()
        first_failure_time = None
        while rclpy.ok() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                now = time.time()
                if first_failure_time is None:
                    first_failure_time = now
                    # print(), not get_logger(): same ~1s-per-call DDS cost as
                    # elsewhere, and this path can fire on every loop iteration.
                    print('[WARN] Frame read failed, camera may be disconnected, '
                          'retrying...', flush=True)
                elif now - first_failure_time > self.disconnect_timeout_sec:
                    print(f'[WARN] No frames for {self.disconnect_timeout_sec:.0f}s, '
                          f'treating camera as disconnected. Stopping and finalizing '
                          f'the recording with what was captured.', flush=True)
                    self.running = False
                time.sleep(0.1)  # avoid busy-spinning while disconnected
                continue
            first_failure_time = None
            self.frame_count += 1

            detections = self.model.infer(frame)
            # Write the clean frame before draw_detections() modifies it in
            # place -- VideoWriter.write() copies the buffer synchronously
            # (memcpy into the GstBuffer before appsrc push returns), so this
            # is safe to follow with in-place drawing.
            self.writer_clean.write(frame)
            if detections:
                self.report_detections(detections)
                self.draw_detections(frame, detections)

            self.writer.write(frame)
            # No rclpy.spin_once() here: this node takes no subscriptions,
            # timers, or services to service, and get_logger()/publish() calls
            # on this Jetson's RMW cost ~0.3-1s+ per call regardless (measured)
            # -- kept off the hot path entirely, see report_detections()/
            # draw_detections().

            if time.time() - t_start > self.max_duration_sec:
                print(f'[INFO] Reached max_duration_sec={self.max_duration_sec:.0f}s, '
                      f'stopping and finalizing the recording.', flush=True)
                self.running = False

        elapsed = time.time() - t_start
        self.get_logger().info(
            f'Stopped after {self.frame_count} frames / {elapsed:.1f}s '
            f'({self.frame_count / elapsed:.1f} fps), {self.detection_count} detections total. '
            f'Annotated: {self.output_video_path}  Clean: {self.output_video_clean_path}')

    def shutdown(self):
        self.running = False
        self.cap.release()
        self.writer.release()  # sends EOS through mp4mux, not a hard kill
        self.writer_clean.release()
        self.model.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = VisionDetectNode()
    signal.signal(signal.SIGINT, node.request_stop)
    signal.signal(signal.SIGTERM, node.request_stop)
    try:
        node.spin_capture_loop()
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
