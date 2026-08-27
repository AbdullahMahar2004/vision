"""
Points the SIYI A8 mini gimbal upward and holds it there, correcting back if
it gets physically bumped. Uses the documented SIYI SDK control protocol
(UDP port 37260 -- separate from the RTSP video stream).

CMD_ID 0x0E ("Set Gimbal Attitude", absolute angle) and 0x08 ("Center") were
tried first and got acknowledged but produced no physical movement on this
unit/firmware. What actually moves it, confirmed empirically, is the same
approach ArduPilot's own AP_Mount_Siyi driver uses in production
(libraries/AP_Mount/AP_Mount_Siyi.cpp): a P-controller loop that reads
attitude (0x0D) and continuously drives the rate/speed command (0x07) toward
a target, rather than a one-shot angle set.

  CMD_ID 0x0D  Acquire Gimbal Attitude (query, empty request)
               response: yaw, pitch, roll, yaw_vel, pitch_vel, roll_vel
               each int16, actual degrees = value / 10
  CMD_ID 0x07  Gimbal Rotation (rate control)
               request: yaw_speed (int8, -100..100), pitch_speed (int8, -100..100)

Two things measured on THIS unit that don't match the public docs, so don't
trust the documented absolute range/sign -- trust this file's constants,
which were calibrated against the real camera:
  - The SIYI SDK doc's pitch range (-135..+45, 0=level) doesn't match what
    0x0D actually reports here: "forward" (the physical boot/rest position)
    reads as ~178 raw, not 0.
  - Sign is inverted from the naive assumption: sending a POSITIVE pitch_rate
    moves the camera UP, and the raw pitch reading DECREASES as it moves up.
Because of this, the target here is expressed as "degrees up from wherever
the gimbal was pointed when this script started" (--pitch-up), not as an
absolute angle in the documented coordinate system.

P-controller (same gain/max-rate ArduPilot uses):
  rate_scalar = clamp(error_deg * 100 * P_GAIN / RATE_MAX_DEG, -100, 100)
  P_GAIN = 1.5, RATE_MAX_DEG = 90  ->  rate_scalar = clamp(error_deg * 1.667, -100, 100)

Includes basic stall protection: if a nonzero rate is commanded but the
reading isn't moving, that's very likely the gimbal pushed against a
mechanical limit -- stop commanding further rather than grinding against it.

Usage:
  python3 gimbal_hold_up.py                 # hold 45 deg up from wherever it starts
  python3 gimbal_hold_up.py --pitch-up 30   # hold 30 deg up instead
"""
import argparse
import binascii
import socket
import struct
import sys
import time

IP = '192.168.144.25'
PORT = 37260

CMD_ACQUIRE_ATTITUDE = 0x0D
CMD_GIMBAL_ROTATION = 0x07

P_GAIN = 1.5
RATE_MAX_DEG = 90.0

STALL_EPSILON_DEG = 0.5   # reading change below this counts as "not moving"
STALL_TIMEOUT_SEC = 1.5   # how long to see no movement before calling it a stall


def build_packet(cmd_id, data=b'', ctrl=1, seq=0):
    body = struct.pack('<BHHB', ctrl, len(data), seq, cmd_id) + data
    header = b'\x55\x66' + body
    crc = binascii.crc_hqx(header, 0)  # CRC-16/XMODEM (poly 0x1021, init 0), over STX+body
    return header + struct.pack('<H', crc)


class Gimbal:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.3)
        self._seq = 0

    def _send(self, cmd_id, data=b''):
        pkt = build_packet(cmd_id, data, seq=self._seq)
        self._seq = (self._seq + 1) & 0xFFFF
        self.sock.sendto(pkt, (IP, PORT))
        try:
            resp, _ = self.sock.recvfrom(1024)
            return resp
        except socket.timeout:
            return None

    def get_attitude(self):
        """Returns (yaw_deg, pitch_deg, roll_deg) or None if no response."""
        resp = self._send(CMD_ACQUIRE_ATTITUDE)
        if resp is None or len(resp) < 21:
            return None
        data = resp[8:20]
        yaw, pitch, roll = struct.unpack('<hhh', data[0:6])
        return yaw / 10.0, pitch / 10.0, roll / 10.0

    def set_rate(self, yaw_rate, pitch_rate):
        yaw_rate = max(-100, min(100, int(round(yaw_rate))))
        pitch_rate = max(-100, min(100, int(round(pitch_rate))))
        data = struct.pack('<bb', yaw_rate, pitch_rate)
        self._send(CMD_GIMBAL_ROTATION, data)


def rate_for_error(error_deg):
    return max(-100.0, min(100.0, error_deg * 100.0 * P_GAIN / RATE_MAX_DEG))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pitch-up', type=float, default=45.0,
                     help='Degrees to tilt up from wherever the gimbal is pointed at '
                          'startup (default 45)')
    ap.add_argument('--yaw', type=float, default=None,
                     help='Target yaw in degrees (default: hold current yaw)')
    ap.add_argument('--tolerance', type=float, default=2.0,
                     help='Degrees of error treated as "arrived" (default 2.0)')
    ap.add_argument('--rate-hz', type=float, default=10.0,
                     help='Control loop frequency (default 10 Hz)')
    args = ap.parse_args()

    gimbal = Gimbal()

    attitude = gimbal.get_attitude()
    if attitude is None:
        print('Could not read gimbal attitude -- camera unreachable on the control port.',
              file=sys.stderr)
        sys.exit(1)
    yaw, pitch, roll = attitude
    target_yaw = args.yaw if args.yaw is not None else yaw
    # Positive pitch_rate moves the camera up, and the raw reading DECREASES
    # as it moves up (calibrated against the real unit) -- so "up" means a
    # smaller target_pitch than the starting reading.
    target_pitch = pitch - args.pitch_up
    print(f'Current attitude: yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}')
    print(f'Target: yaw={target_yaw:.1f} pitch={target_pitch:.1f} '
          f'({args.pitch_up:.0f} deg up from start) (+/-{args.tolerance} deg), Ctrl+C to stop')

    period = 1.0 / args.rate_hz
    was_at_target = False
    last_pitch = pitch
    last_move_time = time.time()
    stalled = False
    try:
        while True:
            t0 = time.time()
            attitude = gimbal.get_attitude()
            if attitude is None:
                print('[WARN] attitude read failed, retrying...', flush=True)
                time.sleep(period)
                continue
            cur_yaw, cur_pitch, cur_roll = attitude
            yaw_err = target_yaw - cur_yaw
            pitch_err = cur_pitch - target_pitch  # inverted, see module docstring
            at_target = abs(yaw_err) <= args.tolerance and abs(pitch_err) <= args.tolerance

            if abs(cur_pitch - last_pitch) > STALL_EPSILON_DEG:
                last_move_time = t0
                last_pitch = cur_pitch
                stalled = False

            if at_target:
                gimbal.set_rate(0, 0)
                if not was_at_target:
                    print(f'[HOLDING] yaw={cur_yaw:.1f} pitch={cur_pitch:.1f}', flush=True)
            elif stalled:
                gimbal.set_rate(0, 0)
            elif t0 - last_move_time > STALL_TIMEOUT_SEC:
                stalled = True
                gimbal.set_rate(0, 0)
                print(f'[STALLED] pitch stuck at {cur_pitch:.1f} (target {target_pitch:.1f}), '
                      f'likely at a mechanical limit -- holding here instead of pushing further',
                      flush=True)
            else:
                gimbal.set_rate(rate_for_error(yaw_err), rate_for_error(pitch_err))
                if was_at_target:
                    print(f'[CORRECTING] drifted to yaw={cur_yaw:.1f} pitch={cur_pitch:.1f}, '
                          f'moving back to target', flush=True)
            was_at_target = at_target

            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        gimbal.set_rate(0, 0)
        print('\nStopped.')


if __name__ == '__main__':
    main()
