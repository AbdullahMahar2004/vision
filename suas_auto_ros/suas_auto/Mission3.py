#!/usr/bin/env python3
"""
mission_full.py
==================
All five mission nodes + shared config + launcher, merged into a single
file. Run it directly with `python3 mission_full.py` -- it starts all
five nodes together in one process, so no separate mission_launch.py or
ROS 2 package entry points are needed for a quick run/test.

  takeoff -> endurance laps -> search/zigzag (+ hover on red-car
  detection, reports real GPS location) -> payload release -> RTL

Each node still hands the baton to the next over /mission/*_done topics
(std_msgs/Bool), exactly as when they were separate files -- merging
them into one process doesn't change that design, it just means you
launch one script instead of five.

WHY THE HANDOFF PATTERN: PX4 OFFBOARD mode just needs *someone*
streaming OffboardControlMode + TrajectorySetpoint at >2 Hz -- it
doesn't care which node. So each flight-phase node:
  1. Stays silent (no timer running) until it receives True on its
     trigger topic.
  2. Runs its own 10 Hz control loop as the only active publisher.
  3. On completion, cancels its own timer *then* publishes True on its
     own /mission/..._done topic, handing control to the next node.
This avoids two nodes ever publishing setpoints at the same time.
The handoff topics use RELIABLE + TRANSIENT_LOCAL ("latched") QoS via
handoff_qos(), so a downstream node spinning up a beat late still gets
the "done" it missed.

BUG FIXED FROM THE ORIGINAL zigzag_box.py: handle_rtl() had a
mixed-indentation line that caused an IndentationError on import.
Fixed here (see SearchZigzagNode / RtlNode).

BEFORE YOU FLY:
  1. Fill in LAP_WAYPOINTS below with the real competition sequence (up
     to 15 (lat, lon, alt_rel) tuples = one lap) from Check-In, and set
     NUM_LAPS (1-10) based on your endurance budget.
  2. Confirm CORNER_BR/TR/BL/TL still match your actual search boundary.
  3. Confirm IMG_WIDTH/IMG_HEIGHT match your real camera resolution (see
     red_car_detector_node.py, run separately -- it's not merged into
     this file since it's a vision node, not a flight-phase node).
  4. Run:  python3 mission_full.py
     (or drop this into a ROS 2 package and add one console_script entry
     point if you'd rather launch it the normal ROS 2 way).

SANITY-CHECK WITHOUT A VEHICLE: `ros2 topic echo /mission/takeoff_done`
-- it won't fire without real VehicleLocalPosition/VehicleStatus data,
but confirms nodes spin cleanly and the topic graph wires up.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from sensor_msgs.msg import NavSatFix

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLocalPosition,
    VehicleStatus,
)

# ============================================================
# mission_common -- shared config + helpers (was mission_common.py)
# ============================================================

EARTH_RADIUS_M = 6371000.0

# ==================== CONFIG ====================

# --- Takeoff / general flight ---
# Reference launch point (informational only -- takeoff climbs straight up
# from wherever the vehicle is armed, using the EKF's own local origin):
#   lat 31.3104685, lon 30.0651584
TAKEOFF_ALT_REL = -4.0        # NED, negative = up. Altitude to climb to before laps.
ACCEPTANCE_RADIUS_M = 2.0     # "arrived at waypoint" radius, used by lap + search nodes
ALT_ACCEPTANCE_M = 0.5        # "arrived at altitude" tolerance, used during takeoff
OFFBOARD_WARMUP_TICKS = 10    # PX4 needs >=10 setpoints streamed before accepting OFFBOARD

# --- Endurance lap (3.2.2 Waypoints) ---
# Fill these in from the competition waypoint packet at Check-In.
# Up to 15 (lat, lon, alt_rel_NED) tuples = ONE lap, ~2 miles.
LAP_ALT_REL = -6.0
LAP_WAYPOINTS = [
    (31.3095769, 30.0659171, LAP_ALT_REL),   # WP1
    (31.3095343, 30.0660105, LAP_ALT_REL),   # WP2
    (31.3094310, 30.0660476, LAP_ALT_REL),   # WP3
    (31.3095769, 30.0659171, LAP_ALT_REL),   # back to WP1 -- closes the loop
]
NUM_LAPS = 1   # 1-10 allowed by the rules; pick based on your endurance budget

# --- Search box (zigzag) -- outer corners of the search boundary.
# NOTE: this boundary is a rotated quadrilateral, not an axis-aligned
# rectangle, so corners are assigned by which edges actually connect in
# the given perimeter (N -> W -> S -> E -> N), not by compass box logic:
#   BL = W vertex, TL = N vertex, BR = S vertex, TR = E vertex
SEARCH_ALT_REL = -4.0
CORNER_BR = (31.3094310, 30.0660476)   # S vertex -- entry corner, search flies here first
CORNER_TR = (31.3095379, 30.0661717)   # E vertex
CORNER_BL = (31.3095769, 30.0659171)   # W vertex
CORNER_TL = (31.3096358, 30.0660996)   # N vertex -- exit corner, pattern ends here
NUM_LANES = 4                     # fixed lane count; set to None to use LANE_SPACING_M
LANE_SPACING_M = 3.0               # only used if NUM_LANES is None

# --- Payload release ---
GRIPPER_INSTANCE = 1
ACK_TIMEOUT_TICKS = 40   # ~2s at 20Hz

# --- Target hover / visual centering (triggered by red_car_detector_node) ---
# Camera resolution -- MUST match your actual camera sensor's <width>/<height>
# in the model SDF, or the "center of frame" math below will be wrong.
IMG_WIDTH = 640
IMG_HEIGHT = 480

PIXEL_CENTER_TOL_PX = 25     # how close to frame-center counts as "centered"
HOVER_STABLE_TICKS = 20      # ~2s at 10Hz -- must stay centered this long before handoff
HOVER_LOST_TICKS = 30        # ~3s of no detection before giving up and resuming zigzag
HOVER_MAX_TICKS = 300        # ~30s hard timeout -- proceed anyway rather than loiter forever

# Rough proportional visual-servo gain: meters of NED nudge per pixel of
# error. This is a crude approximation (no real camera intrinsics/altitude
# scaling) -- tune down if it overshoots/oscillates, tune up if it creeps
# too slowly. Assumes a roughly nadir-pointing camera aligned with the
# vehicle body frame; if it nudges the wrong direction, flip the sign.
VISUAL_SERVO_GAIN_M_PER_PX = 0.01

# ==================== QoS ====================

def pub_qos():
    """For publishing to PX4 (/fmu/in/...)."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def sub_qos():
    """For subscribing to PX4 (/fmu/out/...)."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def handoff_qos():
    """
    For the /mission/*_done Bool topics that pass the baton between nodes.
    RELIABLE + TRANSIENT_LOCAL ("latched") so a downstream node that spins
    up a beat late still receives a "done" that already fired.
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

# ==================== Geometry ====================

def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def latlon_to_local(local_pos, lat, lon, alt_rel):
    """
    Convert (lat, lon) to the vehicle's local NED frame using the EKF's
    published reference origin. `local_pos` is the latest
    VehicleLocalPosition message (must have valid ref_lat/ref_lon, i.e.
    xy_global True).
    """
    ref_lat = local_pos.ref_lat
    ref_lon = local_pos.ref_lon
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    x = dlat * EARTH_RADIUS_M
    y = dlon * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    z = alt_rel
    return x, y, z


def local_to_latlon(local_pos, x, y):
    """
    Inverse of latlon_to_local -- convert a local NED (x, y) offset back
    to (lat, lon) using the EKF's reference origin. Used to report where
    a detected target actually is in real-world GPS coordinates, not just
    which pixel it showed up at.
    """
    ref_lat = local_pos.ref_lat
    ref_lon = local_pos.ref_lon
    dlat_rad = x / EARTH_RADIUS_M
    dlon_rad = y / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat)))
    lat = ref_lat + math.degrees(dlat_rad)
    lon = ref_lon + math.degrees(dlon_rad)
    return lat, lon


def generate_zigzag(logger, bl, br, tl, tr, num_lanes=None, lane_spacing=None):
    """Same lane-generation logic as the original zigzag_box.py."""
    total_height = haversine_m(bl[0], bl[1], tl[0], tl[1])

    if num_lanes is not None:
        num_lanes = max(1, int(num_lanes))
        logger.info(
            f'Search area height: {total_height:.1f}m -- {num_lanes} lanes '
            f'(fixed by NUM_LANES, ~{total_height / num_lanes:.2f}m/lane).')
    else:
        num_lanes = max(1, int(total_height / lane_spacing))
        logger.info(
            f'Search area height: {total_height:.1f}m -- {num_lanes} lanes '
            f'(auto-derived from {lane_spacing}m spacing).')

    waypoints = []
    go_right = False
    for i in range(num_lanes + 1):
        frac = i / num_lanes if num_lanes > 0 else 0
        left_lat = bl[0] + frac * (tl[0] - bl[0])
        left_lon = bl[1] + frac * (tl[1] - bl[1])
        right_lat = br[0] + frac * (tr[0] - br[0])
        right_lon = br[1] + frac * (tr[1] - br[1])

        if go_right:
            waypoints.append((left_lat, left_lon))
            waypoints.append((right_lat, right_lon))
        else:
            waypoints.append((right_lat, right_lon))
            waypoints.append((left_lat, left_lon))
        go_right = not go_right

    return waypoints


# ============================================================
# Phase 1: TakeoffNode (was takeoff_node.py)
# ============================================================

class TakeoffNode(Node):

    def __init__(self):
        super().__init__('takeoff_node')

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', pub_qos())
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', pub_qos())
        self.vehiclecommand_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', pub_qos())

        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_cb, sub_qos())
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.status_cb, sub_qos())
        self.create_subscription(
            VehicleCommandAck, '/fmu/out/vehicle_command_ack', self.ack_cb, sub_qos())

        self.done_pub = self.create_publisher(Bool, '/mission/takeoff_done', handoff_qos())

        self.local_pos = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.have_local_ref = False
        self.offboard_mode_confirmed = False
        self.mode_switch_sent_tick = None

        self.offboard_setpoint_counter = 0
        self.armed_and_offboard_sent = False
        self.armed = False
        self.finished = False

        self.timer = self.create_timer(0.1, self.timer_cb)  # 10 Hz
        self.get_logger().info(
            f'TakeoffNode ready, target alt {-TAKEOFF_ALT_REL:.1f}m AGL. Starting...')

    # ------------------------------------------------------------------
    def local_cb(self, msg: VehicleLocalPosition):
        self.local_pos = msg
        if msg.xy_global and msg.z_global:
            self.have_local_ref = True

    def status_cb(self, msg: VehicleStatus):
        self.vehicle_status = msg
        if msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.offboard_mode_confirmed = True

    def ack_cb(self, msg: VehicleCommandAck):
        if msg.result != VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED:
            names = {
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE: 'DO_SET_MODE (offboard switch)',
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM: 'ARM',
            }
            label = names.get(msg.command, f'command {msg.command}')
            self.get_logger().error(
                f'-- REJECTED: {label} -- result code {msg.result}. '
                f'Check QGC Messages / MAVLink console for the reason.')

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x, y, z):
        sp = TrajectorySetpoint()
        sp.position = [x, y, z]
        nan = float('nan')
        sp.velocity = [nan, nan, nan]
        sp.acceleration = [nan, nan, nan]
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(sp)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehiclecommand_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    # ------------------------------------------------------------------
    def timer_cb(self):
        if self.finished:
            return

        self.publish_offboard_heartbeat()
        self.publish_setpoint(0.0, 0.0, TAKEOFF_ALT_REL)

        if self.offboard_setpoint_counter < OFFBOARD_WARMUP_TICKS:
            self.offboard_setpoint_counter += 1
            return

        if not self.have_local_ref:
            if self.offboard_setpoint_counter % 20 == 0:  # ~every 2s
                self.get_logger().warn(
                    '-- Waiting for valid local position estimate (GPS/EKF) '
                    'before requesting OFFBOARD. Not switching modes yet.')
            self.offboard_setpoint_counter += 1
            return

        if not self.armed_and_offboard_sent:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)  # 6 = OFFBOARD
            self.armed_and_offboard_sent = True
            self.mode_switch_sent_tick = self.offboard_setpoint_counter
            self.get_logger().info('-- OFFBOARD mode switch sent, waiting for confirmation before arming...')
            self.offboard_setpoint_counter += 1
            return

        if not self.offboard_mode_confirmed:
            ticks_waited = self.offboard_setpoint_counter - self.mode_switch_sent_tick
            if ticks_waited > 30:  # 3s with no confirmation -> something's wrong
                self.get_logger().error(
                    '-- OFFBOARD mode never confirmed 3s after switch request. '
                    'Vehicle is likely rejecting it (no RC link / COM_RCL_EXCEPT, '
                    'or position estimate not good enough). Check QGC Messages.')
            self.offboard_setpoint_counter += 1
            return

        if not self.armed:
            self.arm()
            self.armed = True
            self.get_logger().info('-- OFFBOARD confirmed, arm command sent, climbing...')

        if abs(self.local_pos.z - TAKEOFF_ALT_REL) < ALT_ACCEPTANCE_M:
            self.get_logger().info(
                f'-- Target altitude reached ({-TAKEOFF_ALT_REL:.1f}m AGL). '
                f'Handing off to endurance lap phase.')
            self.finish()

    def finish(self):
        self.finished = True
        self.timer.cancel()
        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)


# ============================================================
# Phase 2: EnduranceLapNode (was endurance_lap_node.py)
# ============================================================

class EnduranceLapNode(Node):

    def __init__(self):
        super().__init__('endurance_lap_node')

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', pub_qos())
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', pub_qos())
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_cb, sub_qos())

        self.done_pub = self.create_publisher(Bool, '/mission/laps_done', handoff_qos())
        self.create_subscription(Bool, '/mission/takeoff_done', self.trigger_cb, handoff_qos())

        self.local_pos = VehicleLocalPosition()
        self.local_waypoints = []   # filled in once we have ref_lat/ref_lon
        self.wp_index = 0
        self.lap_num = 1

        self.started = False
        self.active = False
        self.timer = None

        if not LAP_WAYPOINTS:
            self.get_logger().error(
                'mission_common.LAP_WAYPOINTS is empty -- fill it in from the '
                'competition waypoint packet before flying.')

        self.get_logger().info(
            f'EnduranceLapNode ready ({NUM_LAPS} laps x {len(LAP_WAYPOINTS)} waypoints). '
            f'Waiting for takeoff_done...')

    # ------------------------------------------------------------------
    def local_cb(self, msg: VehicleLocalPosition):
        self.local_pos = msg

    def trigger_cb(self, msg: Bool):
        if msg.data and not self.started:
            self.started = True
            self.active = True
            self.get_logger().info('-- Takeoff complete, starting endurance laps')
            self.timer = self.create_timer(0.1, self.timer_cb)  # 10 Hz

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x, y, z):
        sp = TrajectorySetpoint()
        sp.position = [x, y, z]
        nan = float('nan')
        sp.velocity = [nan, nan, nan]
        sp.acceleration = [nan, nan, nan]
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(sp)

    # ------------------------------------------------------------------
    def timer_cb(self):
        if not self.active:
            return

        self.publish_offboard_heartbeat()

        if not self.local_waypoints:
            if not self.local_pos.xy_global:
                # EKF origin not published yet -- hold position, wait.
                self.publish_setpoint(0.0, 0.0, LAP_ALT_REL)
                return
            self.local_waypoints = [
                latlon_to_local(self.local_pos, lat, lon, alt)
                for (lat, lon, alt) in LAP_WAYPOINTS
            ]
            if not self.local_waypoints:
                self.get_logger().error('No LAP_WAYPOINTS configured, cannot fly laps. Skipping to search phase.')
                self.finish()
                return

        x, y, z = self.local_waypoints[self.wp_index]
        self.publish_setpoint(x, y, z)

        dx = self.local_pos.x - x
        dy = self.local_pos.y - y
        dz = self.local_pos.z - z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < ACCEPTANCE_RADIUS_M:
            self.wp_index += 1
            if self.wp_index >= len(self.local_waypoints):
                self.get_logger().info(f'-- Lap {self.lap_num}/{NUM_LAPS} complete')
                if self.lap_num >= NUM_LAPS:
                    self.get_logger().info('-- All endurance laps complete, handing off to search phase')
                    self.finish()
                    return
                self.lap_num += 1
                self.wp_index = 0
            else:
                self.get_logger().info(
                    f'Lap {self.lap_num}/{NUM_LAPS} -- waypoint {self.wp_index + 1}/{len(self.local_waypoints)}')

    def finish(self):
        self.active = False
        if self.timer is not None:
            self.timer.cancel()
        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)


# ============================================================
# Phase 3: SearchZigzagNode (was search_zigzag_node.py)
# ============================================================

class SearchZigzagNode(Node):

    def __init__(self):
        super().__init__('search_zigzag_node')

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', pub_qos())
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', pub_qos())
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_cb, sub_qos())

        self.done_pub = self.create_publisher(Bool, '/mission/search_done', handoff_qos())
        self.create_subscription(Bool, '/mission/laps_done', self.trigger_cb, handoff_qos())

        self.target_loc_pub = self.create_publisher(NavSatFix, '/mission/target_location', handoff_qos())

        self.create_subscription(Bool, '/detection/red_object_found', self.detect_found_cb, sub_qos())
        self.create_subscription(Point, '/detection/red_object_pixel', self.detect_pixel_cb, sub_qos())

        self.local_pos = VehicleLocalPosition()

        self.raw_waypoints = generate_zigzag(
            self.get_logger(), CORNER_BL, CORNER_BR, CORNER_TL, CORNER_TR,
            num_lanes=NUM_LANES, lane_spacing=LANE_SPACING_M)
        self.local_waypoints = []
        self.wp_index = 0
        self._last_logged_wp = -1

        # FLY_TO_ENTRY -> ZIGZAG -> HOVER_TARGET
        self.substate = 'FLY_TO_ENTRY'
        self.started = False
        self.active = False
        self.timer = None

        # Detection / hover-centering state
        self.target_found = False
        self.target_pixel = None
        self.hover_setpoint = None
        self.hover_stable_count = 0
        self.hover_lost_count = 0
        self.hover_tick_count = 0
        self.saved_zigzag_wp_index = 0

        self.get_logger().info(
            f'SearchZigzagNode ready, {len(self.raw_waypoints)} zigzag waypoints generated. '
            f'Waiting for laps_done...')

    # ------------------------------------------------------------------
    def local_cb(self, msg: VehicleLocalPosition):
        self.local_pos = msg

    def detect_found_cb(self, msg: Bool):
        self.target_found = msg.data

    def detect_pixel_cb(self, msg: Point):
        self.target_pixel = msg

    def trigger_cb(self, msg: Bool):
        if msg.data and not self.started:
            self.started = True
            self.active = True
            self.get_logger().info('-- Endurance laps complete, flying to search box entry')
            self.timer = self.create_timer(0.1, self.timer_cb)  # 10 Hz

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x, y, z):
        sp = TrajectorySetpoint()
        sp.position = [x, y, z]
        nan = float('nan')
        sp.velocity = [nan, nan, nan]
        sp.acceleration = [nan, nan, nan]
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(sp)

    # ------------------------------------------------------------------
    def timer_cb(self):
        if not self.active:
            return

        self.publish_offboard_heartbeat()

        if not self.local_pos.xy_global:
            # Hold current altitude until the EKF origin is available.
            self.publish_setpoint(0.0, 0.0, SEARCH_ALT_REL)
            return

        if self.substate == 'FLY_TO_ENTRY':
            self.handle_fly_to_entry()
        elif self.substate == 'ZIGZAG':
            self.handle_zigzag()
        elif self.substate == 'HOVER_TARGET':
            self.handle_hover_target()

    def handle_fly_to_entry(self):
        if not self.local_waypoints:
            self.local_waypoints = [
                latlon_to_local(self.local_pos, lat, lon, SEARCH_ALT_REL)
                for (lat, lon) in self.raw_waypoints
            ]

        entry_x, entry_y, entry_z = latlon_to_local(self.local_pos, *CORNER_BR, SEARCH_ALT_REL)
        self.publish_setpoint(entry_x, entry_y, entry_z)

        dx = self.local_pos.x - entry_x
        dy = self.local_pos.y - entry_y
        dz = self.local_pos.z - entry_z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < ACCEPTANCE_RADIUS_M:
            self.get_logger().info('-- Box entry reached, starting zigzag pattern')
            self.wp_index = 0
            self.substate = 'ZIGZAG'

    def handle_zigzag(self):
        if self.target_found:
            self.get_logger().info('-- Red target detected! Interrupting zigzag to hover and center.')
            self.saved_zigzag_wp_index = self.wp_index
            self.hover_setpoint = (self.local_pos.x, self.local_pos.y, SEARCH_ALT_REL)
            self.hover_stable_count = 0
            self.hover_lost_count = 0
            self.hover_tick_count = 0
            self.substate = 'HOVER_TARGET'
            return

        if self.wp_index >= len(self.local_waypoints):
            self.get_logger().info('-- Zigzag pattern complete with no detection, handing off to payload release')
            self.finish()
            return

        x, y, z = self.local_waypoints[self.wp_index]

        if self.wp_index == self._last_logged_wp + 1:
            self.get_logger().info(
                f'Zigzag leg {self.wp_index + 1}/{len(self.local_waypoints)}')
            self._last_logged_wp = self.wp_index

        self.publish_setpoint(x, y, z)

        dx = self.local_pos.x - x
        dy = self.local_pos.y - y
        dz = self.local_pos.z - z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < ACCEPTANCE_RADIUS_M:
            self.wp_index += 1

    def report_target_location(self):
        """
        Called once we're hovering (roughly) over the target. Since the
        camera is assumed nadir, the vehicle's own current lat/lon is a
        reasonable estimate of the target's real-world location.
        """
        lat, lon = local_to_latlon(self.local_pos, self.local_pos.x, self.local_pos.y)
        alt_agl = -self.local_pos.z   # NED z is negative-up
        self.get_logger().info(
            f'>>> TARGET LOCATION: lat={lat:.7f}, lon={lon:.7f}, alt={alt_agl:.1f}m AGL')

        msg = NavSatFix()
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt_agl
        self.target_loc_pub.publish(msg)

    def handle_hover_target(self):
        self.hover_tick_count += 1

        if not self.target_found:
            self.hover_lost_count += 1
            # keep holding the last commanded position while detection is briefly lost
            self.publish_setpoint(*self.hover_setpoint)
            if self.hover_lost_count > HOVER_LOST_TICKS:
                self.get_logger().warn('-- Target lost during hover, resuming zigzag pattern')
                self.wp_index = self.saved_zigzag_wp_index
                self.substate = 'ZIGZAG'
            return

        self.hover_lost_count = 0

        if self.hover_tick_count > HOVER_MAX_TICKS:
            self.get_logger().warn('-- Hover centering timed out, proceeding with best-effort position')
            self.report_target_location()
            self.finish()
            return

        if self.target_pixel is None:
            self.publish_setpoint(*self.hover_setpoint)
            return

        err_x_px = self.target_pixel.x - (IMG_WIDTH / 2.0)
        err_y_px = self.target_pixel.y - (IMG_HEIGHT / 2.0)

        if abs(err_x_px) < PIXEL_CENTER_TOL_PX and abs(err_y_px) < PIXEL_CENTER_TOL_PX:
            self.hover_stable_count += 1
        else:
            self.hover_stable_count = 0
            x, y, z = self.hover_setpoint
            # Rough nadir-camera assumption: image-y (down/up in frame) maps to
            # body-x (fwd/back), image-x maps to body-y (right/left). Flip signs
            # here if it drifts the wrong way on your camera mount.
            x += err_y_px * VISUAL_SERVO_GAIN_M_PER_PX
            y += err_x_px * VISUAL_SERVO_GAIN_M_PER_PX
            self.hover_setpoint = (x, y, z)

        self.publish_setpoint(*self.hover_setpoint)

        if self.hover_stable_count >= HOVER_STABLE_TICKS:
            self.get_logger().info('-- Target centered and stable, hover complete -- handing off to payload release')
            self.report_target_location()
            self.finish()

    def finish(self):
        self.active = False
        if self.timer is not None:
            self.timer.cancel()
        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)


# ============================================================
# Phase 4: PayloadReleaseNode (was payload_release_node.py)
# ============================================================

MAV_CMD_DO_GRIPPER = 211
GRIPPER_ACTION_RELEASE = 0
VEHICLE_CMD_RESULT_ACCEPTED = 0


class PayloadReleaseNode(Node):

    def __init__(self):
        super().__init__('payload_release_node')

        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', pub_qos())
        self.create_subscription(
            VehicleCommandAck, '/fmu/out/vehicle_command_ack', self.ack_cb, sub_qos())

        self.done_pub = self.create_publisher(Bool, '/mission/payload_done', handoff_qos())
        self.create_subscription(Bool, '/mission/search_done', self.trigger_cb, handoff_qos())

        self.state = 'IDLE'   # IDLE -> SEND_RELEASE -> WAIT_ACK -> DONE
        self.tick_counter = 0
        self.release_ack_ok = None
        self.timer = None

        self.get_logger().info('PayloadReleaseNode ready, waiting for search_done...')

    # ------------------------------------------------------------------
    def trigger_cb(self, msg: Bool):
        if msg.data and self.state == 'IDLE':
            self.state = 'SEND_RELEASE'
            self.get_logger().info('-- Search complete, triggering payload release')
            self.timer = self.create_timer(0.05, self.timer_cb)  # 20 Hz

    def ack_cb(self, msg: VehicleCommandAck):
        if msg.command != MAV_CMD_DO_GRIPPER or self.state != 'WAIT_ACK':
            return
        self.release_ack_ok = (msg.result == VEHICLE_CMD_RESULT_ACCEPTED)
        if self.release_ack_ok:
            self.get_logger().info('PX4 ACCEPTED gripper release command.')
        else:
            self.get_logger().error(f'PX4 REJECTED gripper release command (result={msg.result}).')

    def timer_cb(self):
        if self.state == 'SEND_RELEASE':
            self.send_gripper_release()
            self.state = 'WAIT_ACK'
            self.tick_counter = 0

        elif self.state == 'WAIT_ACK':
            self.tick_counter += 1
            if self.release_ack_ok is True:
                self.finish(success=True, reason='PX4 confirmed the gripper release.')
            elif self.release_ack_ok is False:
                self.finish(success=False, reason='PX4 rejected the gripper release command.')
            elif self.tick_counter >= ACK_TIMEOUT_TICKS:
                self.finish(success=False, reason='No response from PX4 (timed out). '
                                                     'Is SITL/hardware running, the DDS agent connected, '
                                                     'and PD_GRIPPER_EN=1 set?')

    def send_gripper_release(self):
        msg = VehicleCommand()
        msg.command = MAV_CMD_DO_GRIPPER
        msg.param1 = float(GRIPPER_INSTANCE)
        msg.param2 = float(GRIPPER_ACTION_RELEASE)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)
        self.get_logger().info(f'Sent DO_GRIPPER release (instance={GRIPPER_INSTANCE}), awaiting ack...')

    def finish(self, success, reason):
        self.state = 'DONE'
        if self.timer is not None:
            self.timer.cancel()
        if success:
            self.get_logger().info('-- PAYLOAD RELEASED, handing off to RTL')
        else:
            self.get_logger().error(f'-- PAYLOAD RELEASE FAILED: {reason} -- proceeding to RTL anyway')
        msg = Bool()
        msg.data = True   # always hand off, even on failure, so the mission still lands
        self.done_pub.publish(msg)


# ============================================================
# Phase 5: RtlNode (was rtl_node.py)
# ============================================================

class RtlNode(Node):

    def __init__(self):
        super().__init__('rtl_node')

        self.vehiclecommand_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', pub_qos())

        self.done_pub = self.create_publisher(Bool, '/mission/rtl_sent', handoff_qos())
        self.create_subscription(Bool, '/mission/payload_done', self.trigger_cb, handoff_qos())

        self.sent = False
        self.get_logger().info('RtlNode ready, waiting for payload_done...')

    def trigger_cb(self, msg: Bool):
        if msg.data and not self.sent:
            self.sent = True
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
            self.get_logger().info('-- RTL commanded, mission complete')
            out = Bool()
            out.data = True
            self.done_pub.publish(out)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehiclecommand_pub.publish(msg)


# ============================================================
# Launcher (was mission_launch.py) -- starts all five together
# in one process using a multi-threaded executor.
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    nodes = [
        TakeoffNode(),
        EnduranceLapNode(),
        SearchZigzagNode(),
        PayloadReleaseNode(),
        RtlNode(),
    ]

    executor = MultiThreadedExecutor(num_threads=len(nodes) + 2)
    for n in nodes:
        executor.add_node(n)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for n in nodes:
            n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
