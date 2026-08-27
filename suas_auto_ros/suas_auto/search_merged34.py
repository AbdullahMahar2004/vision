import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)


class BoxZigzagMission(Node):
    """
    Self-contained mission: no search_area_monitor.py, no
    waypoint_mission_node.py -- this node owns the whole flight by itself.

    Only mission-defining input is the four box corners below. Flow:

      TAKEOFF      -> arm, switch to OFFBOARD, climb straight up to
                       SEARCH_ALT_REL above the arm point.
      FLY_TO_ENTRY -> fly to CORNER_BR (box entry corner) at that altitude.
      ZIGZAG       -> fly the lawnmower pattern generated from the 4
                       corners (same generator/fix as zigzag10.py).
      RTL          -> command PX4's own Return-To-Launch and stop
                       publishing OFFBOARD setpoints so RTL can take over.
      DONE         -> idle, publishes nothing further.
    """

    # ---------------- CONFIG (edit only these for a new box/altitude) ----------------
    SEARCH_ALT_REL = -4.0      # cruise/search altitude, 4m AGL (NED: negative = up)
    ACCEPTANCE_RADIUS_M = 2.0
    ALT_ACCEPTANCE_M = 0.5     # how close to target altitude counts as "arrived" during takeoff

    NUM_LANES = 4              # fixed lane count; set to None to use LANE_SPACING_M instead
    LANE_SPACING_M = 3.0       # only used if NUM_LANES is None

    # Box corners -- edit these four to move/resize the search area.
    CORNER_BR = (31.310491, 30.065188)   # entry corner -- mission flies here first
    CORNER_TR = (31.310422, 30.065244)
    CORNER_BL = (31.310500, 30.065160)
    CORNER_TL = (31.310453, 30.065307)   # exit corner -- pattern ends here

    EARTH_RADIUS_M = 6371000.0

    # PX4 offboard needs >=10 setpoints streamed before it will accept the
    # mode switch -- same warm-up count takeoff_sitl22.py used.
    OFFBOARD_WARMUP_TICKS = 10

    def __init__(self):
        super().__init__('box_zigzag_mission')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.vehiclecommand_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_cb, qos)

        self.local_pos = VehicleLocalPosition()
        self.have_local_ref = False

        self.raw_waypoints = self.generate_zigzag(
            self.CORNER_BL, self.CORNER_BR, self.CORNER_TL, self.CORNER_TR,
            num_lanes=self.NUM_LANES, lane_spacing=self.LANE_SPACING_M)
        self.local_waypoints = []   # filled in once we have ref_lat/ref_lon
        self.wp_index = 0

        self.offboard_setpoint_counter = 0
        self.armed_and_offboard_sent = False

        # TAKEOFF -> FLY_TO_ENTRY -> ZIGZAG -> RTL -> DONE
        self.state = 'TAKEOFF'

        self.create_timer(0.1, self.timer_cb)  # 10 Hz
        self.get_logger().info(
            f'BoxZigzagMission ready. {len(self.raw_waypoints)} zigzag waypoints '
            f'generated, target alt {-self.SEARCH_ALT_REL:.1f}m AGL. Starting TAKEOFF...')

    # ------------------------------------------------------------------
    @staticmethod
    def haversine_m(lat1, lon1, lat2, lon2):
        R = BoxZigzagMission.EARTH_RADIUS_M
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (math.sin(dphi / 2) ** 2
             + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(a))

    def generate_zigzag(self, bl, br, tl, tr, num_lanes=None, lane_spacing=None):
        total_height = self.haversine_m(bl[0], bl[1], tl[0], tl[1])

        if num_lanes is not None:
            num_lanes = max(1, int(num_lanes))
            self.get_logger().info(
                f'Search area height: {total_height:.1f}m -- {num_lanes} lanes '
                f'(fixed by NUM_LANES, ~{total_height / num_lanes:.2f}m/lane).')
        else:
            num_lanes = max(1, int(total_height / lane_spacing))
            self.get_logger().info(
                f'Search area height: {total_height:.1f}m -- {num_lanes} lanes '
                f'(auto-derived from {lane_spacing}m spacing).')

        waypoints = []
        # Start by going LEFT (BR -> BL first): the mission flies to
        # CORNER_BR itself in FLY_TO_ENTRY before the zigzag starts, so
        # the first zigzag leg must continue on from there, not re-visit
        # it. See zigzag10.py for the full explanation of this fix.
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

    def latlon_to_local(self, lat, lon):
        ref_lat = self.local_pos.ref_lat
        ref_lon = self.local_pos.ref_lon
        dlat = math.radians(lat - ref_lat)
        dlon = math.radians(lon - ref_lon)
        x = dlat * self.EARTH_RADIUS_M
        y = dlon * self.EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
        z = self.SEARCH_ALT_REL
        return x, y, z

    # ------------------------------------------------------------------
    def local_cb(self, msg: VehicleLocalPosition):
        self.local_pos = msg
        if msg.xy_global and msg.z_global:
            self.have_local_ref = True

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
        if self.state == 'TAKEOFF':
            self.handle_takeoff()
        elif self.state == 'FLY_TO_ENTRY':
            self.handle_fly_to_entry()
        elif self.state == 'ZIGZAG':
            self.handle_zigzag()
        elif self.state == 'RTL':
            self.handle_rtl()
        elif self.state == 'DONE':
            pass  # publishes nothing further

    def handle_takeoff(self):
        # Stream heartbeat + a straight-up setpoint before/while arming --
        # PX4 requires the OFFBOARD stream running for OFFBOARD_WARMUP_TICKS
        # before it will accept the mode switch.
        self.publish_offboard_heartbeat()
        self.publish_setpoint(0.0, 0.0, self.SEARCH_ALT_REL)

        if self.offboard_setpoint_counter == self.OFFBOARD_WARMUP_TICKS and not self.armed_and_offboard_sent:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)  # 6 = OFFBOARD
            self.arm()
            self.armed_and_offboard_sent = True
            self.get_logger().info('-- OFFBOARD mode + arm commands sent, climbing...')

        if self.offboard_setpoint_counter < self.OFFBOARD_WARMUP_TICKS:
            self.offboard_setpoint_counter += 1
            return

        if self.armed_and_offboard_sent and abs(self.local_pos.z - self.SEARCH_ALT_REL) < self.ALT_ACCEPTANCE_M:
            self.get_logger().info(f'-- Target altitude reached ({-self.SEARCH_ALT_REL:.1f}m AGL), flying to box entry')
            self.state = 'FLY_TO_ENTRY'

    def handle_fly_to_entry(self):
        self.publish_offboard_heartbeat()

        if not self.have_local_ref:
            # Hold current altitude until the EKF origin is available.
            self.publish_setpoint(0.0, 0.0, self.SEARCH_ALT_REL)
            return

        if not self.local_waypoints:
            self.local_waypoints = [
                self.latlon_to_local(lat, lon) for (lat, lon) in self.raw_waypoints
            ]

        entry_x, entry_y, entry_z = self.latlon_to_local(*self.CORNER_BR)
        self.publish_setpoint(entry_x, entry_y, entry_z)

        dx = self.local_pos.x - entry_x
        dy = self.local_pos.y - entry_y
        dz = self.local_pos.z - entry_z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < self.ACCEPTANCE_RADIUS_M:
            self.get_logger().info('-- Box entry reached, starting zigzag pattern')
            self.wp_index = 0
            self.state = 'ZIGZAG'

    def handle_zigzag(self):
        self.publish_offboard_heartbeat()

        if self.wp_index >= len(self.local_waypoints):
            self.get_logger().info('-- Zigzag pattern complete, commanding RTL')
            self.state = 'RTL'
            return

        x, y, z = self.local_waypoints[self.wp_index]

        if self.wp_index == getattr(self, '_last_logged_wp', -1) + 1:
            self.get_logger().info(
                f'Zigzag leg {self.wp_index + 1}/{len(self.local_waypoints)}')
            self._last_logged_wp = self.wp_index

        self.publish_setpoint(x, y, z)

        dx = self.local_pos.x - x
        dy = self.local_pos.y - y
        dz = self.local_pos.z - z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < self.ACCEPTANCE_RADIUS_M:
            self.wp_index += 1

    def handle_rtl(self):
        # Command RTL once, then go quiet -- do NOT keep publishing OFFBOARD
        # setpoints, otherwise the vehicle can fall straight back into
        # OFFBOARD mode instead of executing RTL.
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self.get_logger().info('-- RTL commanded, mission node going idle')
        self.state = 'DONE'


def main(args=None):
    rclpy.init(args=args)
    node = BoxZigzagMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
