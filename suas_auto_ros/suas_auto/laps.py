import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
import math


# ---------------------------------------------------------------------------
# WAYPOINT LAP  (Phase 1) — from search_waypoints.txt
# Rectangle A -> B -> C -> D -> center -> back to A
# ---------------------------------------------------------------------------
LAP_WAYPOINTS = [
    (29.9871, 30.8351),   # A — corner (start)
    (29.9871, 30.8361),   # B — corner
    (29.9881, 30.8361),   # C — corner
    (29.9881, 30.8351),   # D — corner
    (29.9876, 30.8356),   # center point
    (29.9871, 30.8351),   # back to A — closes the loop
]

# ---------------------------------------------------------------------------
# LAWNMOWER SEARCH AREA  (Phases 3-4) — outer corners from search_waypoints.txt
# ---------------------------------------------------------------------------
CORNER_BL = (29.9871, 30.8351)   # A
CORNER_BR = (29.9871, 30.8361)   # B
CORNER_TL = (29.9881, 30.8351)   # D
CORNER_TR = (29.9881, 30.8361)   # C

TARGET_ALT   = -4.0   # climb altitude, NED (negative = up) — from search_waypoints.txt alt=10
STEP_SIZE    =   6.0   # lawnmower row spacing, meters
WP_THRESHOLD =   0.5   # "waypoint reached" radius, meters


def gps_to_ned(lat, lon, ref_lat, ref_lon):
    north_m = (lat - ref_lat) * 111139.0
    east_m  = (lon - ref_lon) * 111139.0 * math.cos(math.radians(ref_lat))
    return north_m, east_m


def generate_lawnmower(bl_ned, br_ned, tl_ned, step_size):
    bl_n, bl_e = bl_ned
    br_n, br_e = br_ned
    tl_n, tl_e = tl_ned

    total_north = tl_n - bl_n

    num_rows = max(1, int(total_north / step_size))

    waypoints = []
    for row in range(num_rows + 1):
        current_north = bl_n + row * step_size
        if current_north > tl_n:
            current_north = tl_n

        if row % 2 == 0:
            start_e, end_e = bl_e, br_e
        else:
            start_e, end_e = br_e, bl_e

        waypoints.append([current_north, start_e])
        waypoints.append([current_north, end_e])

    return waypoints, num_rows


class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')

        self.yaw_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude',
            self.subscribe_vehicle_attitude, rclpy.qos.qos_profile_sensor_data
        )
        self.pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self.subscribe_vehicle_position, rclpy.qos.qos_profile_sensor_data
        )

        self.offboard_pub       = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_pub     = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehiclecommand_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.counter     = 0
        self.current_yaw = 0.0
        self.ref_lat     = 0.0
        self.ref_lon     = 0.0
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_z   = 0.0
        self.ref_valid   = False

        # Phase 0 — climb
        self.alt_stable_count = 0

        # Phase 1 — waypoint lap
        self.lap_ned   = None   # filled in once ref_lat/lon are known
        self.lap_index = 0

        # Phase 3/4 — lawnmower
        self.waypoints    = []
        self.wp_index      = 0
        self.total_rows    = 0
        self.mission_done  = False

        self.phase = 0

        self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('=' * 50)
        self.get_logger().info('  MISSION NODE STARTED')
        self.get_logger().info(f'  Target altitude : {TARGET_ALT} m NED')
        self.get_logger().info(f'  Lap waypoints   : {len(LAP_WAYPOINTS)}')
        self.get_logger().info(f'  Step size       : {STEP_SIZE} m')
        self.get_logger().info('=' * 50)

    # -----------------------------------------------------------------
    # Subscribers
    # -----------------------------------------------------------------
    def subscribe_vehicle_attitude(self, msg):
        self.current_yaw = math.atan2(
            2 * (msg.q[0] * msg.q[3] + msg.q[1] * msg.q[2]),
            1 - 2 * (msg.q[2] ** 2 + msg.q[3] ** 2)
        )

    def subscribe_vehicle_position(self, msg):
        self.ref_lat   = msg.ref_lat
        self.ref_lon   = msg.ref_lon
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_z = msg.z

        if abs(self.ref_lat) > 0.001 and abs(self.ref_lon) > 0.001:
            self.ref_valid = True

    # -----------------------------------------------------------------
    # Commands
    # -----------------------------------------------------------------
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = param1
        msg.param2           = param2
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)
        self.vehiclecommand_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info('=' * 50)
        self.get_logger().info('  ARM COMMAND SENT')
        self.get_logger().info('=' * 50)

    def rtl(self):
        self.get_logger().info('=' * 50)
        self.get_logger().info('  MISSION COMPLETE — RETURNING TO LAUNCH (RTL)')
        self.get_logger().info('=' * 50)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RTL)
        self.mission_done = True

    def distance_to_wp(self, wp):
        dn = wp[0] - self.current_x
        de = wp[1] - self.current_y
        return math.sqrt(dn * dn + de * de)

    def build_lawnmower_waypoints(self):
        self.get_logger().info('=' * 50)
        self.get_logger().info('  BUILDING LAWNMOWER WAYPOINTS')
        self.get_logger().info('=' * 50)

        bl_n, bl_e = gps_to_ned(CORNER_BL[0], CORNER_BL[1], self.ref_lat, self.ref_lon)
        br_n, br_e = gps_to_ned(CORNER_BR[0], CORNER_BR[1], self.ref_lat, self.ref_lon)
        tl_n, tl_e = gps_to_ned(CORNER_TL[0], CORNER_TL[1], self.ref_lat, self.ref_lon)

        wps, num_rows = generate_lawnmower((bl_n, bl_e), (br_n, br_e), (tl_n, tl_e), STEP_SIZE)
        self.waypoints  = wps
        self.total_rows = num_rows

        self.get_logger().info(f'  Total rows      : {num_rows}')
        self.get_logger().info(f'  Total waypoints : {len(wps)}')
        self.get_logger().info('=' * 50)

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    def timer_callback(self):
        ocm = OffboardControlMode(position=True)
        ocm.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(ocm)

        if self.counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.get_logger().info('  OFFBOARD MODE SET')
            self.arm()

        sp = TrajectorySetpoint()
        sp.yaw = self.current_yaw

        # ---------------- Phase 0 — climb to altitude ----------------
        if self.phase == 0:
            sp.position = [0.0, 0.0, TARGET_ALT]
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

            if self.counter % 10 == 0:
                self.get_logger().info(f'[Phase 0] Climbing — current_z={self.current_z:.2f}m target={TARGET_ALT:.1f}m')

            if abs(self.current_z - TARGET_ALT) < 0.5:
                self.alt_stable_count += 1
                if self.alt_stable_count >= 10:
                    self.get_logger().info('  ALTITUDE REACHED — switching to Phase 1 (waypoint lap)')
                    self.phase = 1
            else:
                self.alt_stable_count = 0

        # ---------------- Phase 1 — waypoint lap ----------------
        elif self.phase == 1:
            if not self.ref_valid:
                sp.position = [0.0, 0.0, TARGET_ALT]
                sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(sp)
                if self.counter % 10 == 0:
                    self.get_logger().info('[Phase 1] Waiting for GPS origin before starting lap...')
            else:
                if self.lap_ned is None:
                    self.lap_ned = [gps_to_ned(lat, lon, self.ref_lat, self.ref_lon) for lat, lon in LAP_WAYPOINTS]
                    self.get_logger().info(f'  Lap NED points: {self.lap_ned}')

                wp = self.lap_ned[self.lap_index]
                sp.position = [wp[0], wp[1], TARGET_ALT]
                sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(sp)

                dist = self.distance_to_wp(wp)
                if self.counter % 10 == 0:
                    self.get_logger().info(
                        f'[Phase 1] Lap WP {self.lap_index + 1}/{len(self.lap_ned)} | dist={dist:.2f}m'
                    )

                if dist < WP_THRESHOLD:
                    self.get_logger().info(f'  LAP WAYPOINT {self.lap_index + 1} REACHED')
                    self.lap_index += 1
                    if self.lap_index >= len(self.lap_ned):
                        self.get_logger().info('  LAP COMPLETE — switching to Phase 2 (search area entry)')
                        self.phase = 2

        # ---------------- Phase 2 — confirm GPS origin / entry ----------------
        elif self.phase == 2:
            sp.position = [0.0, 0.0, TARGET_ALT]
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

            self.build_lawnmower_waypoints()
            self.get_logger().info('  ENTERING SEARCH AREA — switching to Phase 3')
            self.phase = 3

        # ---------------- Phase 3 — fly lawnmower pattern ----------------
        elif self.phase == 3:
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info('  LAWNMOWER COMPLETE — switching to Phase 4 (RTL)')
                self.phase = 4
            else:
                wp = self.waypoints[self.wp_index]
                sp.position = [wp[0], wp[1], TARGET_ALT]
                sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(sp)

                dist = self.distance_to_wp(wp)
                if self.counter % 10 == 0:
                    row_num = self.wp_index // 2
                    self.get_logger().info(
                        f'[Phase 3] WP {self.wp_index:02d}/{len(self.waypoints) - 1} | '
                        f'Row {row_num}/{self.total_rows} | dist={dist:.2f}m'
                    )

                if dist < WP_THRESHOLD:
                    self.get_logger().info(f'  WAYPOINT {self.wp_index:02d} REACHED')
                    self.wp_index += 1

        # ---------------- Phase 4 — RTL ----------------
        elif self.phase == 4:
            if not self.mission_done:
                self.rtl()

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
