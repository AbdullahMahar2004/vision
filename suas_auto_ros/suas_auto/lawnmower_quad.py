import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
import math


# ---------------------------------------------------------------------------
# WAYPOINT LAP  (Phase 1)
# ---------------------------------------------------------------------------
LAP_WAYPOINTS = [
    (31.3104510, 30.0653845)
]

# ---------------------------------------------------------------------------
# LAWNMOWER SEARCH AREA  (Phases 3-4)
# ---------------------------------------------------------------------------
CORNER_BL = (31.3104682, 30.0656138)
CORNER_BR = (31.3103262, 30.0653523)
CORNER_TL = (31.3102987, 30.0657519)
CORNER_TR = (31.3101428, 30.0654917)

TARGET_ALT   = -5.0   # climb altitude, NED (negative = up)
STEP_SIZE    =  2.0   # lawnmower row spacing, meters
WP_THRESHOLD =  0.2   # "waypoint reached" radius, meters


def gps_to_ned(lat, lon, ref_lat, ref_lon):
    north_m = (lat - ref_lat) * 111139.0
    east_m  = (lon - ref_lon) * 111139.0 * math.cos(math.radians(ref_lat))
    return north_m, east_m


def get_row_intersections(corners, north):
    intersections = []
    n = len(corners)
    for i in range(n):
        n1, e1 = corners[i]
        n2, e2 = corners[(i + 1) % n]
        if (n1 <= north <= n2) or (n2 <= north <= n1):
            if abs(n2 - n1) < 1e-9:
                continue
            t = (north - n1) / (n2 - n1)
            e = e1 + t * (e2 - e1)
            intersections.append(e)
    intersections.sort()
    return intersections


def generate_lawnmower(corners, step_size):
    norths = [c[0] for c in corners]
    min_north, max_north = min(norths), max(norths)

    num_rows = max(1, int((max_north - min_north) / step_size))

    waypoints = []
    for row in range(num_rows + 1):
        current_north = min_north + row * step_size
        if current_north > max_north:
            current_north = max_north

        intersections = get_row_intersections(corners, current_north)
        if len(intersections) < 2:
            continue

        start_e, end_e = intersections[0], intersections[-1]
        if row % 2 == 1:
            start_e, end_e = end_e, start_e

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
        self.trajectory_pub     = self.create_publisher(TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',  10)
        self.vehiclecommand_pub = self.create_publisher(VehicleCommand,      '/fmu/in/vehicle_command',      10)

        self.counter     = 0
        self.current_yaw = 0.0
        self.ref_lat     = 0.0
        self.ref_lon     = 0.0
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_z   = 0.0
        self.ref_valid   = False

        self.alt_stable_count = 0

        self.lap_ned   = None
        self.lap_index = 0

        self.waypoints   = []
        self.wp_index    = 0
        self.total_rows  = 0
        self.mission_done = False

        self.phase = 0

        self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('=' * 50)
        self.get_logger().info('  MISSION NODE STARTED')
        self.get_logger().info(f'  Target altitude : {TARGET_ALT} m NED')
        self.get_logger().info(f'  Lap waypoints   : {len(LAP_WAYPOINTS)}')
        self.get_logger().info(f'  Step size       : {STEP_SIZE} m')
        self.get_logger().info(f'  WP threshold    : {WP_THRESHOLD} m')
        self.get_logger().info('=' * 50)

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
        tr_n, tr_e = gps_to_ned(CORNER_TR[0], CORNER_TR[1], self.ref_lat, self.ref_lon)
        tl_n, tl_e = gps_to_ned(CORNER_TL[0], CORNER_TL[1], self.ref_lat, self.ref_lon)

        self.get_logger().info(f'  BL NED : north={bl_n:.2f} m, east={bl_e:.2f} m')
        self.get_logger().info(f'  BR NED : north={br_n:.2f} m, east={br_e:.2f} m')
        self.get_logger().info(f'  TL NED : north={tl_n:.2f} m, east={tl_e:.2f} m')
        self.get_logger().info(f'  TR NED : north={tr_n:.2f} m, east={tr_e:.2f} m')

        corners = [(bl_n, bl_e), (br_n, br_e), (tr_n, tr_e), (tl_n, tl_e)]
        wps, num_rows = generate_lawnmower(corners, STEP_SIZE)
        self.waypoints  = wps
        self.total_rows = num_rows

        self.get_logger().info(f'  Total rows      : {num_rows}')
        self.get_logger().info(f'  Total waypoints : {len(wps)}')
        self.get_logger().info('  Waypoint list:')
        for i, wp in enumerate(wps):
            self.get_logger().info(f'    WP {i:02d} — north={wp[0]:.2f} m, east={wp[1]:.2f} m')
        self.get_logger().info('=' * 50)

    def timer_callback(self):
        ocm = OffboardControlMode(position=True)
        ocm.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(ocm)

        if self.counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.get_logger().info('  OFFBOARD MODE SET')
            self.arm()

        sp = TrajectorySetpoint()

        # ---------------- Phase 0 — climb to altitude ----------------
        if self.phase == 0:
            sp.yaw = self.current_yaw
            sp.position  = [0.0, 0.0, TARGET_ALT]
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

            if self.counter % 10 == 0:
                self.get_logger().info(
                    f'[Phase 0] Climbing — current_z={self.current_z:.2f}m target={TARGET_ALT:.1f}m'
                )

            if abs(self.current_z - TARGET_ALT) < 0.5:
                self.alt_stable_count += 1
                if self.alt_stable_count >= 10:
                    self.get_logger().info('=' * 50)
                    self.get_logger().info('  ALTITUDE REACHED — switching to Phase 1 (waypoint lap)')
                    self.get_logger().info('=' * 50)
                    self.phase = 1
            else:
                self.alt_stable_count = 0

        # ---------------- Phase 1 — waypoint lap ----------------
        elif self.phase == 1:
            if not self.ref_valid:
                sp.yaw = self.current_yaw
                sp.position  = [0.0, 0.0, TARGET_ALT]
                sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(sp)
                if self.counter % 10 == 0:
                    self.get_logger().info('[Phase 1] Waiting for GPS origin...')
            else:
                if self.lap_ned is None:
                    self.lap_ned = [
                        gps_to_ned(lat, lon, self.ref_lat, self.ref_lon)
                        for lat, lon in LAP_WAYPOINTS
                    ]
                    self.get_logger().info(f'  Lap NED points: {self.lap_ned}')

                wp = self.lap_ned[self.lap_index]
                sp.yaw = math.atan2(wp[1] - self.current_y, wp[0] - self.current_x)
                sp.position  = [wp[0], wp[1], TARGET_ALT]
                sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(sp)

                dist = self.distance_to_wp(wp)
                if self.counter % 10 == 0:
                    self.get_logger().info(
                        f'[Phase 1] Lap WP {self.lap_index + 1}/{len(self.lap_ned)} | '
                        f'target=[N={wp[0]:.2f}, E={wp[1]:.2f}] | '
                        f'pos=[N={self.current_x:.2f}, E={self.current_y:.2f}] | '
                        f'dist={dist:.2f}m'
                    )

                if dist < WP_THRESHOLD:
                    self.get_logger().info(f'  LAP WAYPOINT {self.lap_index + 1} REACHED')
                    self.lap_index += 1
                    if self.lap_index >= len(self.lap_ned):
                        self.get_logger().info('=' * 50)
                        self.get_logger().info('  LAP COMPLETE — switching to Phase 2')
                        self.get_logger().info('=' * 50)
                        self.phase = 2

        # ---------------- Phase 2 — build waypoints and enter search area ----------------
        elif self.phase == 2:
            sp.yaw = self.current_yaw
            sp.position  = [0.0, 0.0, TARGET_ALT]
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

            self.build_lawnmower_waypoints()
            self.get_logger().info('  ENTERING SEARCH AREA — switching to Phase 3')
            self.phase = 3

        # ---------------- Phase 3 — fly lawnmower pattern ----------------
        elif self.phase == 3:
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info('=' * 50)
                self.get_logger().info('  LAWNMOWER COMPLETE — switching to Phase 4 (RTL)')
                self.get_logger().info('=' * 50)
                self.phase = 4
            else:
                wp = self.waypoints[self.wp_index]
                sp.yaw = math.atan2(wp[1] - self.current_y, wp[0] - self.current_x)
                sp.position  = [wp[0], wp[1], TARGET_ALT]
                sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(sp)

                dist = self.distance_to_wp(wp)
                if self.counter % 10 == 0:
                    row_num = self.wp_index // 2
                    self.get_logger().info(
                        f'[Phase 3] WP {self.wp_index:02d}/{len(self.waypoints) - 1} | '
                        f'Row {row_num}/{self.total_rows} | '
                        f'target=[N={wp[0]:.2f}, E={wp[1]:.2f}] | '
                        f'pos=[N={self.current_x:.2f}, E={self.current_y:.2f}] | '
                        f'dist={dist:.2f}m'
                    )

                if dist < WP_THRESHOLD:
                    self.get_logger().info('=' * 50)
                    self.get_logger().info(f'  WAYPOINT {self.wp_index:02d} REACHED')
                    self.get_logger().info(f'  north={wp[0]:.2f} m, east={wp[1]:.2f} m')
                    self.get_logger().info('=' * 50)
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
