import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
import math

# ─────────────────────────────────────────────
# SEARCH AREA — 4 GPS corners (decimal degrees)
# Replace these with your actual competition coordinates
# ─────────────────────────────────────────────
CORNER_BR = (31.310491, 30.065188)
CORNER_TR = (31.310422, 30.065244)
CORNER_BL = (31.310500, 30.065160)
CORNER_TL = (31.310453, 30.065307)  

# ─────────────────────────────────────────────
# MISSION PARAMETERS
# ─────────────────────────────────────────────
TARGET_ALT   = -5.0    # NED altitude in meters (negative = up), so 5m AGL
STEP_SIZE    =  0.5    # meters between lawnmower rows
WP_THRESHOLD =  0.5    # meters — distance to consider a waypoint reached


def gps_to_ned(lat, lon, ref_lat, ref_lon):
    """
    Flat-earth GPS → NED conversion.
    Returns (north_m, east_m) relative to ref_lat/ref_lon.
    North is positive northward, East is positive eastward.
    """
    north_m = (lat - ref_lat) * 111139.0
    east_m  = (lon - ref_lon) * 111139.0 * math.cos(math.radians(ref_lat))
    return north_m, east_m


def generate_lawnmower(bl_ned, br_ned, tl_ned, step_size):
    """
    Generate an ordered list of NED [north, east] waypoints
    for a boustrophedon (lawnmower) pattern.

    Entry:  bottom-left corner
    Row 0:  BL → BR  (west to east)
    Step north by step_size
    Row 1:  BR → BL  (east to west)
    Step north by step_size
    ...repeat until top of area...

    Returns: list of [north_m, east_m] waypoints (no altitude — added later)
    """
    bl_n, bl_e = bl_ned
    br_n, br_e = br_ned
    tl_n, tl_e = tl_ned

    # Total north span and east span
    total_north = tl_n - bl_n   # positive (northward)
    east_span   = br_e - bl_e   # positive (eastward)

    # Number of rows
    num_rows = max(1, int(total_north / step_size))

    waypoints = []

    for row in range(num_rows + 1):
        current_north = bl_n + row * step_size
        # Clamp to top boundary
        if current_north > tl_n:
            current_north = tl_n

        if row % 2 == 0:
            # Even row: west → east (BL side → BR side)
            start_e = bl_e
            end_e   = br_e
        else:
            # Odd row: east → west (BR side → BL side)
            start_e = br_e
            end_e   = bl_e

        waypoints.append([current_north, start_e])
        waypoints.append([current_north, end_e])

    return waypoints, num_rows


class LawnmowerNode(Node):

    def __init__(self):
        super().__init__('lawnmower_node')

        # ── Subscriptions ──────────────────────────────────────────────────
        self.yaw_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.subscribe_vehicle_attitude,
            rclpy.qos.qos_profile_sensor_data
        )
        self.pos_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.subscribe_vehicle_position,
            rclpy.qos.qos_profile_sensor_data
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self.offboard_pub    = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_pub  = self.create_publisher(TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',  10)
        self.vehiclecommand_pub = self.create_publisher(VehicleCommand,   '/fmu/in/vehicle_command',      10)

        # ── State ──────────────────────────────────────────────────────────
        self.counter      = 0
        self.current_yaw  = 0.0
        self.ref_lat      = 0.0
        self.ref_lon      = 0.0
        self.current_x    = 0.0   # NED north
        self.current_y    = 0.0   # NED east
        self.current_z    = 0.0   # NED down (negative = up)

        self.phase            = 0
        self.alt_stable_count = 0
        self.ref_valid        = False
        self.waypoints        = []
        self.wp_index         = 0
        self.total_rows       = 0
        self.mission_done     = False

        # ── Timer 10 Hz ────────────────────────────────────────────────────
        self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('=' * 50)
        self.get_logger().info('  LAWNMOWER NODE STARTED')
        self.get_logger().info(f'  Target altitude : {TARGET_ALT} m NED ({abs(TARGET_ALT)} m AGL)')
        self.get_logger().info(f'  Step size       : {STEP_SIZE} m')
        self.get_logger().info(f'  WP threshold    : {WP_THRESHOLD} m')
        self.get_logger().info('=' * 50)

    # ── Callbacks ──────────────────────────────────────────────────────────

    def subscribe_vehicle_attitude(self, msg):
        self.current_yaw = math.atan2(
            2 * (msg.q[0] * msg.q[3] + msg.q[1] * msg.q[2]),
            1 - 2 * (msg.q[2] ** 2 + msg.q[3] ** 2)
        )

    def subscribe_vehicle_position(self, msg):
        self.ref_lat   = msg.ref_lat
        self.ref_lon   = msg.ref_lon
        self.current_x = msg.x    # north
        self.current_y = msg.y    # east
        self.current_z = msg.z    # down

        # ref_lat/ref_lon become valid once PX4 locks GPS origin
        if abs(self.ref_lat) > 0.001 and abs(self.ref_lon) > 0.001:
            self.ref_valid = True

    # ── VehicleCommand helper ──────────────────────────────────────────────

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

    def land(self):
        self.get_logger().info('=' * 50)
        self.get_logger().info('  ALL ROWS COMPLETE — LANDING')
        self.get_logger().info('=' * 50)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.mission_done = True

    # ── Waypoint generation ────────────────────────────────────────────────

    def build_waypoints(self):
        self.get_logger().info('=' * 50)
        self.get_logger().info('  BUILDING LAWNMOWER WAYPOINTS')
        self.get_logger().info('=' * 50)

        bl_n, bl_e = gps_to_ned(CORNER_BL[0], CORNER_BL[1], self.ref_lat, self.ref_lon)
        br_n, br_e = gps_to_ned(CORNER_BR[0], CORNER_BR[1], self.ref_lat, self.ref_lon)
        tl_n, tl_e = gps_to_ned(CORNER_TL[0], CORNER_TL[1], self.ref_lat, self.ref_lon)
        tr_n, tr_e = gps_to_ned(CORNER_TR[0], CORNER_TR[1], self.ref_lat, self.ref_lon)

        self.get_logger().info(f'  BL NED : north={bl_n:.2f} m, east={bl_e:.2f} m')
        self.get_logger().info(f'  BR NED : north={br_n:.2f} m, east={br_e:.2f} m')
        self.get_logger().info(f'  TL NED : north={tl_n:.2f} m, east={tl_e:.2f} m')
        self.get_logger().info(f'  TR NED : north={tr_n:.2f} m, east={tr_e:.2f} m')

        wps, num_rows = generate_lawnmower(
            (bl_n, bl_e), (br_n, br_e), (tl_n, tl_e), STEP_SIZE
        )
        self.waypoints   = wps
        self.total_rows  = num_rows

        self.get_logger().info(f'  Total rows      : {num_rows}')
        self.get_logger().info(f'  Total waypoints : {len(wps)}')
        self.get_logger().info('  Waypoint list:')
        for i, wp in enumerate(wps):
            self.get_logger().info(f'    WP {i:02d} — north={wp[0]:.2f} m, east={wp[1]:.2f} m')
        self.get_logger().info('=' * 50)

    # ── Distance helper ───────────────────────────────────────────────────

    def distance_to_wp(self, wp):
        dn = wp[0] - self.current_x
        de = wp[1] - self.current_y
        return math.sqrt(dn * dn + de * de)

    # ── Main timer loop ────────────────────────────────────────────────────

    def timer_callback(self):

        # Always publish OffboardControlMode to keep PX4 happy
        ocm = OffboardControlMode(position=True)
        ocm.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(ocm)

        # Arm + set offboard mode at counter=10
        if self.counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.get_logger().info('=' * 50)
            self.get_logger().info('  OFFBOARD MODE SET')
            self.get_logger().info('=' * 50)
            self.arm()

        sp = TrajectorySetpoint()
        sp.yaw = self.current_yaw

        # ── PHASE 0: Climb to target altitude ─────────────────────────────
        if self.phase == 0:
            sp.position = [0.0, 0.0, TARGET_ALT]
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

            if self.counter % 10 == 0:
                self.get_logger().info(
                    f'[Phase 0] Climbing — current_z={self.current_z:.2f} m  '
                    f'target={TARGET_ALT:.1f} m'
                )

            if abs(self.current_z - TARGET_ALT) < 0.5:
                self.alt_stable_count += 1
                if self.alt_stable_count >= 10:   # stable for 1 second
                    self.get_logger().info('=' * 50)
                    self.get_logger().info('  ALTITUDE REACHED — switching to Phase 1')
                    self.get_logger().info('=' * 50)
                    self.phase = 1
            else:
                self.alt_stable_count = 0

        # ── PHASE 1: Wait for GPS origin (ref_lat/ref_lon) ────────────────
        elif self.phase == 1:
            sp.position = [0.0, 0.0, TARGET_ALT]
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

            if self.counter % 10 == 0:
                self.get_logger().info(
                    f'[Phase 1] Waiting for GPS origin — '
                    f'ref_lat={self.ref_lat:.6f}, ref_lon={self.ref_lon:.6f}, '
                    f'valid={self.ref_valid}'
                )

            if self.ref_valid:
                self.get_logger().info('=' * 50)
                self.get_logger().info('  GPS ORIGIN VALID — switching to Phase 2')
                self.get_logger().info(f'  ref_lat={self.ref_lat:.7f}')
                self.get_logger().info(f'  ref_lon={self.ref_lon:.7f}')
                self.get_logger().info('=' * 50)
                self.phase = 2

        # ── PHASE 2: Build waypoints ───────────────────────────────────────
        elif self.phase == 2:
            sp.position = [0.0, 0.0, TARGET_ALT]
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

            self.build_waypoints()
            self.get_logger().info('  ENTERING SEARCH AREA — switching to Phase 3')
            self.phase = 3

        # ── PHASE 3: Fly lawnmower waypoints ──────────────────────────────
        elif self.phase == 3:
            if self.mission_done:
                return

            if self.wp_index >= len(self.waypoints):
                # All waypoints done
                self.land()
                return

            wp = self.waypoints[self.wp_index]
            sp.position  = [wp[0], wp[1], TARGET_ALT]
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

            dist = self.distance_to_wp(wp)

            if self.counter % 10 == 0:
                row_num = self.wp_index // 2
                self.get_logger().info(
                    f'[Phase 3] WP {self.wp_index:02d}/{len(self.waypoints)-1} | '
                    f'Row {row_num}/{self.total_rows} | '
                    f'target=[N={wp[0]:.2f}, E={wp[1]:.2f}] | '
                    f'pos=[N={self.current_x:.2f}, E={self.current_y:.2f}] | '
                    f'dist={dist:.2f} m'
                )

            if dist < WP_THRESHOLD:
                self.get_logger().info('=' * 50)
                self.get_logger().info(f'  WAYPOINT {self.wp_index:02d} REACHED')
                self.get_logger().info(f'  north={wp[0]:.2f} m, east={wp[1]:.2f} m')

                # Row transition logging
                if self.wp_index % 2 == 1:
                    # We just finished the END point of a row
                    row_num = self.wp_index // 2
                    direction = 'west→east' if row_num % 2 == 0 else 'east→west'
                    self.get_logger().info(f'  ROW {row_num} COMPLETE ({direction})')
                    if self.wp_index + 1 < len(self.waypoints):
                        self.get_logger().info(f'  Stepping north by {STEP_SIZE} m')

                # Corner logging
                if self.wp_index == 0:
                    self.get_logger().info('  >>> BOTTOM-LEFT CORNER REACHED <<<')
                elif self.wp_index == 1:
                    self.get_logger().info('  >>> BOTTOM-RIGHT CORNER REACHED <<<')
                elif self.wp_index == len(self.waypoints) - 2:
                    self.get_logger().info('  >>> TOP ENTRY CORNER REACHED <<<')
                elif self.wp_index == len(self.waypoints) - 1:
                    self.get_logger().info('  >>> FINAL CORNER REACHED — SEARCH COMPLETE <<<')

                self.get_logger().info('=' * 50)
                self.wp_index += 1

        # ── PHASE 4: Landing (mission_done=True, just keep publishing) ─────
        # PX4 handles the land command, we just stop sending setpoints
        # (OffboardControlMode is still published above to maintain comms)

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = LawnmowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
