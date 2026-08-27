import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
from pyproj import Transformer
import math

class WaypointNode(Node):
    def __init__(self):
        super().__init__('waypoint_node')
        self.yaw_sub = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.subscribe_vehicle_attitude, rclpy.qos.qos_profile_sensor_data)
        self.vehicle_pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.subscribe_vehicle_position, rclpy.qos.qos_profile_sensor_data)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehiclecommand_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.counter = 0
        self.current_yaw = 0.0
        self.ref_lat = 0.0
        self.ref_lon = 0.0
        self.current_z = 0.0
        self.alt_reached_count = 0

        self.target_lat = 31.310279
        self.target_lon = 30.065369
        self.target_alt = -10.0

        self.phase = 0
        self.waypoint_ned = None

        self.create_timer(0.1, self.timer_callback)

    def subscribe_vehicle_attitude(self, vehicle_attitude):
        self.current_yaw = math.atan2(
            2*(vehicle_attitude.q[0]*vehicle_attitude.q[3] + vehicle_attitude.q[1]*vehicle_attitude.q[2]),
            1 - 2*(vehicle_attitude.q[2]**2 + vehicle_attitude.q[3]**2)
        )

    def subscribe_vehicle_position(self, vehicle_local_position):
        self.ref_lat = vehicle_local_position.ref_lat
        self.ref_lon = vehicle_local_position.ref_lon
        self.current_z = vehicle_local_position.z

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

    def timer_callback(self):
        msg = OffboardControlMode(position=True)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)
        sp = TrajectorySetpoint()

        if self.phase == 0 and abs(self.current_z + 10.0) > 0.5:
            self.alt_reached_count = 0
            sp.position = [0.0, 0.0, -10.0]
            sp.yaw = self.current_yaw
            if self.counter % 10 == 0:
                self.get_logger().info(f'Phase 0 — current_z: {self.current_z:.2f}m')
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

        elif self.phase == 0:
            self.alt_reached_count += 1
            self.get_logger().info(f'Altitude reached count: {self.alt_reached_count}')
            sp.position = [0.0, 0.0, -10.0]
            sp.yaw = self.current_yaw
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)
            if self.alt_reached_count >= 5:
                self.get_logger().info('Switching to phase 1')
                self.phase = 1

        elif self.phase == 1:
            if self.waypoint_ned is None:
                north_m = (self.target_lat - self.ref_lat) * 111139.0
                east_m = (self.target_lon - self.ref_lon) * 111139.0 * math.cos(math.radians(self.ref_lat))
                self.waypoint_ned = [north_m, east_m, self.target_alt]
                self.get_logger().info(f'Waypoint NED: north={north_m:.2f}m, east={east_m:.2f}m')
            sp.position = self.waypoint_ned
            sp.yaw = self.current_yaw
            if self.counter % 10 == 0:
                self.get_logger().info(f'Phase 1 — flying to waypoint')
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(sp)

        if self.counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.arm()

        self.counter += 1

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()