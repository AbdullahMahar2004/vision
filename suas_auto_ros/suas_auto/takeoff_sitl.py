import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleAttitude
import math

class TakeoffNode(Node):
    def __init__(self):
        super().__init__('takeoff_node')
        self.yaw_sub = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.subscribe_vehicle_attitude, rclpy.qos.qos_profile_sensor_data)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehiclecommand_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.counter = 0
        self.current_yaw = 0.0
        self.create_timer(0.1, self.timer_callback)

    def subscribe_vehicle_attitude(self, vehicle_attitude):
        self.current_yaw = math.atan2( 2*(vehicle_attitude.q[0]*vehicle_attitude.q[3] + vehicle_attitude.q[1]*vehicle_attitude.q[2]), 1 - 2*(vehicle_attitude.q[2]**2 + vehicle_attitude.q[3]**2) )
        self.get_logger().info(f'Current yaw read: {math.degrees(self.current_yaw):.2f} degrees')

    def publish_vehicle_command(self, command, param1=0.0 , param2=0.0):
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
        sp.position = [0.0, 0.0, -10.0]
        sp.yaw = self.current_yaw
        if self.counter % 10 == 0:
            self.get_logger().info(f'Setpoint yaw: {math.degrees(self.current_yaw):.2f} degrees')
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
    node = TakeoffNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()