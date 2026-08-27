import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleCommand

class PayloadNode(Node):
    def __init__(self):
        super().__init__('payload_node')
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehiclecommand_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.counter = 0
        self.released = False

        self.create_timer(0.1, self.timer_callback)


        
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
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)


        if self.counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.get_logger().info('='*40)  
            self.get_logger().info('  OFFBOARD MODE SET + ARM COMMANDED')
            self.get_logger().info('='*40)
            self.arm()

        if self.counter == 30 and self.released == False:
            self.release_payload()

        self.counter += 1


    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info('='*40)
        self.get_logger().info('  PAYLOAD ARM ')
        self.get_logger().info('='*40)

    def release_payload(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR, 1.0, -1.0)
        self.get_logger().info('='*40)
        self.get_logger().info(' PAYLOAD RELEASED')
        self.get_logger().info('='*40)
        self.released = True



def main(args=None):
    rclpy.init(args=args)
    node = PayloadNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()