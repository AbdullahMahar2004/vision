import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
import math

class TakeoffNode(Node):
    def __init__(self):
        super().__init__('takeoff_node')
        self.yaw_sub = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.subscribe_vehicle_attitude, rclpy.qos.qos_profile_sensor_data)
        self.alt_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.subscribe_vehicle_local_position, rclpy.qos.qos_profile_sensor_data)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehiclecommand_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.counter = 0
        self.current_yaw = 0.0
        self.current_altitude = 0.0
        self.altitude_reached = False
        self.land_started = False
        self.hover_start_time = None
        self.hover_count = 0

        self.create_timer(0.1, self.timer_callback)

    def subscribe_vehicle_attitude(self, vehicle_attitude):
        self.current_yaw = math.atan2( 2*(vehicle_attitude.q[0]*vehicle_attitude.q[3] + vehicle_attitude.q[1]*vehicle_attitude.q[2]), 1 - 2*(vehicle_attitude.q[2]**2 + vehicle_attitude.q[3]**2) )

    def subscribe_vehicle_local_position(self, vehicle_local_position):
        self.current_altitude = vehicle_local_position.z

    def check_altitude_reached(self):
        if self.current_altitude <= -9.5 and self.current_altitude >= -10.5 and self.altitude_reached == False:
            self.get_logger().info('='*40)
            self.get_logger().info('  TARGET ALTITUDE REACHED')
            self.get_logger().info('='*40)
            self.altitude_reached = True
            self.hover_start_time = self.get_clock().now()

    def check_hover(self):
        self.elapsed  = (self.get_clock().now() - self.hover_start_time).nanoseconds / 1e9
        if int(self.elapsed) >= self.hover_count + 1 and self.hover_count < 5:
            self.get_logger().info('='*40)
            self.get_logger().info(f'  Hovered for {self.hover_count} seconds' )
            self.get_logger().info('='*40)

            self.hover_count += 1

        elif self.hover_count >= 5 and self.land_started == False:
            self.land()
        
            
        
            



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
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(sp)

        if self.counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.get_logger().info('='*40)  
            self.get_logger().info('  OFFBOARD MODE SET + ARM COMMANDED')
            self.get_logger().info('='*40)
            self.arm()

        self.check_altitude_reached()

        if self.altitude_reached:
            self.check_hover()
        self.counter += 1


    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info('='*40)
        self.get_logger().info('  DRONE ARMED — TAKEOFF STARTED')
        self.get_logger().info('='*40)

    def land(self):
        self.get_logger().info('='*40)
        self.get_logger().info(' Landing started' )
        self.get_logger().info('='*40)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.land_started = True

        


def main(args=None):
    rclpy.init(args=args)
    node = TakeoffNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()