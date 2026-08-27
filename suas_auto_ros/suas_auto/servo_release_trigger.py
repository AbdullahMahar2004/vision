#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleCommand, VehicleCommandAck


# --- ACTUATOR CONFIG -------------------------------------
ACTUATOR_INDEX = 1          # param2 in DO_SET_ACTUATOR: which actuator set/output
VAL_RELEASE    = 1.0        # commanded value to open/release
VAL_RESET      = -1.0       # commanded value to close/reset
HOLD_TICKS     = 40         # ticks to hold release before resetting (~2s at 20Hz)
ACK_TIMEOUT_TICKS = 40      # ticks to wait for an ack before declaring failure (~2s at 20Hz)
# -----------------------------------------------------------

# MAVLink command 187 = MAV_CMD_DO_SET_ACTUATOR
MAV_CMD_DO_SET_ACTUATOR = 187

# VehicleCommandAck.result values (from px4_msgs)
VEHICLE_CMD_RESULT_ACCEPTED = 0


class ServoReleaseTrigger(Node):

    def __init__(self):
        super().__init__('servo_release_trigger')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.ack_sub = self.create_subscription(
            VehicleCommandAck, '/fmu/out/vehicle_command_ack',
            self.ack_callback, qos)

        self.state = 'SEND_RELEASE'
        self.tick_counter = 0
        self.release_ack_ok = None   # None = no ack yet, True/False = result
        self.reset_ack_ok = None

        self.get_logger().info('Servo release trigger node started.')
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

    def ack_callback(self, msg):
        if msg.command != MAV_CMD_DO_SET_ACTUATOR:
            return  # not our command, ignore

        accepted = (msg.result == VEHICLE_CMD_RESULT_ACCEPTED)

        if self.state == 'WAIT_ACK_RELEASE':
            self.release_ack_ok = accepted
            if accepted:
                self.get_logger().info('PX4 ACCEPTED release command.')
            else:
                self.get_logger().error(f'PX4 REJECTED release command (result={msg.result}).')

        elif self.state == 'WAIT_ACK_RESET':
            self.reset_ack_ok = accepted
            if accepted:
                self.get_logger().info('PX4 ACCEPTED reset command.')
            else:
                self.get_logger().error(f'PX4 REJECTED reset command (result={msg.result}).')

    def timer_callback(self):
        if self.state == 'SEND_RELEASE':
            self.send_actuator_command(VAL_RELEASE)
            self.state = 'WAIT_ACK_RELEASE'
            self.tick_counter = 0

        elif self.state == 'WAIT_ACK_RELEASE':
            self.tick_counter += 1
            if self.release_ack_ok is True:
                self.state = 'HOLD'
                self.tick_counter = 0
            elif self.release_ack_ok is False:
                self.finish(success=False, reason='PX4 rejected the release command.')
            elif self.tick_counter >= ACK_TIMEOUT_TICKS:
                self.finish(success=False, reason='No response from PX4 (release command timed out). '
                                                     'Is SITL/hardware running and the DDS agent connected?')

        elif self.state == 'HOLD':
            self.tick_counter += 1
            if self.tick_counter >= HOLD_TICKS:
                self.state = 'SEND_RESET'

        elif self.state == 'SEND_RESET':
            self.send_actuator_command(VAL_RESET)
            self.state = 'WAIT_ACK_RESET'
            self.tick_counter = 0

        elif self.state == 'WAIT_ACK_RESET':
            self.tick_counter += 1
            if self.reset_ack_ok is True:
                self.finish(success=True, reason='Release and reset both confirmed by PX4.')
            elif self.reset_ack_ok is False:
                self.finish(success=False, reason='Release succeeded, but PX4 rejected the reset command.')
            elif self.tick_counter >= ACK_TIMEOUT_TICKS:
                self.finish(success=False, reason='Release succeeded, but no response to the reset command.')

    def send_actuator_command(self, value):
	    msg = VehicleCommand()
	    msg.command = MAV_CMD_DO_SET_ACTUATOR
	    # Route the value into the correct param slot (param1..param6)
	    # based on ACTUATOR_INDEX. All other params stay NaN (unused).
	    params = [float('nan')] * 6
	    params[ACTUATOR_INDEX - 1] = float(value)  # ACTUATOR_INDEX=1 -> param1, 2 -> param2, etc.
	    msg.param1, msg.param2, msg.param3, msg.param4, msg.param5, msg.param6 = params

	    msg.target_system = 1
	    msg.target_component = 1
	    msg.source_system = 1
	    msg.source_component = 1
	    msg.from_external = True
	    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
	    self.command_pub.publish(msg)
	    self.get_logger().info(f'Sent DO_SET_ACTUATOR: value={value} -> param{ACTUATOR_INDEX} (awaiting ack...)')
	    
    def finish(self, success, reason):
        self.timer.cancel()
        if success:
            self.get_logger().info('SERVO TRIGGERED')
        else:
            self.get_logger().error(f'TRIGGER FAILED: {reason}')
        self.create_timer(0.2, self.shutdown_node)

    def shutdown_node(self):
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ServoReleaseTrigger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
