#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''


class DBWNode(object):

    def __init__(self):
        rospy.init_node('dbw_node')

        self.dbw_enabled = None
        self.current_vel = None
        self.linear_vel = None
        self.angular_vel = None
        self.throttle = None
        self.brake = None
        self.steering = None

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', 0.1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.0)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.0)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # Create `Controller` object
        self.controller = Controller(vehicle_mass,
                                     fuel_capacity,
                                     brake_deadband,
                                     decel_limit,
                                     accel_limit,
                                     wheel_radius,
                                     wheel_base,
                                     steer_ratio,
                                     max_lat_accel,
                                     max_steer_angle)

        # Subscribe to our topics
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            if None not in (self.current_vel, self.linear_vel, self.angular_vel):
                self.throttle, self.brake, self.steering = self.controller.control(self.current_vel,
                                                                                   self.dbw_enabled,
                                                                                   self.linear_vel,
                                                                                   self.angular_vel)
            if self.dbw_enabled:
                self.publish(self.throttle, self.brake, self.steering)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        """
        Publishes the throttle, brake and steer commands
        :param throttle: the throttle value
        :param brake: the brake value
        :param steer: the steering value
        """
        throttle_cmd = ThrottleCmd()
        throttle_cmd.enable = True
        throttle_cmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        throttle_cmd.pedal_cmd = throttle
        self.throttle_pub.publish(throttle_cmd)

        steering_cmd = SteeringCmd()
        steering_cmd.enable = True
        steering_cmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(steering_cmd)

        brake_cmd = BrakeCmd()
        brake_cmd.enable = True
        brake_cmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        brake_cmd.pedal_cmd = brake
        self.brake_pub.publish(brake_cmd)

    def twist_cb(self, msg):
        """
        Twist message callback
        :param msg: the twist message
        """
        self.linear_vel = msg.twist.linear.x
        self.angular_vel = msg.twist.angular.x

    def velocity_cb(self, msg):
        """
        Velocity message callback
        :param msg: the velocity message
        """
        self.current_vel = msg.twist.linear.x

    def dbw_enabled_cb(self, msg):
        """
        The dbw enabled message callback
        :param msg: the dbw enabled callback
        """
        self.dbw_enabled = msg


if __name__ == '__main__':
    DBWNode()
