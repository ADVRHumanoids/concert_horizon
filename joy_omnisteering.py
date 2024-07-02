#!/usr/bin/env python3
import numpy as np
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
import rospy

class JoyOmnisteering:
    def __init__(self):

        self.joy_msg = None
        self.base_weight = .6
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        rospy.wait_for_message('/joy', Joy, timeout=0.5)
        self.omnisteering_pub = rospy.Publisher('/omnisteering/cmd_vel', Twist, queue_size=1, tcp_nodelay=True)

    def joy_callback(self, msg: Joy):
        self.joy_msg = msg

    def run(self):

        vel_msg = Twist()
        if np.abs(self.joy_msg.axes[0]) > 0.1 or np.abs(self.joy_msg.axes[1]) > 0.1:
            vel_msg.linear.x = self.base_weight * self.joy_msg.axes[0]
            vel_msg.linear.y = self.base_weight * self.joy_msg.axes[1]


        if np.abs(self.joy_msg.axes[3]) > 0.1:
            # move base on x-axis in local frame
            vel_msg.angular.z = self.base_weight * self.joy_msg.axes[3]



        self.omnisteering_pub.publish(vel_msg)

if __name__ == '__main__':

    rospy.init_node('joy_omnisteering')
    jc = JoyOmnisteering()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        jc.run()
        rate.sleep()