#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Joy
from geometry_msgs.msg import WrenchStamped
import rospy
import math
import colorama


class JoyForce:
    def __init__(self):

        self.joy_msg = None
        self.force = np.zeros(3)
        self.angle = 0
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        rospy.wait_for_message('/joy', Joy, timeout=0.5)
        self.force_weight = rospy.get_param('~max_force', default=100.0)
        self.link_name = rospy.get_param('~force_sensor_link', default='base_link')
        wrench_topic = rospy.get_param('~wrench_topic', default='/joy_commands/wrench')
        
        self.wrench_pub = rospy.Publisher(wrench_topic, WrenchStamped, queue_size=1)

    def joy_callback(self, msg: Joy):
        self.joy_msg = msg

    def run(self):

        if np.abs(self.joy_msg.axes[0]) > 0.1 or np.abs(self.joy_msg.axes[1]) > 0.1:
            # move base on x-axis in local frame
            vec = np.array([self.force_weight * self.joy_msg.axes[1],
                            self.force_weight * self.joy_msg.axes[0],
                            0])

            # reference = np.array([[solution['q'][0, 0] + rot_vec[0], solution['q'][1, 0] + rot_vec[1], 0., 0., 0., 0., 0.]]).T
            self.force[:2] = vec[:2]

        else:
            # move it back in the middle
            # reference = np.array([[solution['q'][0, 0], solution['q'][1, 0], 0., 0., 0., 0., 0.]]).T
            self.force[:2] = np.zeros(2)


        if np.abs(self.joy_msg.axes[4]) > 0.1:
            # move base on x-axis in local frame
            val_z = self.force_weight * self.joy_msg.axes[4]
            self.force[2] = val_z

        else:
            # move it back in the middle
            self.force[2] = 0

        # rotate vector
        # self.force = self._rotate_vector(self.force, solution['q'][[6, 3, 4, 5], 0])
        # self.angle = self._angle(rot_vec, solution['q'][[6, 3, 4, 5], 0])

    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    def _rotate_vector(self, vector, quaternion):

        # normalize the quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)

        # construct a pure quaternion
        v = np.array([0, vector[0], vector[1], vector[2]])

        # rotate the vector p = q* v q
        rotated_v = self._quaternion_multiply(quaternion, self._quaternion_multiply(v, self._conjugate_quaternion(quaternion)))

        # extract the rotated vector
        rotated_vector = rotated_v[1:]

        return rotated_vector

    def _conjugate_quaternion(self, q):
        q_conjugate = np.copy(q)
        q_conjugate[1:] *= -1.0
        return q_conjugate

    def _angle(self, v1, quat):

        rot_mat = R.from_quat(quat).as_matrix()
        v2 = np.dot(rot_mat, [1, 0, 0])

        # Compute the angle in radians
        theta = np.arctan2(v1[0]*v2[1]-v2[0]*v1[1],v1[0]*v2[0]+v1[1]*v2[1])

        return theta

    def getForce(self):
        return self.force

    def getAngle(self):
        return self.angle

if __name__ == '__main__':

    rospy.init_node('joy_trial')
    jc = JoyForce()

    while not rospy.is_shutdown():
        jc.run()
        
        wrench = WrenchStamped()
        
        wrench.header.frame_id = jc.link_name
        
        f = jc.getForce()
        
        wrench.wrench.force.x = -f[1]
        wrench.wrench.force.y = f[0]
        wrench.wrench.torque.z = f[2]
        
        jc.wrench_pub.publish(wrench)