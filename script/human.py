#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
from ps3 import PS3
from ackermann_msgs.msg import AckermannDriveStamped


class Human(object):
    def __init__(self):
        self.human_pub = rospy.Publisher('human_cmd', AckermannDriveStamped, queue_size=5)
        self.joy_sub = rospy.Subscriber('joy', Joy, self.update_joy, queue_size=5)
        self.ps3 = PS3()

    def update_joy(self, msg):
        self.ps3.update(msg)
        button_events = self.ps3.btn_events
        acceleration = self.ps3.left_stick
        steer = self.ps3.right_stick
        drive = AckermannDriveStamped()
        drive.drive.speed = acceleration
        drive.drive.steering_angle = steer
        self.human_pub.publish(drive)

if __name__ =='__main__':
    rospy.init_node('human')
    human = Human()
    rospy.spin()
