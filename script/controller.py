#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from vesc_msgs.msg import VescStateStamped
from sensor_msgs.msg import Joy
import importlib
from collections import defaultdict
import numpy as np
from ps3 import PS3

HUMAN_DEADBUTTON = 'L2'
NN_DEADBUTTON = 'L1'


class Controller(object):
    def __init__(self):
        # human subscriber
        self.human_sub = rospy.Subscriber('human_cmd', AckermannDriveStamped, self.update_human_cmd, queue_size=10)
        # nn subscriber
        self.nn_sub = rospy.Subscriber('nn_cmd', AckermannDriveStamped, self.update_nn_cmd, queue_size=10)
        # lidar subscriber
        self.lidar_sub = rospy.Subscriber('lidar_cmd', AckermannDriveStamped, self.update_lidar_cmd, queue_size=10)
        # joy subscriber
        self.joy_sub = rospy.Subscriber('joy', Joy, self.update_joy, queue_size=10)
        
        # publisher
        self.ackermann_pub = rospy.Publisher('ackermann_cmd', AckermannDriveStamped, queue_size=5)

        # msgs
        self.joy_cmd = None
        self.nn_cmd = None
        self.lidar_cmd = None
        self.human_cmd = None
        # ps3 controller
        self.ps3 = PS3()
        self.human_pressed = False
        self.nn_pressed = False

    def update_human_cmd(self, msg):
        self.human_cmd = msg

    def update_nn_cmd(self, msg):
        self.nn_cmd = msg

    def update_lidar_cmd(self, msg):
        self.lidar_cmd = msg

    def update_joy(self, msg):
        self.ps3.update(msg)
        # get button events
        button_events = self.ps3.btn_events
        if HUMAN_DEADBUTTON+'_pressed' in button_events:
            self.human_pressed = True
        if HUMAN_DEADBUTTON+'_released' in button_events:
            self.human_pressed = False
        print(button_events)
        if NN_DEADBUTTON+'_pressed' in button_events:
            self.nn_pressed = True
        if NN_DEADBUTTON+'_released' in button_events:
            self.nn_pressed = False
        
        if self.human_pressed:
            cmd = self.human_cmd
        elif self.nn_pressed:
            cmd = self.nn_cmd
        else:
            cmd = AckermannDriveStamped()
        self._publish_cmd(cmd)

    def _publish_cmd(self, msg):
        self.ackermann_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('controller')
    c = Controller()
    rospy.spin()

