#!/usr/bin/env python

import rospy
from ps3 import PS3
from sensor_msgs.msg import Joy
from bsu_racecar.msg import ps3_cmd

class CommandServer(object):
    """
        Distribute Joy commands using PS3 class
    """
    def __init__(self):
        self.ps3 = PS3()
        rospy.Subscriber('joy', Joy, self.joy_callback, queue_size=2)
        self.ps3_pub = rospy.Publisher('ps3_cmd', ps3_cmd, queue_size=2)

    def joy_callback(self, joy):
        self.ps3.update(joy)
        button_events = self.ps3.btn_events
        stick =  [self.ps3.left_stick, self.ps3.right_stick]
        cmd = ps3_cmd()
        cmd.key_commands = button_events
        cmd.stick_control = stick
        self.ps3_pub.publish(cmd)

if __name__ == '__main__':
    rospy.init_node('command_server')
    command_server = CommandServer()
    rospy.spin()
