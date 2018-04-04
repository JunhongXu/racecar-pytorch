#!/usr/bin/env python

import subprocess
import rospy
from ps3 import PS3
from bsu_racecar.msg import ps3_cmd
import datetime
import os
import shlex


class Recorder(object):
    def __init__(self):
        # find record parameter
        if not rospy.has_param('recorder'): 
            rospy.loginfo('[!]No recorder parameters')
        # load parameters
        self.record_params = rospy.get_param('recorder')
        self.ps3 = PS3()
        self.recording = False
        self.rosbag_proc = None
        # subscribers
        rospy.Subscriber('ps3_cmd', ps3_cmd, self.cmd_callback, queue_size=5)
        
    def cmd_callback(self, cmd):
        button_events = cmd.key_commands
        if 'start_pressed' in button_events:
            self.recording = not self.recording
            if self.recording:
                rospy.loginfo("Start recording sensor data!")
                # use command line to start writing to rosbag
                bag_root_dir = self.record_params['record_dir']

                # extract topics from topic list
                topics = self.record_params['topics']
                args = ''
                for t in topics:
                    args += t + ' '
                cmd = 'rosbag record -o %s/ %s' % (bag_root_dir, args)
                cmd = shlex.split(cmd)
                self.rosbag_proc = subprocess.Popen(cmd)
            else:
                rospy.loginfo("Stop recording sensor data!")
                self.rosbag_proc.send_signal(subprocess.signal.SIGINT)

if __name__ == '__main__':
    try:
        rospy.init_node('recorder')
        recorder = Recorder()
        rospy.spin()
    except:
        pass
