#!/usr/bin/env python
import roslib
import rospy
import tf
from nav_msgs.msg import Odometry

def publish_map_odom(msg):
    pose = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    x, y, z = pose.x, pose.y, pose.z
    o_x, o_y, o_w, o_z = orientation.x, orientation.y, orientation.w, orientation.z
    br = tf.TransformBroadcaster()
    br.sendTransform((x, y, z),
                     (o_x, o_y, o_z, o_w),
                     rospy.Time.now(),
                     '/zed/odom',
                     'map'
                     )


if __name__ == '__main__':
    rospy.init_node('map_to_odom')
    sub = rospy.Subscriber('/zed/odom', Odometry, publish_map_odom)
    rospy.spin()
