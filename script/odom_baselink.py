#!/usr/bin/env python
import roslib
import rospy
import tf

def publish_odom_base_link():
    br = tf.TransformBroadcaster()
    br.sendTransform((-0.1, 0, 0),
                     tf.transformations.quaternion_from_euler(0, 0, 0),
                     rospy.Time.now(),
                     'base_link',
                     '/zed/odom'
                     )


if __name__ == '__main__':
    rospy.init_node('odom_base_link')
    while True:
        publish_odom_base_link()
        rospy.sleep(1/60)
