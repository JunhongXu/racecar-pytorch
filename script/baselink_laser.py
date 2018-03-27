#!/usr/bin/env python
import roslib
import rospy
import tf

def publish_odom_base_link():
    br = tf.TransformBroadcaster()
    br.sendTransform((0.15, 0, 0),
                     tf.transformations.quaternion_from_euler(0, 0, 0),
                     rospy.Time.now(),
                     'laser',
                     'base_link'
                     )


if __name__ == '__main__':
    rospy.init_node('base_link_laser')
    while True:
        publish_odom_base_link()
        rospy.sleep(1/60)
