#!/usr/bin/env python

from model import ActionPredictionPlainNetwork 
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int16
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from torch.autograd import Variable
import cv2
import numpy as np


class Policy(object):
    def __init__(self):
        # initialize network
        self.net = ActionPredictionPlainNetwork()

        self.net.load()
        self.net.eval()
        self.net.cuda()

        self.bridge = CvBridge()
        print('Policy start!')
        self.action_publisher = rospy.Publisher('nn_cmd', AckermannDriveStamped, queue_size=5)

        # subscribe to rgb images
        rospy.Subscriber('/zed/rgb/image_rect_color', Image, self.act)
        rospy.init_node('policy', anonymous=True)
        rospy.spin()


    def act(self, image):
        image = self.bridge.imgmsg_to_cv2(image)
        image = cv2.resize(image, (128, 128))
        image = np.transpose(image, (2, 0, 1)) / 255.
        image = image.astype(np.float32)
        self.image = Variable(torch.from_numpy(image[np.newaxis]), volatile=True).cuda()

        pred_action, _ = self.net.forward(self.image)
        pred_action = pred_action.data.cpu().numpy().flatten()
        forward = pred_action[0]
        angular = pred_action[1]
        twist = AckermannDriveStamped()
        twist.drive.speed = forward
        twist.drive.steering_angle = angular
        self.action_publisher.publish(twist)

if __name__ == '__main__':
    try:
        policy = Policy()
    except rospy.ROSInterruptException as e:
        print(e)
