#!/usr/bin/env python
from __future__ import print_function
from yolo_pytorch.yolo import YOLO
import torch
from yolo_pytorch.model import yolo_cfg
from torch.autograd import Variable
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from bsu_racecar.msg import bbox_data, bbox_array
from sensor_msgs.msg import Image

# TODO: Implement a class that subscribe to /zed/rgb/image_rect_color; process it using yolo; and publish to yolo/processed_rgb

class Detector(object):
    def __init__(self):
        import os
        print('working dir', os.getcwd())
        # initializing YOLO model
        self.yolo = YOLO(yolo_cfg)
        self.yolo.load_state_dict(torch.load('yolo_pytorch/model_utils/yolo.pth'))
        self.yolo.cuda()
        self.yolo.eval()
        self.bridge = CvBridge()
        self.detection_pub = rospy.Publisher("yolo/image",Image, queue_size=5)
        self.img_sub = rospy.Subscriber("/zed/rgb/image_rect_color", Image, self.detect, queue_size=5)
    def detect(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image)
            img_shape = cv_image.shape
            gpu_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            gpu_img = cv2.resize(gpu_img, yolo_cfg.image_size)
            gpu_img = gpu_img.transpose(2, 0, 1)[np.newaxis]
            gpu_img = Variable(torch.from_numpy(gpu_img), volatile=True).cuda().float()/255.
            # detect
            results = self.yolo.forward(gpu_img, img_shape)
            img = self._draw(cv_image, results)
            # send
            # img = cv2.resize(img, (224, 224))
            img = self.bridge.cv2_to_imgmsg(img)
            self.detection_pub.publish(img)

        except CvBridgeError as e:
            print(e)

    def _draw(self, img, results):
        if results == 0:
            return img
        else:
            box_coords, class_probs, class_index = results
            box_coords, class_probs, class_index = box_coords.cpu().numpy(), class_probs.cpu().numpy(), class_index.cpu().numpy()
            for box_coord, class_prob, c in zip(box_coords, class_probs, class_index):
                cv2.rectangle(img, tuple(box_coord[:2]), tuple(box_coord[2:]), color=(255, 0, 0), thickness=2)
                cv2.putText(img, '%s:%.4f' % (yolo_cfg.classes[c], class_prob), tuple(box_coord[:2]), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(0, 0, 255))
            return img


if __name__ == "__main__":
    rospy.init_node("yolo_detector")
    detector = Detector()
    rospy.spin()
