"""
This script gives function for detecting from videos or from images
"""

import yolo
from model import yolo_cfg
import torch
from torch.autograd import Variable
import cv2
import numpy as np


def detect(image, model, img_shape):
    # resize image
    gpu_img = cv2.resize(image, yolo_cfg.image_size)
    # convert to gpu tensor
    gpu_img = gpu_img[np.newaxis].transpose(0, 3, 1, 2)
    gpu_img = Variable(torch.from_numpy(gpu_img), volatile=True).float().cuda() / 255.
    results = model.forward(gpu_img, img_shape)
    return results


def show(results, image):
    if results == 0:
        cv2.imshow('frame', image)
        cv2.waitKey(30)
    else:
        box_coords, class_probs, class_index = results
        box_coords, class_probs, class_index = box_coords.cpu().numpy(), class_probs.cpu().numpy(), \
                                               class_index.cpu().numpy()
        for box_coord, class_prob, c in zip(box_coords, class_probs, class_index):
            cv2.rectangle(image, tuple(box_coord[:2]), tuple(box_coord[2:]), color=(255, 0, 0), thickness=2)
            cv2.putText(image, '%s:%.4f' %(yolo_cfg.classes[c], class_prob), tuple(box_coord[:2]),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1,
                        color=(0, 0, 255))
        cv2.imshow('frame', image)
        cv2.waitKey(30)


def detect_video(video_pth):
    model = yolo.YOLO(yolo_cfg)
    model.load_state_dict(torch.load('model_utils/yolo.pth'))
    model.cuda()
    model.eval()
    cap = cv2.VideoCapture(video_pth)

    while cap.isOpened():
        ret, bgr = cap.read()
        if bgr is None:
            break
        h, w, c = bgr.shape
        h_new = 600
        w_new = h_new * h//w
        bgr = cv2.resize(bgr, (int(h_new), int(w_new)))
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_img_size = frame.shape[:2]
        # resize image
        gpu_img = cv2.resize(frame, yolo_cfg.image_size)
        # convert to gpu tensor
        gpu_img = gpu_img[np.newaxis].transpose(0, 3, 1, 2)
        gpu_img = Variable(torch.from_numpy(gpu_img), volatile=True).float().cuda()/255.
        # feed to yolo
        result = model.forward(gpu_img, img_size=original_img_size)
        if result == 0:
            cv2.imshow('frame', bgr)
            cv2.waitKey(30)
            continue
        else:
            box_coords, class_probs, class_index = result

        # convert to numpy array
        box_coords, class_probs, class_index = box_coords.cpu().numpy(), class_probs.cpu().numpy(), \
                                               class_index.cpu().numpy()
        for box_coord, class_prob, c in zip(box_coords, class_probs, class_index):
            cv2.rectangle(bgr, tuple(box_coord[:2]), tuple(box_coord[2:]), color=(255, 0, 0), thickness=2)
            cv2.putText(bgr, '%s:%.4f' %(yolo_cfg.classes[c], class_prob), tuple(box_coord[:2]),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1,
                        color=(0, 0, 255))

        cv2.imshow('frame', bgr)
        cv2.waitKey(25)


def detect_cam():
    cap = cv2.VideoCapture(1)
    model = yolo.YOLO(yolo_cfg)
    model.load_state_dict(torch.load('model_utils/yolo.pth'))
    model.cuda()
    model.eval()
    while True:
        ret, bgr = cap.read()
        h, w, c = bgr.shape
        bgr = bgr[:, :int(w//2), :]
        results = detect(bgr, model, img_shape=bgr.shape)
        # show(results, bgr)
        print(results)

    

if __name__ == '__main__':
    # detect_video('video5.mp4')
    detecti_cam()
