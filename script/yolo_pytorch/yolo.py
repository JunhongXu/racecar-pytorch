import torch.nn as nn
from torch.nn import functional as F
from operations import choose_box, non_max_suppression, box_to_corners
import torch


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2 if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Detector(nn.Module):
    def __init__(self, cfg):
        """
        Detector class is responsible for:
        1. adding prior to the boxes
        1. convert bounding box coordinates, height, and width to the original scale
        2. filter out non-object detections
        3. Apply non-maximal suppression
        """
        super(Detector, self).__init__()
        self.feat_size = cfg.feat_size

    def forward(self, box, box_confidence, class_score, prior, threshold=0.5):
        """

        :param box: (N, h*w, 5, 4) that contains coordinates, height, and width of each box
        :param box_confidence: (N, h*w, 5, 1) contains the confidence of each box
        :param class_score: (N, h*w, 5, nclass) contains probability of each class in each box
        :return: box coordinate, height, and width plus the class of this detection
        """
        # adding prior to the box
        box[:, :, :, :2] += prior[..., :2]
        box[:, :, :, 2:] *= prior[..., 2:]
        box = box_to_corners(box) / self.feat_size
        # filter out non-object detections
        box, class_probs, class_index = choose_box(box, box_confidence, class_score, threshold)
        return box, class_probs, class_index


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.size()
        s = self.stride
        x = x.view(B, C, H // s, s, W // s, s).transpose(3, 4).contiguous()
        x = x.view(B, C, H // s * W // s, s * s).transpose(2, 3).contiguous()
        x = x.view(B, C, s * s, H // s, W // s).transpose(1, 2).contiguous()
        return x.view(B, s * s * C, H // s, W // s)


# darknet feature detector
class DarknetBone(nn.Module):
    def __init__(self, cfg):
        super(DarknetBone, self).__init__()
        self.cfg = cfg
        in_channel, out_channel = 3, 16 if cfg.tiny else 32
        flag, pool, size_flag = cfg.flag, cfg.pool, cfg.size_flag
        layers1, layers2 = [], []
        for i in range(cfg.num):
            ksize = 1 if i in size_flag else 3
            if i < 13:
                layers1.append(ConvLayer(in_channel, out_channel, ksize, same_padding=True))
                layers1.append(nn.MaxPool2d(2)) if i in pool else None
                layers1 += [nn.ReflectionPad2d([0, 1, 0, 1]), nn.MaxPool2d(2, 1)] if i == 5 and cfg.tiny else []
            else:
                layers2.append(nn.MaxPool2d(2)) if i in pool else None
                layers2.append(ConvLayer(in_channel, out_channel, ksize, same_padding=True))
            in_channel, out_channel = out_channel, out_channel * 2 if flag[i] else out_channel // 2
        self.main1 = nn.Sequential(*layers1)
        self.main2 = nn.Sequential(*layers2)

    def forward(self, x):
        xd = self.main1(x)
        if self.cfg.tiny:
            return xd
        else:
            x = self.main2(xd)
            return x, xd


class YOLO(nn.Module):
    def __init__(self, cfg):
        super(YOLO, self).__init__()
        self.cfg = cfg
        self.prior = self._prior()
        self.darknet = DarknetBone(cfg)
        self.detector = Detector(cfg)
        # yolo detector
        self.conv1 = nn.Sequential(
                ConvLayer(1024, 1024, 3, same_padding=True),
                ConvLayer(1024, 1024, 3, same_padding=True))
        self.conv2 = nn.Sequential(
            ConvLayer(512, 64, 1, same_padding=True),
            ReorgLayer(2))
        self.conv = nn.Sequential(
            ConvLayer(1280, 1024, 3, same_padding=True),
            nn.Conv2d(1024, cfg.anchor_num * (cfg.class_num + 5), 1))

    def _prior(self):
        num_anchors = self.cfg.anchors.shape[0]
        feat_size = self.cfg.feat_size
        p = []
        for row in range(feat_size):
            for col in range(feat_size):
                for a in range(num_anchors):
                    a_w = self.cfg.anchors[a, 0]
                    a_h = self.cfg.anchors[a, 1]
                    _p = [col, row, a_w, a_h]
                    p.append(_p)
        return torch.FloatTensor(p).view(1, feat_size*feat_size, num_anchors, -1).cuda()

    def forward(self, x, img_size=None):
        """
        :param x: input image of shape (N, 3, H, W)
        :param img_size: rescaling purposes, only used for testing
        :return:
        """
        # x1 is the last layer, x2 is the output of -9 layer
        x1, x2 = self.darknet.forward(x)
        x = self.conv(torch.cat([self.conv2(x2), self.conv1(x1)], 1))

        if self.training:
            return x
        else:
            # reshape activation to N, h*w, anchors, 425
            n, c, h, w = x.size()
            x = x.permute(0, 2, 3, 1).contiguous().view(n, -1, self.cfg.anchor_num, (self.cfg.class_num + 5))
            box_coords = F.sigmoid(x[:, :, :, :2])
            box_dims = x[:, :, :, 2:4].exp()
            class_scores = F.softmax(x[:, :, :, 5:], dim=-1)
            box_confidence = F.sigmoid(x[:, :, :, 4:5])
            box = torch.cat([box_coords, box_dims], dim=3)
            box, class_probs, class_index = self.detector.forward(box.data, box_confidence.data, class_scores.data, self.prior, 0.5)
            image_shape = box.new([img_size[1], img_size[0], img_size[1], img_size[0]])
            if len(box.size()) == 0:
                # detects nothing
                return 0
            box *= image_shape
            keep, count = non_max_suppression(box, class_probs)
            return box[keep[:count]], class_probs[keep[:count]], class_index[keep[:count]]


