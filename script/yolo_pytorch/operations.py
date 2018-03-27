import torch
import torch.nn as nn
import numpy as np


# (x,y,w,h)--->(x1, y1, x2, y2)
def box_to_corners(box_pred):
    box_mins = box_pred[..., 0:2] - (box_pred[..., 2:] / 2.)
    box_maxes = box_pred[..., 0:2] + (box_pred[..., 2:] / 2.)
    return torch.cat([box_mins[..., 0:1], box_mins[..., 1:2],
                      box_maxes[..., 0:1], box_maxes[..., 1:2]], -1)


def prior(box, anchor_prior):
    """
    Adds the prior to the box.
    """
    # c_x + b_x; c_y + b_y
    n, dim, anchors, _ = box.size()
    offset = box.new(np.arange(np.sqrt(dim))).repeat(2).view(1, -1, 1, 1).expand_as(box[:, :, :, :2])
    box[:, :, :, :2] += offset

    # p_w * e^b_w; p_h * e^b_h
    anchor_prior = box.new(anchor_prior).expand_as(box[:, :, :, 2:])
    box[:, :, :, 2:] *= anchor_prior
    return box


def choose_box(box, box_confidence, cls_pred, threshold=0.5):
    """
    Choose the bounding boxes with only class_pred * box_confidence > threshold
    """
    # p(c|p_r), n, h*w, nbox, nclasses
    cls_pred = box_confidence * cls_pred

    # choose the max value and its index number from cls_pred, n, h*w, nbox
    max_cls_pred, max_cls_index = torch.max(cls_pred, dim=-1)
    # only keep values where max_cls_pred > threshold
    index = max_cls_pred > threshold

    max_cls_pred = torch.masked_select(max_cls_pred, index)
    max_cls_index = torch.masked_select(max_cls_index, index)

    index = index.unsqueeze(3)
    box = torch.masked_select(box, index)
    return box.view(-1, 4), max_cls_pred, max_cls_index


def non_max_suppression(boxes, scores, overlap=0.5, top_k=200):
    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.is_cuda: keep = keep.cuda()
    if boxes.numel() == 0:
        return keep
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1, yy1, xx2, yy2 = boxes.new(), boxes.new(), boxes.new(), boxes.new()
    w, h = boxes.new(), boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        # check sizes of xx1 and xx2.. after each iteration
        w, h = torch.clamp(xx2 - xx1, min=0.0), torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

