import numpy
import torch
import os
from PIL import Image

def box_xyxy_expand2square(box, w = 1600, h = 900):
    if w == h:
        return box , w, w
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box , w, w
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box , h, h

def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    # normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    normalized_box = (norm_x1, norm_y1, norm_x2, norm_y2)
    return normalized_box

def norm_box_xywh(box, *, w, h):
    x1, y1, x2, y2 = box
    cx = (x1+x2) / 2
    cy = (y1+y2) / 2
    bw = (x2-x1)
    bh = (y2-y1)
    norm_cx = max(0.0, min(cx / w, 1.0))
    norm_cy = max(0.0, min(cy / h, 1.0))
    norm_bw = max(0.0, min(bw / w, 1.0))
    norm_bh = max(0.0, min(bh / h, 1.0))
    normalized_box_xywh = (norm_cx, norm_cy, norm_bw, norm_bh)
    return normalized_box_xywh

def flatten(boxes, is_3d = False):
    flatten_list =  [a for a in boxes]
    if not is_3d:
        normalized_boxes = []
        for box in flatten_list:
            box, new_w, new_h = box_xyxy_expand2square(box)
            normalized_boxes.append(norm_box_xyxy(box, w = new_w, h= new_h))
        return normalized_boxes
    else:
        return flatten_list

def cxcywh_flatten(boxes, is_3d = False):
    flatten_list =  [a for a in boxes]
    if not is_3d:
        normalized_boxes = []
        for box in flatten_list:
            box, new_w, new_h = box_xyxy_expand2square(box)
            normalized_boxes.append(norm_box_xywh(box, w = new_w, h= new_h))
        return normalized_boxes
    else:
        return flatten_list