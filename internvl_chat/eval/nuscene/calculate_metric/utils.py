import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from mmdet.core.bbox.iou_calculators import bbox_overlaps
def restore_bbox(bbox, w = 1600, h = 900):
    x1, y1, x2, y2 = bbox
    x1_new = x1 * w if x1>=0 else 0
    x2_new = x2 * w if x2>=0 else 0
    y1_new = (y1 * w) -(w-h) //2 if y1>=0 else 0
    y2_new = (y2 * w) - (w-h) //2 if y2>=0 else 0
    return[x1_new, y1_new, x2_new, y2_new]

def polygon_iou(box1, box2):
    """
    计算两个旋转框的IoU

    Args:
    box1: 第一个旋转框的坐标，形状为(4, 2)，每一行代表一个点的坐标，顺时针或逆时针排列
    box2: 第二个旋转框的坐标，形状为(4, 2)，每一行代表一个点的坐标，顺时针或逆时针排列

    Returns:
    iou: 两个旋转框的IoU值
    """
    # 创建Shapely多边形对象
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    # 计算交集面积
    intersection_area = poly1.intersection(poly2).area
    
    # 计算旋转框的面积
    area1 = poly1.area
    area2 = poly2.area

    # 计算IoU
    iou = intersection_area / (area1 + area2 - intersection_area)
    # print(intersection_area)
    # print(area1)
    # print(area2)
    return iou

def euler_to_rotation_matrix(euler):
    # Convert euler angles to rotation matrix
    r = Rotation.from_euler('ZYX', euler)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def get_corners(bbox, wlh_factor: float = 1.0): # -> np.ndarray:
    '''
     获取3d框的角点
    '''
    x,y,z,w,l,h,yaw = bbox
    # w, l, h = [w, l, h]*wlh_factor
    orientation = (yaw,0,0)
    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    
    corners = np.dot(euler_to_rotation_matrix(orientation), corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners.T

def calculate_iou(bbox1, bbox2):
    # bbox1和bbox2分别为两个边界框的坐标 [x1, y1, x2, y2]
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area

    return iou

# 计算不同IoU阈值下的mAP和F1 score
def compute_map_f1(gt_boxes, pred_boxes, pred_scores, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    """
    计算不同IoU阈值下的mAP和F1 score
    gt_boxes: Ground Truth框，列表形式，每个元素为 [x1, y1, x2, y2]
    pred_boxes: 预测框，列表形式，每个元素为 [x1, y1, x2, y2]
    pred_scores: 预测框的得分，列表形式，长度与pred_boxes相同
    iou_thresholds: 用于计算mAP的IoU阈值数组，默认为[0.5, 0.55, ..., 0.95]
    """
    # pred_scores_sorted = [1.0]*len(pred_boxes)
    sorted_indices = np.argsort(pred_scores)[::-1]  # 按得分降序排列
    pred_boxes_sorted = [pred_boxes[i] for i in sorted_indices]
    pred_scores_sorted = [pred_scores[i] for i in sorted_indices]
    # 准备计算每个IoU阈值的AP和F1 score
    ap_list = []
    f1_list = []
    
    for threshold in iou_thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        tp = []  # True Positive
        fp = []  # False Positive
        num_gt = len(gt_boxes)
        
        # 标记每个GT是否被预测框正确匹配
        matched_gt = [False] * num_gt
        
        for i in range(len(pred_boxes_sorted)):
            pred_box = pred_boxes_sorted[i]
            iou_max = 0
            matched_gt_idx = -1
            
            # 找到和当前预测框IoU最大值的GT
            for j in range(num_gt):
                if not matched_gt[j]:
                    # iou = calculate_iou(pred_box, gt_boxes[j])
                    iou = bbox_overlaps(torch.FloatTensor([pred_box]), torch.FloatTensor([gt_boxes[j]])).item()
                    if iou > iou_max:
                        iou_max = iou
                        matched_gt_idx = j

            if iou_max >= threshold and not matched_gt[matched_gt_idx]:
                matched_gt[matched_gt_idx] = True
                true_positives += 1
                tp.append(1)
                fp.append(0)
            else:
                false_positives += 1
                tp.append(0)
                fp.append(1)

        
        false_negatives = num_gt - true_positives
        
        # 计算 Precision, Recall 和 F1 score
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        # # 计算当前IoU阈值下的AP
        # precision_at_recall, recall_at_precision, _ = precision_recall_curve([1] * true_positives + [0] * false_positives,
        #                                                                       [pred_scores_sorted[i] for i in range(true_positives + false_positives)])
        # ap = average_precision_score([1] * true_positives + [0] * false_positives, [pred_scores_sorted[i] for i in range(true_positives + false_positives)])
        
        # 计算累计TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        prec = tp_cumsum / (tp_cumsum + fp_cumsum)
        rec = tp_cumsum / num_gt
        rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
        prec = np.interp(rec_interp, rec, prec, right=0)
        rec = rec_interp

        prec = np.copy(prec)
        prec = prec[round(100 * 0.1) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= 0.1  # Clip low precision
        prec[prec < 0] = 0

        rec = rec[round(100 * 0.1) + 1:]
        ap = float(np.mean(prec)) / (1.0 - 0.1)
        # ap = float(np.mean(prec))
        # 记录AP和F1 score
        ap_list.append(ap)
        f1_list.append(f1)
    
    # 计算不同IoU阈值下的mAP和F1 score
    mean_ap = np.mean(ap_list)
    mean_f1 = np.mean(f1_list)

    return mean_ap, mean_f1, ap_list, f1_list

def calculate_single_3d_iou(boxes_pred, boxes_true):
    """
    Calculate 3D IoU (Intersection over Union) between predicted and ground truth boxes.
    
    Args:
    boxes_pred: Tensor of shape (N, 7) representing predicted boxes in (x, y, z, w, l, h, yaw) format.
    boxes_true: Tensor of shape (N, 7) representing ground truth boxes in (x, y, z, w, l, h, yaw) format.
    
    Returns:
    Tensor of shape (N,) containing 3D IoU values.
    """
    # Convert boxes to (x_min, y_min, z_min, x_max, y_max, z_max) format
    boxes_pred_corners = torch.cat([boxes_pred[:3] - boxes_pred[3:6] / 2,
                                    boxes_pred[:3] + boxes_pred[3:6] / 2])
    boxes_true_corners = torch.cat([boxes_true[:3] - boxes_true[3:6] / 2,
                                    boxes_true[:3] + boxes_true[3:6] / 2])
    
    # Calculate intersection volume
    intersection_min = torch.max(boxes_pred_corners[:3], boxes_true_corners[:3])
    intersection_max = torch.min(boxes_pred_corners[3:], boxes_true_corners[3:])
    intersection_dims = torch.clamp(intersection_max - intersection_min, min=0)
    intersection_volume = intersection_dims[0] * intersection_dims[:, 1] * intersection_dims[:, 2]
    
    # Calculate pred box volume
    pred_volume = (boxes_pred[:, 3] * boxes_pred[:, 4] * boxes_pred[:, 5]).clamp(min=0)
    
    # Calculate true box volume
    true_volume = (boxes_true[:, 3] * boxes_true[:, 4] * boxes_true[:, 5]).clamp(min=0)
    
    # Calculate union volume
    union_volume = pred_volume + true_volume - intersection_volume
    
    # Calculate IoU
    iou = intersection_volume / union_volume
    
    return iou

def calculate_3d_iou(boxes_pred, boxes_true):
    """
    Calculate 3D IoU (Intersection over Union) between predicted and ground truth boxes.
    
    Args:
    boxes_pred: Tensor of shape (N, 7) representing predicted boxes in (x, y, z, w, l, h, yaw) format.
    boxes_true: Tensor of shape (N, 7) representing ground truth boxes in (x, y, z, w, l, h, yaw) format.
    
    Returns:
    Tensor of shape (N,) containing 3D IoU values.
    """
    # Convert boxes to (x_min, y_min, z_min, x_max, y_max, z_max) format
    boxes_pred_corners = torch.cat([boxes_pred[:, :3] - boxes_pred[:, 3:6] / 2,
                                    boxes_pred[:, :3] + boxes_pred[:, 3:6] / 2], dim=1)
    boxes_true_corners = torch.cat([boxes_true[:, :3] - boxes_true[:, 3:6] / 2,
                                    boxes_true[:, :3] + boxes_true[:, 3:6] / 2], dim=1)
    
    # Calculate intersection volume
    intersection_min = torch.max(boxes_pred_corners[:, :3], boxes_true_corners[:, :3])
    intersection_max = torch.min(boxes_pred_corners[:, 3:], boxes_true_corners[:, 3:])
    intersection_dims = torch.clamp(intersection_max - intersection_min, min=0)
    intersection_volume = intersection_dims[:, 0] * intersection_dims[:, 1] * intersection_dims[:, 2]
    
    # Calculate pred box volume
    pred_volume = (boxes_pred[:, 3] * boxes_pred[:, 4] * boxes_pred[:, 5]).clamp(min=0)
    
    # Calculate true box volume
    true_volume = (boxes_true[:, 3] * boxes_true[:, 4] * boxes_true[:, 5]).clamp(min=0)
    
    # Calculate union volume
    union_volume = pred_volume + true_volume - intersection_volume
    
    # Calculate IoU
    iou = intersection_volume / union_volume
    
    return iou

def scale_iou(sample_annotation, sample_result):
    # Validate inputs.
    sa_size = np.array(sample_annotation[3:6])
    sr_size = np.array(sample_result[3:6])
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou
def calculate_3d_tp_fp(boxes_pred, bbox_gt, min_distance):
    # 返回TP的个数和FP的个数
    tp = 0
    fp = 0
    boxes_pred_center = boxes_pred[:, :2] 
    boxes_gt_center = bbox_gt[:, :2]
    # 采用一对一匹配的方式
    for i in range(len(boxes_pred_center)):
        distance = np.linalg.norm(boxes_pred_center[i]-boxes_gt_center[i])
        if distance < min_distance:
            tp = tp+1
        else:
            fp = fp+1
    return tp,fp

def is_calculate_3d_tp(boxes_pred, bbox_gt, min_distance):
    # 返回是否是tp
    tfp =[]
    boxes_pred_center = boxes_pred[:, :2] 
    boxes_gt_center = bbox_gt[:, :2]
    # 采用一对一匹配的方式
    for i in range(len(boxes_pred_center)):
        distance = np.linalg.norm(boxes_pred_center[i]-boxes_gt_center[i])
        if distance < min_distance:
            tfp.append(1)
        else:
            tfp.append(0)
    return tfp

def calculate_2d_tp_fp(boxes_pred, bbox_gt, min_iou):
    # 返回TP的个数和FP的个数
    tp = 0
    fp = 0
    # 采用一对一匹配的方式
    for i in range(len(boxes_pred)):
        iou = calculate_iou(boxes_pred[i], bbox_gt[i])
        if iou > min_iou:
            tp = tp+1
        else:
            fp = fp+1
    return tp,fp

def velocity_l2(boxes_pred, bbox_gt): # -> float:
    pred_box_velocity = boxes_pred[-2:]
    gt_box_velocity = bbox_gt[-2:]

    return np.linalg.norm(np.array(pred_box_velocity) - np.array(gt_box_velocity))


def center_distance(pred_box, gt_box):
    pred_box = pred_box[:2]
    gt_box = gt_box[:2]
    return np.linalg.norm(np.array(pred_box) - np.array(gt_box))


def angle_diff(boxes_pred, bbox_gt):
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """
    period = 2 * np.pi

    x = boxes_pred[7]
    y = bbox_gt[7]
    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]
            
    return abs(diff)
def box_iou(boxes1, boxes2):
    """
    Calculate IoU (Intersection over Union) between two sets of boxes.
    
    Args:
    boxes1: Tensor of shape (N, 4) representing boxes in (x_min, y_min, x_max, y_max) format.
    boxes2: Tensor of shape (M, 4) representing boxes in (x_min, y_min, x_max, y_max) format.
    
    Returns:
    Tensor of shape (N, M) containing IoU values.
    """
    # Compute intersection coordinates
    inter_xmin = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    inter_ymin = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    inter_xmax = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    inter_ymax = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    
    # Compute intersection area
    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_width * inter_height
    
    # Compute box areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute union area
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    # Compute IoU
    iou = inter_area / union_area
    
    return iou

def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Match ground truth boxes with predicted boxes based on IoU.
    
    Args:
    gt_boxes: Tensor of shape (N, 4) representing ground truth boxes in (x_min, y_min, x_max, y_max) format.
    pred_boxes: Tensor of shape (M, 4) representing predicted boxes in (x_min, y_min, x_max, y_max) format.
    iou_threshold: IoU threshold for matching boxes.
    
    Returns:
    List of matched indices. Each element is a tuple (gt_idx, pred_idx).
    """
    # Calculate IoU between GT and predicted boxes
    iou_matrix = box_iou(gt_boxes, pred_boxes)
    
    # Initialize list to store matched indices
    matched_indices = []
    match_iou = []
    # Iterate through IoU matrix to find matches
    for i in range(len(gt_boxes)):
        max_iou = torch.max(iou_matrix[i])
        if max_iou >= iou_threshold:
            pred_idx = torch.argmax(iou_matrix[i])
            matched_indices.append((i, pred_idx))
            match_iou.append(iou_matrix[i][pred_idx])
    
    return matched_indices, match_iou

if __name__ == '__main__':
    # gen_bboxs = [[0.443359375, 0.5078125, 0.61328125, 0.5859375]]
    # gt_bboxs = [[0.4878790283203125, 0.5265496253967286, 0.5352613830566406, 0.5644100952148438]] #, [0.5567982482910157, 0.5162682342529297, 0.6002484512329102, 0.563305320739746]
    gen_bboxs = [[0.48518508672714233, 0.5152429342269897, 0.5366843342781067, 0.555976390838623], [0.5334815979003906, 0.5141007900238037, 0.5779705047607422, 0.5494953393936157]]
    gt_bboxs = [[0.4878790283203125, 0.5265496253967284, 0.5352613830566406, 0.5644100952148438], [0.5567982292175293, 0.5162682342529297, 0.6002484321594239, 0.563305320739746]]
    restore_gen_bboxs = [restore_bbox(bbox) for bbox in gen_bboxs]
    restore_gt_bboxs = [restore_bbox(bbox) for bbox in gt_bboxs]
    scale_gen_bboxs = [[i *1600 for i in bbox] for bbox in gen_bboxs]
    scale_gt_bboxs = [[i *1600 for i in bbox] for bbox in gt_bboxs]
    
    iou = np.zeros((len(gen_bboxs), len(gt_bboxs)))
    restore_iou = np.zeros((len(gen_bboxs), len(gt_bboxs)))
    exquare_iou = np.zeros((len(gen_bboxs), len(gt_bboxs)))
    for i, gen_bbox in enumerate(gen_bboxs):
        for j, gt_bbox in enumerate(gt_bboxs):
            iou[i][j] = calculate_iou(gen_bbox, gt_bbox)
            restore_iou[i][j] = calculate_iou(restore_bbox(gen_bbox), restore_bbox(gt_bbox))
            exquare_iou[i][j] = calculate_iou([i * 1600 for i in gen_bbox], [i * 1600 for i in gt_bbox])
    iou_mmdet = bbox_overlaps(torch.FloatTensor(gen_bboxs), torch.FloatTensor(gt_bboxs))
    restore_iou_mmdet = bbox_overlaps(torch.FloatTensor(restore_gen_bboxs), torch.FloatTensor(restore_gt_bboxs))
    exquare_iou_mmdet = bbox_overlaps(torch.FloatTensor(scale_gen_bboxs), torch.FloatTensor(scale_gt_bboxs))
    print(iou)
    print(restore_iou)
    print(exquare_iou)
    print(iou_mmdet)
    print(restore_iou_mmdet)
    print(exquare_iou_mmdet)

