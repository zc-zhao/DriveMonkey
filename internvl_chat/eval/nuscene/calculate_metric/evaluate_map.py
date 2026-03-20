import numpy as np
import pickle
import torch
from tqdm import tqdm
import json
import re
import torch.nn.functional as F
def find_numbers(input_string):
    # 使用正则表达式找到所有1到99之间的数字
    numbers = re.findall(r'(?<!\d)\d{1,2}(?!\d)', input_string)
    # 将找到的数字字符串转换为整数列表
    numbers = [int(num) for num in numbers]
    return numbers

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def denormalize_bbox(normalized_bboxes, pc_range=None):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

def calculate_map(scores, bboxes, gtbboxes):
    
    # ✅ 关键：空预测直接返回0，避免所有后续问题
    if scores is None or len(scores) == 0:
        return 0.0, [0.0] * 4
    
    all_ap = []
    all_f1 = []
    for dist_th in [0.5, 1, 2, 4]:
        npos = len(gtbboxes)
        sorted_indices = torch.argsort(scores, descending=True)
        bboxes = bboxes[sorted_indices]
        # tp = torch.zeros(len(scores)).to(scores.device)  
        # fp = torch.zeros(len(scores)).to(scores.device) 
        tp = []  # Accumulator of true positives.
        fp = []  # Accumulator of false positives.
        matched_gts = torch.zeros(len(gtbboxes)).to(scores.device)
        
        for i, bbox in enumerate(bboxes):
            min_dist = np.inf
            for j, gtbbox in enumerate(gtbboxes):
                if not matched_gts[j]:
                    this_distance = torch.norm(bbox[:2] - gtbbox[:2])
                    if this_distance < min_dist:
                        min_dist = this_distance
                        

            is_match = min_dist < dist_th
            if is_match:
                matched_gts[j] = 1
                tp.append(1)
                fp.append(0)
            else:
                # No match. Mark this as a false positive.
                tp.append(0)
                fp.append(1)
                

        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)

        prec = tp / (fp + tp)
        rec = tp / float(npos)

        rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
        prec = np.interp(rec_interp, rec, prec, right=0)
        rec = rec_interp

        prec = np.copy(prec)
        prec = prec[round(100 * 0.1) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= 0.1  # Clip low precision
        prec[prec < 0] = 0

        rec = rec[round(100 * 0.1) + 1:]
        f1_scores = 2 * (prec * rec) / (prec + rec)

        f1 = np.mean(f1_scores)        

        all_ap.append(float(np.mean(prec)) / (1.0 - 0.1))
        all_f1.append(f1)


    return np.mean(all_ap),all_f1


data_path = [
  "Your result json file path list"
]


# 加载数据并计算 mAP
for dp in data_path:
    map_all = []
    F1_score_all = []
    
    all_f1_05 = []
    all_f1_1 = []
    all_f1_2 = []
    all_f1_4 = []
    data_3d_test = pickle.load(open(dp, 'rb')) if dp.endswith('.pkl') else json.load(open(dp, 'r'))

    valid_data = 0
    # score_threshold = 0.1
    score_threshold = 0.0
    test_data = []

    category_map = {}
    category_f1 = {}
    category_count = {}
    for i in tqdm(range(len(data_3d_test))):

        bbox_3d_gt = torch.tensor(data_3d_test[i]['bbox_3d_gt'])
        try:
            pred_3d_bbox = torch.tensor(data_3d_test[i]['bbox_pred_3d'])
            cls_3d_score = torch.tensor(data_3d_test[i]['cls_3d_score'])
            iou_3d_pred = None
        except Exception as e:
            pred_3d_bbox = torch.tensor(data_3d_test[i]['bbox_pred'])
            cls_3d_score = torch.tensor(data_3d_test[i]['cls_score'])
            iou_3d_pred = torch.tensor(data_3d_test[i]['iou_pred']) if 'iou_pred' in data_3d_test[i] and len(data_3d_test[i]['iou_pred']) > 0 else None
        pred_3d_bbox = denormalize_bbox(pred_3d_bbox)

        # if data_3d_test[i]['token'] != '4a1972f8731b4cdea40fc69a38a735b1':
        #         continue
        # test_data.append(data_3d_test[i])
        
        # cls_scores = F.softmax(cls_3d_score, dim=1)
        cls_scores = cls_3d_score.sigmoid()
        try:
            if iou_3d_pred is not None:
                iou_3d_pred = (iou_3d_pred + 1)*0.5
                iou_3d_pred = torch.clamp(iou_3d_pred,min = 0.0, max=1.0)
                cls_scores = cls_scores * torch.pow(iou_3d_pred, 1)
            scores, indexs = cls_scores.view(-1).topk(100)
            bbox_index = indexs // 10
            bbox_preds = pred_3d_bbox[bbox_index]
        except:
            continue
        
        # # ###score 过滤   
        if score_threshold is not None:
            thresh_mask = scores > score_threshold
        # scores, indexs = cls_scores.view(-1).topk(100)
        # bbox_index = indexs // 10
        bbox_preds_mask= bbox_preds[thresh_mask]
        scores_mask = scores[thresh_mask]
        if len(bbox_preds_mask) == 0:
            map_all.append(0)
            F1_score_all.append(0)
            continue

        ###### 得到k值
        pred_text = '' if 'gen' not in data_3d_test[i] else data_3d_test[i]['gen']
        try:
            k_num = find_numbers(pred_text)[0]
        except:

            print(f"Error finding numbers in text: {pred_text}")
            k_num = 0
            
        if k_num != 0:
            valid_data += 1

        if k_num < len(scores_mask):
            scores_k, indexs = scores_mask.topk(k_num)
            bbox_preds_k = bbox_preds_mask[indexs]
            
        try:
            mAP, all_f1 = calculate_map(scores_k, bbox_preds_k, bbox_3d_gt)
            if k_num == 0:
                print(f"mAP: {mAP}, all_f1: {all_f1}\n")
        except:
            # if k_num == 0:
            #     print(f"mAP: {mAP}, all_f1: {all_f1}\n")
            continue
        
        f1_05 = all_f1[0]
        f1_1 = all_f1[1]
        f1_2 = all_f1[2]
        f1_4 = all_f1[3]

        F1_score = np.mean(all_f1)
        map_all.append(mAP)
        F1_score_all.append(F1_score)
        all_f1_05.append(f1_05)
        all_f1_1.append(f1_1)
        all_f1_2.append(f1_2)
        all_f1_4.append(f1_4)

    print(f"mAP: {sum(map_all) / len(map_all)}")
    print(f"F1_score: {sum(F1_score_all) / len(F1_score_all)}")
    print(f"all_f1_05: {sum(all_f1_05) / len(all_f1_05)}")
    print(f"all_f1_1: {sum(all_f1_1) / len(all_f1_1)}")
    print(f"all_f1_2: {sum(all_f1_2) / len(all_f1_2)}")
    print(f"all_f1_4: {sum(all_f1_4) / len(all_f1_4)}")

    print(f"valid_data: {valid_data}")
