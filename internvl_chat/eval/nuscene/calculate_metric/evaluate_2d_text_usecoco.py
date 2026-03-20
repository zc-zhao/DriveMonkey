import language_evaluation
import json
import re
import ast
import time
import os
import sys
import math
# sys.path.append(os.getcwd())
# print(os.getcwd())
from utils import *
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from coco_eval import Scorer, Our_COCO, Our_COCOeval
    
def restore_bbox_qwen(box ,w =1600, h= 900):
    new_box = [num / 1000 for num in box]
    x1,y1,x2,y2 = new_box
    return [x1 * w, y1* h, x2 *w, y2 * h]

# def restore_bbox(box ,w =1600, h= 900):
#     try:
#         x1,y1,x2,y2 = box
#         return [x1 * w, y1* h, x2 *w, y2 * h]
#     except:
#         return []
def restore_bbox(bbox, w = 1600, h = 900):
    try:
        x1, y1, x2, y2 = bbox
        x1_new = x1 * w if x1>=0 else 0
        x2_new = x2 * w if x2>=0 else 0
        y1_new = (y1 * w) -(w-h) //2 if y1>=0 else 0
        y2_new = (y2 * w) - (w-h) //2 if y2>=0 else 0
        return[x1_new, y1_new, x2_new, y2_new]
    except:
        return []

def xyxy2xywh(_bbox):
    return[
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]

def extract_coordinate(text):
        # pattern = r'\[[^\]]*\]'
        pattern = r'\[.*?\]'
        coordinates_with_brackets = re.findall(pattern, text)
        coordinates = [ast.literal_eval(coord) for coord in coordinates_with_brackets]
        return coordinates
def qwen_extract_coordinate(text):
    pattern = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
    coordinates_with_brackets = re.findall(pattern, text)
    # coordinates = [list(map(int, a.replace('(', '').replace(')','').split(','))) for a in coordinates_with_brackets]
    coordinates = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in coordinates_with_brackets]
    return coordinates
    # coordinates = [ast.literal_eval(coord) for coord in coordinates_with_brackets]

def qwen_re_extract_coordinate(text):
    pattern = r'(\d+) (\d+) (\d+) (\d+)'
    coordinates_with_brackets = re.findall(pattern, text)
    if len(coordinates_with_brackets) == 0:
        pattern = r'(\d+) (\d+), (\d+) (\d+)'
        coordinates_with_brackets = re.findall(pattern, text)
    if len(coordinates_with_brackets) == 0:
        pattern = r'(\d+), (\d+), (\d+), (\d+)'
        coordinates_with_brackets = re.findall(pattern, text)
    # coordinates = [list(map(int, a.replace('(', '').replace(')','').split(','))) for a in coordinates_with_brackets]
    coordinates = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in coordinates_with_brackets]
    return coordinates

def calculate_metric_2d(output_data, cacluate_language):
    camera_view = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
    pred_result = []
    gt_result = {'images': [], 'annotations': [], "categories": [{'id': int(0), 'name': 'all'}]}
    image_info = {}
    img_id = 0
    ann_id = 0
    predicts = []
    answers = []
    COCO_evaluator = language_evaluation.CocoEvaluator(coco_types=['BLEU', 'ROUGE_L','CIDEr']) #'METEOR',
    MIOU = []
    mAP = []
    coco_mAP = []
    coco_mAP_05 = []
    coco_mAP_75 = []
    mAP_list = []
    F1_score = []
    F1_score_list = []
    ##******用于2D bbox的测试******###
    for data in tqdm(output_data):
        gen_answer = data['gen']
        gt_answer = data['gt']
        gen_answer = gen_answer[0] if isinstance(gen_answer, list) else gen_answer
        gt_answer = gt_answer[0] if isinstance(gt_answer, list) else gt_answer
        if data.get('category') is None or data['category'] !='2D_perception_location':
            if cacluate_language:
                try:
                    if gen_answer:
                        if len(gen_answer)>200:
                            gen_answer = gen_answer.split(',')[0]
                        # coco_result = COCO_evaluator.run_evaluation(gen_answer, gt_answer)
                        # coco_results.append(coco_result)
                        predicts.append(gen_answer)
                        answers.append(gt_answer)

                except:
                    print(gt_answer)
                    print(gen_answer)
                    print('\n')
        ###******************************2D框的检测，进行匹配***************************###
        else:
            # continue
            ev_pred_result = []
            ev_gt_result = {'images': [], 'annotations': [], "categories": [{'id': int(0), 'name': 'all'}]}

            gen_bbox = qwen_extract_coordinate(gen_answer)
            gt_bbox = qwen_extract_coordinate(gt_answer)
            if len(gen_bbox) == 0:
                gen_bbox = qwen_re_extract_coordinate(gen_answer)
            # gen_bbox = extract_coordinate(gen_answer)
            # gt_bbox = extract_coordinate(gt_answer)
            if len(gen_bbox)>0 or len(gt_bbox)> 0 :
                # gen_bbox = [restore_bbox(box) for box in gen_bbox]
                # gt_bbox = [restore_bbox(box) for box in gt_bbox]
                gen_bbox = [restore_bbox_qwen(box) for box in gen_bbox]
                gt_bbox = [restore_bbox_qwen(box) for box in gt_bbox]

                if isinstance(data['cam_path'], dict):
                    camera_path = data['cam_path'][re.findall(r'CAM_[\w]*', data['question'])[0]]
                elif isinstance(data['cam_path'], list):
                    camera_idx = camera_view.index(re.findall(r'CAM_[\w]*', data['question'])[0])
                    camera_path = data['cam_path'][camera_idx]

                if camera_path not in image_info:
                    img_id += 1
                    image_info[camera_path] = img_id
                    img_info = {'id':int(img_id), 'file_name': camera_path, 'width':1600, 'height':900}
                    gt_result['images'].append(img_info)
                    ev_gt_result['images'].append(img_info)

                    ann_img_id = img_id
                else:
                    ann_img_id = image_info[camera_path]
                    img_info = {'id':int(ann_img_id), 'file_name': camera_path, 'width':1600, 'height':900}
                    ev_gt_result['images'].append(img_info)
                
                for i in range(len(gen_bbox)):
                    pred_data = dict()
                    pred_data['image_id'] = ann_img_id
                    pred_data['bbox'] = xyxy2xywh(gen_bbox[i])
                    pred_data['score'] = float(1.0)
                    pred_data['category_id'] = int(0)
                    pred_result.append(pred_data)
                    ev_pred_result.append(pred_data)
                
                for i in range(len(gt_bbox)):
                    bbox = xyxy2xywh(gt_bbox[i])
                    bbox = [round(num, 2) for num in bbox]
                    area = float(bbox[2] * bbox[3])
                    ann_id += 1
                    ann = {'image_id': ann_img_id,
                    'id': int(ann_id),
                    'category_id': int(0),
                    'bbox': bbox,
                    'iscrowd': 0,
                    'area': area}
                    gt_result['annotations'].append(ann)
                    ev_gt_result['annotations'].append(ann)

                pred_scores = [1.0] * len(gen_bbox)
                iou_thresholds = np.linspace(0.5, 0.95, 10)

                try:
                    coco_gt = Our_COCO(ev_gt_result)
                    coco_det = coco_gt.loadRes(ev_pred_result)
                    cocoEval = Our_COCOeval(cocoGt=coco_gt, cocoDt=coco_det, iouType= 'bbox')
                    cocoEval.params.maxDets = [100,300, 1000]
                    # cocoEval.params.maxDets = [1, 10, 100]

                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    cocoEval.summarize()
                    stats = cocoEval.stats
                    coco_ap, coco_ap_05, coco_ap_75 = [round(num, 1) for num in stats[:3]]
                    mean_ap, mean_f1, ap_list, f1_list = compute_map_f1(gt_bbox, gen_bbox, pred_scores, iou_thresholds=iou_thresholds)
                    cost_matrix = np.zeros((len(gen_bbox), len(gt_bbox)))
                    for i, gen in enumerate(gen_bbox):
                        for j, gt in enumerate(gt_bbox):
                            cost_matrix[i, j] = 1 - bbox_overlaps(torch.FloatTensor([gen]), torch.FloatTensor([gt])).item()
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    matched_iou = []
                    for i, j in zip(row_ind, col_ind):
                        if cost_matrix[i, j] < 1: 
                            matched_iou.append(1 - cost_matrix[i, j])
                    # 计算MIOU
                    miou = np.mean(matched_iou) if matched_iou else 0
                except:
                    print(gt_bbox)
                    print(gen_bbox)
                    mean_ap, mean_f1 = 0.0, 0.0
                    miou = 0
                    ap_list = f1_list = [0.0 for _ in iou_thresholds]
                    coco_ap, coco_ap_05, coco_ap_75 = 0.0, 0.0, 0.0

                    # continue
                if miou > 0:
                    MIOU.append(miou)
                mAP.append(mean_ap)
                F1_score.append(mean_f1)
                coco_mAP.append(coco_ap)
                coco_mAP_05.append(coco_ap_05)
                coco_mAP_75.append(coco_ap_75)
                mAP_dict = {}
                F1_score_dict = {}
                for i, (ap, f1) in enumerate(zip(ap_list, f1_list)):
                    mAP_dict[iou_thresholds[i]] = ap
                    F1_score_dict[iou_thresholds[i]] = f1
                mAP_list.append(mAP_dict)
                F1_score_list.append(F1_score_dict)
            else:
                continue
    print("mean AP: {:.4f}".format(sum(mAP) / len(mAP)))
    print("coco mAP: {:.4f}".format(sum(coco_mAP) / len(coco_mAP)))
    print("coco mAP 0.5: {:.4f}".format(sum(coco_mAP_05) / len(coco_mAP_05)))
    print("coco mAP 0.75: {:.4f}".format(sum(coco_mAP_75) / len(coco_mAP_75)))
    print("F1 Score: {:.4f}".format(sum(F1_score) / len(F1_score)))
    print("mean iou 2d: {:.4f}".format(sum(MIOU) / len(MIOU)))
    mAP_dict = {}
    F1_score_dict = {}
    for key in mAP_list[0].keys():
        mAP_dict[key] = sum([x[key] for x in mAP_list]) / len(mAP_list)
    print("mAP_dict:",mAP_dict)
    for key in F1_score_list[0].keys():
        F1_score_dict[key] = sum([x[key] for x in F1_score_list]) / len(F1_score_list)
    print("F1_score_dict:",F1_score_dict)

    coco_gt = Our_COCO(gt_result)
    coco_det = coco_gt.loadRes(pred_result)
    cocoEval = Our_COCOeval(cocoGt=coco_gt, cocoDt=coco_det, iouType= 'bbox')
    cocoEval.params.maxDets = [100,300, 1000]
    # cocoEval.params.maxDets = [1, 10, 100]

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats = cocoEval.stats
    coco_ap, coco_ap_05, coco_ap_75 = stats[:3]
    print("coco mAP: {:.4f}".format(coco_ap))
    print("coco mAP 0.5: {:.4f}".format(coco_ap_05))
    print("coco mAP 0.75: {:.4f}".format(coco_ap_75))

    if cacluate_language:
        all_metric = COCO_evaluator.run_evaluation(predicts, answers)
        Bleu = []
        for key, value in all_metric.items():
            if 'Bleu' in key:
                Bleu.append(value)
            elif 'CIDEr' in key:
                log_cider = math.log(value + 1, 10)
        if len(Bleu) > 0 :
            all_metric['Bleu'] = sum(Bleu) / len(Bleu)
        if 'CIDEr' in all_metric:
            all_metric['CIDEr_log'] = log_cider
        print(all_metric)
        all_preds = {i: [predicts[i]] for i in range(len(predicts))}
        all_gts = {i: [answers[i]] for i in range(len(answers))}
        scorer = Scorer(all_preds, all_gts)
        scorer.compute_scores()

if __name__ == "__main__":
    file_path = 'Your result json file path'
    output = json.load(open(file_path,'r'))
    calculate_metric_2d(output, True)
