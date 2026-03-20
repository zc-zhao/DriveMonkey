from cmath import isnan, nan
import json
import re
import ast
import os
import sys
from shapely.geometry.polygon import orient
from tqdm import tqdm
import pickle
sys.path.append(os.getcwd())
print(os.getcwd())
from utils import *

# precisions
def calculate_metric_det_3d(output_data: list):
    tp_num = 0
    precisions =0
    trans_err = 0
    vel_err = 0
    orient_err = 0
    scale_error =0
    all_distance = [0.5, 1, 2, 4]
    for distance in all_distance:
        all_tp =0
        all_fp =0
        for data in output_data:
            if data.get('bbox_3d_gt', None) and data['pred_3d_bbox']:
                bbox_3d_gt = np.array(data['bbox_3d_gt'])
                pred_3d_bbox = np.array(data['pred_3d_bbox'])
                assert len(bbox_3d_gt) == len(pred_3d_bbox)
                tp, fp = calculate_3d_tp_fp(pred_3d_bbox[:, :7], bbox_3d_gt[:, :7], min_distance=distance)
                all_tp += tp
                all_fp += fp
                
        precision = all_tp/(all_tp+all_fp)
        print(f'all_tp_{distance}:', all_tp)
        print(f'all_bbox_{distance}:', all_tp + all_fp)
        print(f'precisions_{distance}:', precision)
        precisions += precision
    
    for data in tqdm(output_data):
        if data.get('bbox_3d_gt', None) and data['pred_3d_bbox']:
            bbox_3d_gt = np.array(data['bbox_3d_gt'])
            pred_3d_bbox = np.array(data['pred_3d_bbox'])
            tfp = is_calculate_3d_tp(pred_3d_bbox[:, :7], bbox_3d_gt[:, :7], min_distance=0.5)
            for idx, i in enumerate(tfp):
                # 只计算tp的
                if i:
                    tp_num = tp_num+1
                    trans_err += center_distance(pred_3d_bbox[idx], bbox_3d_gt[idx])
                    vel_err += velocity_l2(pred_3d_bbox[idx], bbox_3d_gt[idx])
                    orient = angle_diff(pred_3d_bbox[idx],bbox_3d_gt[idx])
                    # orient_err += angle_diff(pred_3d_bbox[idx],bbox_3d_gt[idx])
                    if not np.isnan(orient):
                        orient_err += orient
                    scale_error += (1 - scale_iou(bbox_3d_gt[idx],pred_3d_bbox[idx]))

                
    vel_err = vel_err /  tp_num  
    orient_err = orient_err/ tp_num
    trans_err = trans_err / tp_num 
    scale_error = scale_error/tp_num
    precisions = precisions/len(all_distance)
    
    print('precisions:', precisions)
    print('vel_err:',vel_err)
    print('orient_err:',orient_err)
    print('trans_err:',trans_err)
    print('scale_error:',scale_error)


if __name__ == "__main__":
    file_path = 'Your result file path'
    output = json.load(open(file_path,'r')) if file_path.endswith('.json') else pickle.load(open(file_path,'rb'))

    calculate_metric_det_3d(output)
    print('\n')