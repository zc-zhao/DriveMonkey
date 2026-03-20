import torch
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost

from ..util import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module(force=True)
class HungarianAssigner3D(BaseAssigner):
    """One-to-one matching between predictions and GT boxes for 3D DETR heads."""

    def __init__(self, cls_cost=None, reg_cost=None, iou_cost=None, pc_range=None, **kwargs):
        self.cls_cost = build_match_cost(cls_cost or dict(type='ClassificationCost', weight=1.0))
        self.reg_cost = build_match_cost(reg_cost or dict(type='BBoxL1Cost', weight=1.0))
        self.iou_cost = build_match_cost(iou_cost) if iou_cost is not None else None
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        if gt_bboxes_ignore is not None:
            raise AssertionError('HungarianAssigner3D does not support gt_bboxes_ignore.')
        if linear_sum_assignment is None:
            raise ImportError('Please install scipy to use HungarianAssigner3D.')

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        cls_cost = self.cls_cost(cls_pred, gt_labels)
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        cost = cls_cost + reg_cost

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost.detach().cpu())
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
