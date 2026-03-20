# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import importlib

# Explicit imports keep MMCV registrations deterministic and avoid relying on
# broken/missing __all__ definitions in subpackages.
_REGISTER_MODULES = [
    '.core.bbox.assigners.hungarian_assigner_3d',
    '.core.bbox.coders.nms_free_coder',
    '.core.bbox.match_costs.match_cost',
    '.datasets.nuscenes_dataset',
    '.datasets.multi_nuscenes_dataset',
    '.datasets.pipelines.loading',
    '.datasets.pipelines.transform_3d',
    '.models.backbones.vovnet',
    '.models.backbones.vovnetcp',
    '.models.dense_heads.dgcnn3d_head',
    '.models.dense_heads.detr3d_head',
    '.models.dense_heads.petr_head',
    '.models.dense_heads.petr2d_head',
    '.models.dense_heads.petrv2_head',
    '.models.dense_heads.petr_head_seg',
    '.models.dense_heads.petrv2_dnhead',
    '.models.detectors.obj_dgcnn',
    '.models.detectors.detr3d',
    '.models.detectors.petr3d',
    '.models.detectors.petr3d_seg',
    '.models.detectors.mspetr3d',
    '.models.losses.Sigmoid_ce_loss',
    '.models.necks.cp_fpn',
    '.models.utils.detr',
    '.models.utils.detr3d_transformer',
    '.models.utils.dgcnn_attn',
    '.models.utils.petr_transformer',
    '.models.utils.positional_encoding',
]

for _module in _REGISTER_MODULES:
    importlib.import_module(_module, package=__name__)
