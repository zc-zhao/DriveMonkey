# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core import (CameraInstance3DBoxes,LiDARInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from internvl.model.PETR.projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from PIL import Image
from mmdet.datasets.pipelines import to_tensor

try:
    from eval.nuscene.degradation import apply_image_degradation
except Exception:
    apply_image_degradation = None

@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
  
    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    
    def simple_llava_test_pts(self, x, img_metas, query, trian=True, rescale=False):
        outs = self.pts_bbox_head(x, img_metas, query)
        if not trian:
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            return bbox_results
        return outs

    def forward_llava_test(self, cams, views = None, degradation_type = None):
        # 相关的配置预定义在这里
        self.ida_aug_conf = {
        "resize_lim": (0.8, 1.0),
        "final_dim": (512, 1408),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
        }
        self.size_divisor = 32
        self.pad_val = 0
        # img = [img] if img is None else img
        # batch 处理
        # all_fuse_feature = []
        all_img_metas = []
        batch_img = []
        for i in range(len(cams)):
            # 得到整个batch image
            img, img_metas = self.process_img(cams[i], views = views, degradation_type = degradation_type)
            batch_img.append(img)
            all_img_metas.append(img_metas)
        batch_img = torch.stack(batch_img)
        img_feats = self.extract_feat(img=batch_img.to('cuda'),img_metas=[img_metas])
        # for i in range(len(img)):
        #     # 推理整个批次
        #     img_feats, img_metas = self.infer_fuse(img[i], cams[i])
        #     all_fuse_feature.append(img_feats)
        #     all_img_metas.append(img_metas)
        # all_fuse_feature = torch.cat(all_fuse_feature)
        
        # img_feature = torch.cat(all_fuse_feature)
        return img_feats, all_img_metas # bs 2500 256

    #******************petr的数据预处理***********************#
    def _sample_augmentation(self):
        # for test
        H, W = self.ida_aug_conf["H"], self.ida_aug_conf["W"]
        fH, fW = self.ida_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.ida_aug_conf["bot_pct_lim"])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate
    def _get_rot(self, h):

        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )
    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def process_img(self, cams = None, views = None, degradation_type = None):
        # petr的数据处理 
        # if degradation_type is not None and str(degradation_type).strip() != '' and apply_image_degradation is None:
        #     raise ImportError('degradation_type is set, but failed to import apply_image_degradation from eval.nuscene.degradation')
        results ={}
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        extrinsics = []
        img_metas = {'flip': False, 'pcd_horizontal_flip': False, 'pcd_vertical_flip': False,'box_type_3d':LiDARInstance3DBoxes}
        #***************************get_data_info*********************************# 
        for cam_type, cam_info in cams.items():
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = np.array(cam_info['cam_intrinsic'])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            extrinsics.append(lidar2cam_rt)  ###The extrinsics mean the tranformation from lidar to camera. If anyone want to use the extrinsics as sensor to lidar, please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and LoadMultiViewImageFromMultiSweepsFiles.
            lidar2cam_rts.append(lidar2cam_rt.T)
        results.update(
                dict(
                    lidar2img=lidar2img_rts,
                    intrinsics=cam_intrinsics,
                    extrinsics=extrinsics    
                ))
        #**************************LoadMultiViewImageFromFiles*********************#
        # img = np.stack([mmcv.imread(path['data_path'].split('Refnuscene/')[-1], 'unchanged') for cam, path in cams.items()], axis=-1)
        all_imgs = []
        for cam, path in cams.items():
            loaded_from_camera = True
            if views is not None:
                if cam in views:
                    img = mmcv.imread(osp.join('nuscenes', path['data_path'].split('Refnuscene/')[-1]), 'unchanged')
                else:
                    img = np.zeros((900, 1600,3), dtype=np.uint8)
                    loaded_from_camera = False
            else:
                img = mmcv.imread(osp.join('nuscenes', path['data_path'].split('Refnuscene/')[-1]), 'unchanged')

            should_apply_degradation = degradation_type is not None and str(degradation_type).strip() != ''
            if should_apply_degradation and loaded_from_camera and apply_image_degradation is not None and img is not None:
                # mmcv.imread 默认是 BGR，degradation 使用 PIL(RGB)，因此做一次通道转换。
                if img.ndim == 3 and img.shape[2] == 3:
                    img_rgb = img[..., ::-1].copy()
                    img_pil = Image.fromarray(img_rgb)
                    img_pil = apply_image_degradation(img_pil, degradation_type)
                    img = np.array(img_pil)[..., ::-1]
            all_imgs.append(img)
        img = np.stack(all_imgs, axis=-1)
        # to_float32 为 True
        img = img.astype(np.float32)
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        
        results['scale_factor'] = 1.0
        #***********************ResizeCropFlipImage********************************# 
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]

        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i].T for i in range(len(results['extrinsics']))] 
        #******************NormalizeMultiviewImage***********************#
        results['img'] = [mmcv.imnormalize(img, mean=np.array(self.mean, dtype=np.float32), std=np.array(self.std, dtype=np.float32), to_rgb=self.to_rgb) for img in results['img']]
        #*****************PadMultiViewImage******************************#
        if self.size_divisor is not None:
           padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]
        img_metas['img_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_size_divisor'] = self.size_divisor
        img_metas['pad_shape'] = [img.shape for img in padded_img] 
        img_metas['lidar2img'] = results['lidar2img']
        
        if isinstance(results['img'], list):
            # process multiple imgs in single frame
            imgs = [img.transpose(2, 0, 1) for img in results['img']]
            imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            results['img'] = DC(to_tensor(imgs), stack=True)
        else:
            img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        img = results['img'].data
        return img, img_metas



    