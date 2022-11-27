# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import Conv2d, ConvModule
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor, nn
from torch.nn import Conv1d

from mmdet3d.models.utils import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import (Det3DDataSample, bbox_overlaps_3d,
                                center_to_corner_box2d, xywhr2xyxyr)
from ..layers import circle_nms, nms_bev


@MODELS.register_module()
class CenterFormHead(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        in_channels (list[int] | int, optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        bbox_coder (dict, optional): Bbox coder configs. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        loss_cls (dict, optional): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict, optional): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
        norm_bbox (bool): Whether normalize the bbox predictions.
            Defaults to True.
        train_cfg (dict, optional): Train-time configs. Default: None.
        test_cfg (dict, optional): Test-time configs. Default: None.
        init_cfg (dict, optional): Config for initialization.
    """

    def __init__(self,
                 in_channels: Union[List[int], int] = 256,
                 tasks: Optional[List[dict]] = None,
                 transformer_embed_dim=384,
                 transformer_decoder: Optional[dict] = None,
                 bbox_coder: Optional[dict] = None,
                 common_heads: dict = dict(),
                 loss_cls: dict = dict(
                     type='mmdet.GaussianFocalLoss',
                     reduction='mean',
                     loss_weight=1),
                 loss_bbox: dict = dict(
                     type='mmdet.L1Loss', reduction='none', loss_weight=2),
                 loss_corner=dict(
                     type='mmdet.MSELoss', reduction='mean', loss_weight=1),
                 loss_iou=dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0,
                     reduction='mean',
                     loss_weight=1),
                 separate_head: dict = dict(
                     type='mmdet.SeparateHead',
                     init_bias=-2.19,
                     final_kernel=3),
                 share_conv_channel: int = 64,
                 num_center_proposals: int = 500,
                 num_heatmap_convs: int = 2,
                 num_cornermap_convs: int = 2,
                 conv_cfg: dict = dict(type='Conv2d'),
                 norm_cfg: dict = dict(type='BN2d'),
                 bias: str = 'auto',
                 norm_bbox: bool = True,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterFormHead, self).__init__(init_cfg=init_cfg, **kwargs)

        # TODO we should rename this variable,
        # for example num_classes_per_task ?
        # {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone']}]
        # TODO seems num_classes is useless
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox
        self.num_center_proposals = num_center_proposals

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)
        self.loss_corner = MODELS.build(loss_corner)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = ConvModule(
            transformer_embed_dim,
            share_conv_channel,
            kernel_size=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        # transformer decoder
        self.fpn_feats_proj = nn.ModuleList([
            Conv2d(self.in_channels, transformer_embed_dim, kernel_size=1)
            for _ in range(3)
        ])
        self.center_feat_proj = Conv1d(
            self.in_channels, transformer_embed_dim, kernel_size=1)

        self.transformer_decoder = MODELS.build(transformer_decoder)
        self.pos_proj = nn.Linear(2, transformer_embed_dim)

        self.task_heads = nn.ModuleList()

        # heatmap_head does not use the aggregated features from
        # DeformableDetrTransformerDecoder
        heatmap_head = copy.deepcopy(separate_head)
        heatmap_head['conv_cfg'] = dict(type='Conv2d')
        heatmap_head['norm_cfg'] = dict(type='BN2d')
        heatmap_head['final_kernel'] = 3
        heatmap_head.update(
            in_channels=in_channels,
            heads=dict(
                center_heatmap=(sum(num_classes), num_heatmap_convs),
                corner_heatmap=(1, num_cornermap_convs)),
            head_conv=share_conv_channel)
        self.heatmap_head = MODELS.build(heatmap_head)
        # separate_head.update(
        #     in_channels=in_channels,
        #     heads=dict(),
        #     head_conv=share_conv_channel,
        #     num_cls=1)
        # self.corner_heatmap_head = MODELS.build(separate_head)

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            separate_head.update(
                in_channels=share_conv_channel,
                heads=heads,
                head_conv=share_conv_channel)
            self.task_heads.append(MODELS.build(separate_head))

    def forward(self, x: Tensor) -> dict:
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(
        self,
        batch_gt_instances_3d: List[InstanceData],
    ) -> Tuple[List[Tensor]]:
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and\
                ``labels_3d`` attributes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the
                    position of the valid boxes.
                - list[torch.Tensor]: Masks indicating which
                    boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks, corner_heatmaps = multi_apply(
            self.get_targets_single, batch_gt_instances_3d)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose heatmaps
        corner_heatmaps = list(map(list, zip(*corner_heatmaps)))
        corner_heatmaps = [torch.stack(hms_) for hms_ in corner_heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks, corner_heatmaps

    def get_targets_single(self,
                           gt_instances_3d: InstanceData) -> Tuple[Tensor]:
        """Generate training targets for a single sample.

        Args:
            gt_instances_3d (:obj:`InstanceData`): Gt_instances of
                single data sample. It usually includes
                ``bboxes_3d`` and ``labels_3d`` attributes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
        gt_labels_3d = gt_instances_3d.labels_3d
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks, corner_heatmaps = [], [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))
            corner_heatmap = torch.zeros(
                (1, feature_map_size[1], feature_map_size[0]),
                dtype=torch.float32,
                device=device)

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    radius = radius // 2
                    # # draw four corner and center TODO: use torch
                    rot = task_boxes[idx][k][6]
                    corner_keypoints = center_to_corner_box2d(
                        center.unsqueeze(0).cpu().numpy(),
                        torch.tensor([[width, length]],
                                     dtype=torch.float32).numpy(),
                        angles=rot,
                        origin=0.5)
                    corner_keypoints = torch.from_numpy(corner_keypoints).to(
                        center)

                    draw_gaussian(corner_heatmap[0], center_int, radius)
                    draw_gaussian(
                        corner_heatmap[0],
                        (corner_keypoints[0, 0] + corner_keypoints[0, 1]) / 2,
                        radius)
                    draw_gaussian(
                        corner_heatmap[0],
                        (corner_keypoints[0, 2] + corner_keypoints[0, 3]) / 2,
                        radius)
                    draw_gaussian(
                        corner_heatmap[0],
                        (corner_keypoints[0, 0] + corner_keypoints[0, 3]) / 2,
                        radius)
                    draw_gaussian(
                        corner_heatmap[0],
                        (corner_keypoints[0, 1] + corner_keypoints[0, 2]) / 2,
                        radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    # vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0)
                    ])

            heatmaps.append(heatmap)
            corner_heatmaps.append(corner_heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks, corner_heatmaps

    def loss(self, pts_feats: List[Tensor],
             batch_data_samples: List[Det3DDataSample], *args,
             **kwargs) -> Dict[str, Tensor]:
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch. It
                contains three level features: `norm`, `down`, `up`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict: Losses of each branch.
        """
        outs = dict()
        outs.update(self.heatmap_head(pts_feats[-1]))
        outs['center_heatmap'] = torch.sigmoid(outs['center_heatmap'])
        outs['corner_heatmap'] = torch.sigmoid(outs['corner_heatmap'])

        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)
        # masks: list[torch.Tensor], len(masks)==num_task_heads,
        # Tensor.shape=(batch, max_obj)
        heatmaps, anno_boxes, gt_inds, gt_masks, corner_heatmaps = self.get_targets(  # noqa: E501
            batch_gt_instance_3d)

        batch, num_cls, H, W = outs['center_heatmap'].size()

        scores, labels = torch.max(
            outs['center_heatmap'].reshape(batch, num_cls, H * W),
            dim=1)  # b,H*W

        if self.heatmap_head.corner_heatmap.training:
            # TODO: ``gt_inds`` should include different heads.
            # hard code here since model on waymo only has one task head.
            batch_id_gt = torch.meshgrid(
                torch.arange(batch),
                torch.arange(gt_inds[0].shape[1]))[0].to(labels)

            scores[batch_id_gt,
                   gt_inds[0]] = scores[batch_id_gt, gt_inds[0]] + gt_masks[0]
            order = scores.sort(1, descending=True)[1]
            # The reference position about the 500 predictions
            order = order[:, :self.num_center_proposals]
            scores[batch_id_gt,
                   gt_inds[0]] = scores[batch_id_gt, gt_inds[0]] - gt_masks[0]
        else:
            order = scores.sort(1, descending=True)[1]
            order = order[:, :self.num_center_proposals]

        # scores = torch.gather(scores, 1, order)
        # labels = torch.gather(labels, 1, order)
        # mask = scores > self.score_threshold

        # expand_order = order.unsqueeze(-1).expand(batch,
        #                                           self.num_center_proposals,
        #                                           anno_boxes[0].shape[-1])
        # anno_boxes[0] = torch.gather(anno_boxes[0], 1, expand_order)

        center_feat = (pts_feats[-1].reshape(batch, -1, H * W).transpose(
            2, 1).contiguous()[batch_id_gt, order])  # B, 500, C

        # create position embedding for each center
        y_coor = order // W
        x_coor = order - y_coor * W
        y_coor, x_coor = y_coor.to(center_feat), x_coor.to(center_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        center_pos = torch.stack([x_coor, y_coor], dim=2)
        center_pos_embed = self.pos_proj(center_pos)

        mlvl_feats = []
        for i, feat in enumerate(pts_feats):
            mlvl_feats.append(self.fpn_feats_proj[i](feat))

        # run transformer
        mlvl_feats = torch.cat(
            (
                mlvl_feats[0].reshape(batch, -1, (H * W) // 4).transpose(
                    2, 1).contiguous(),
                mlvl_feats[1].reshape(batch, -1, (H * W) // 16).transpose(
                    2, 1).contiguous(),
                mlvl_feats[2].reshape(batch, -1, H * W).transpose(
                    2, 1).contiguous(),
            ),
            dim=1,
        )  # B ,sum(H*W), C
        spatial_shapes = torch.as_tensor(
            [(H, W), (H // 2, W // 2), (H // 4, W // 4)],
            dtype=torch.long,
            device=center_feat.device,
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))

        levels = len(pts_feats)
        # reference_points = center_pos[:, :, None, :]

        center_feat = self.center_feat_proj(center_feat.transpose(
            2, 1)).transpose(2, 1).contiguous()

        center_proposal_feat, _ = self.transformer_decoder(
            query=center_feat,
            key=None,
            value=mlvl_feats,
            query_pos=center_pos_embed,
            key_padding_mask=None,
            reference_points=center_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=torch.ones(batch, levels, 2).to(center_feat))

        # hard code here. There is only one task head in Waymo.
        outs.update(self(center_proposal_feat.transpose(2, 1).contiguous())[0])

        # TODO: check labels
        anno_boxes, selected_mask, _ = self.get_corresponding_box(
            order, gt_inds[0], gt_masks[0],
            labels[:, :self.num_center_proposals], anno_boxes[0])

        losses = self.loss_by_feat([outs], batch_gt_instance_3d, heatmaps,
                                   corner_heatmaps, [anno_boxes],
                                   [selected_mask], order, gt_inds,
                                   labels[:, :self.num_center_proposals])

        return losses

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], heatmaps,
                     corner_heatmaps, anno_boxes, masks, order,
                     heatmap_gt_inds, heatmap_pos_labels, *args, **kwargs):
        """Loss function for CenterHead.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and\
                ``labels_3d`` attributes.

        Returns:
            dict[str,torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # Heatmap focal loss
            # TODO: Fast focal loss in CenterForm
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict['center_heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))

            # Regression loss for dimension, offset, height, rotation
            target_box = anno_boxes[task_id]
            preds_dict['anno_box'] = torch.cat(
                (preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                 preds_dict['rot']),
                dim=1)

            gt_num = masks[task_id].float().sum()
            pred = preds_dict['anno_box'].permute(0, 2, 1).contiguous()
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()

            isnotnan = (~torch.isnan(target_box)).float()
            mask_bbox_loss = mask * isnotnan
            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask_bbox_loss * mask_bbox_loss.new_tensor(
                code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(gt_num + 1e-4))

            # IoU loss
            decoded_pred_bboxes = self.decode_pred_bboxes(
                preds_dict['anno_box'], order,
                preds_dict['center_heatmap'].shape[-2],
                preds_dict['center_heatmap'].shape[-1])
            decoded_pred_bboxes = decoded_pred_bboxes.reshape(-1, 7)
            gt_bboxes = self.decode_gt_bboxes(
                target_box, order, preds_dict['center_heatmap'].shape[-2],
                preds_dict['center_heatmap'].shape[-1])
            gt_bboxes = gt_bboxes.reshape(-1, 7)

            # TODO: check it
            iou_targets = bbox_overlaps_3d(
                decoded_pred_bboxes,
                gt_bboxes)[range(decoded_pred_bboxes.shape[0]),
                           range(gt_bboxes.shape[0])]
            isnotnan = (~torch.isnan(iou_targets)).float()
            mask = masks[task_id].reshape(-1)
            mask_iou_loss = mask * isnotnan
            iou_targets = 2 * iou_targets - 1
            loss_iou = self.loss_iou(
                preds_dict['iou'].reshape(-1),
                iou_targets,
                mask_iou_loss,
                avg_factor=(gt_num + 1e-4))

            # Corner loss
            mask_corner_loss = corner_heatmaps[task_id] > 0
            num_corners = mask_corner_loss.float().sum().item()
            loss_corner = self.loss_corner(
                preds_dict['corner_heatmap'],
                corner_heatmaps[task_id],
                mask_corner_loss,
                avg_factor=(num_corners + 1e-4))

            loss_dict[f'task{task_id}.loss_center'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            loss_dict[f'task{task_id}.loss_iou'] = loss_iou
            loss_dict[f'task{task_id}.loss_corner'] = loss_corner

        return loss_dict

    def get_corresponding_box(self, x_ind, y_ind, y_mask, y_cls, target_box):
        # find the id in y which has the same ind in x

        # x_ind: sorted order about prediction score. [bs, 500]
        # y_ind: the position index of gt in [bs, 500]. index is in range
        # [0, H*W）
        # y_mask: Whether it's a gt. It's a mask. [bs, 500]
        # y_cls: the true label in the 500 objects. [bs, 500]
        # target_box,: the encodered targets [4, 500, 10]. No sorted.
        select_target = torch.zeros(x_ind.shape[0], x_ind.shape[1],
                                    target_box.shape[2]).to(target_box)
        select_mask = torch.zeros_like(x_ind).to(y_mask)
        select_cls = torch.zeros_like(x_ind).to(y_cls)

        for i in range(x_ind.shape[0]):
            idx = torch.arange(y_ind[i].shape[-1]).to(x_ind)
            idx = idx[y_mask[i]]
            box_cls = y_cls[i][y_mask[i]]
            valid_y_ind = y_ind[i][y_mask[i]]
            match = (
                x_ind[i].unsqueeze(1) == valid_y_ind.unsqueeze(0)).nonzero()
            select_target[i, match[:, 0]] = target_box[i, idx[match[:, 1]]]  #
            select_mask[i, match[:, 0]] = 1
            select_cls[i, match[:, 0]] = box_cls[match[:, 1]]

        return select_target, select_mask, select_cls

    def decode_pred_bboxes(self, pred_boxs, order, H, W):
        batch = pred_boxs.shape[0]
        obj_num = order.shape[1]
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(pred_boxs)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(pred_boxs)

        batch_id = np.indices((batch, obj_num))[0]
        batch_id = torch.from_numpy(batch_id).to(order)
        xs = xs.view(batch, H * W)[batch_id,
                                   order].unsqueeze(1) + pred_boxs[:, 0:1]
        ys = ys.view(batch, H * W)[batch_id,
                                   order].unsqueeze(1) + pred_boxs[:, 1:2]

        xs = xs * self.train_cfg['out_size_factor'] * self.train_cfg[
            'voxel_size'][0] + self.train_cfg['point_cloud_range'][0]
        ys = ys * self.train_cfg['out_size_factor'] * self.train_cfg[
            'voxel_size'][1] + self.train_cfg['point_cloud_range'][1]

        rot = torch.atan2(pred_boxs[:, 6:7], pred_boxs[:, 7:8])
        pred = torch.cat(
            [xs, ys, pred_boxs[:, 2:3],
             torch.exp(pred_boxs[:, 3:6]), rot],
            dim=1)

        return torch.transpose(pred, 1, 2).contiguous()  # B M 7

    def decode_gt_bboxes(self, gt_boxs, order, H, W):
        batch = gt_boxs.shape[0]
        obj_num = order.shape[1]
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(gt_boxs)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(gt_boxs)

        batch_id = np.indices((batch, obj_num))[0]
        batch_id = torch.from_numpy(batch_id).to(order)

        batch_gt_dim = torch.exp(gt_boxs[..., 3:6])
        batch_gt_hei = gt_boxs[..., 2:3]
        batch_gt_rot = torch.atan2(gt_boxs[..., -2:-1], gt_boxs[..., -1:])
        xs = xs.view(batch, H * W)[batch_id, order].unsqueeze(2) + gt_boxs[...,
                                                                           0:1]
        ys = ys.view(batch, H * W)[batch_id, order].unsqueeze(2) + gt_boxs[...,
                                                                           1:2]

        xs = xs * self.train_cfg['out_size_factor'] * self.train_cfg[
            'voxel_size'][0] + self.train_cfg['point_cloud_range'][0]
        ys = ys * self.train_cfg['out_size_factor'] * self.train_cfg[
            'voxel_size'][1] + self.train_cfg['point_cloud_range'][1]

        batch_box_targets = torch.cat(
            [xs, ys, batch_gt_hei, batch_gt_dim, batch_gt_rot], dim=-1)

        return batch_box_targets  # B M 7

    def predict(self,
                pts_feats: Dict[str, torch.Tensor],
                batch_data_samples: List[Det3DDataSample],
                rescale=True,
                **kwargs) -> List[InstanceData]:
        """
        Args:
            pts_feats (dict): Point features..
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes meta information of data.
            rescale (bool): Whether rescale the resutls to
                the original scale.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData contains 3d Bounding boxes and corresponding
            scores and labels.
        """
        outs = dict()
        outs.update(self.heatmap_head(pts_feats[-1]))
        outs['center_heatmap'] = torch.sigmoid(outs['center_heatmap'])
        outs['corner_heatmap'] = torch.sigmoid(outs['corner_heatmap'])

        batch, num_cls, H, W = outs['center_heatmap'].size()
        scores, labels = torch.max(
            outs['center_heatmap'].reshape(batch, num_cls, H * W),
            dim=1)  # b,H*W

        order = scores.sort(1, descending=True)[1]
        order = order[:, :self.num_center_proposals]
        outs['order'] = order
        outs['scores'] = torch.gather(scores, 1, order)
        outs['clses'] = torch.gather(labels, 1, order)

        batch_id_gt = torch.meshgrid(
            torch.arange(batch),
            torch.arange(self.num_center_proposals))[0].to(labels)

        center_feat = (pts_feats[-1].reshape(batch, -1, H * W).transpose(
            2, 1).contiguous()[batch_id_gt, order])  # B, 500, C

        # create position embedding for each center
        y_coor = order // W
        x_coor = order - y_coor * W
        y_coor, x_coor = y_coor.to(center_feat), x_coor.to(center_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        center_pos = torch.stack([x_coor, y_coor], dim=2)
        center_pos_embed = self.pos_proj(center_pos)

        mlvl_feats = []
        for i, feat in enumerate(pts_feats):
            mlvl_feats.append(self.fpn_feats_proj[i](feat))

        # run transformer
        mlvl_feats = torch.cat(
            (
                mlvl_feats[0].reshape(batch, -1, (H * W) // 4).transpose(
                    2, 1).contiguous(),
                mlvl_feats[1].reshape(batch, -1, (H * W) // 16).transpose(
                    2, 1).contiguous(),
                mlvl_feats[2].reshape(batch, -1, H * W).transpose(
                    2, 1).contiguous(),
            ),
            dim=1,
        )  # B ,sum(H*W), C
        spatial_shapes = torch.as_tensor(
            [(H, W), (H // 2, W // 2), (H // 4, W // 4)],
            dtype=torch.long,
            device=center_feat.device,
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))

        levels = len(pts_feats)
        # reference_points = center_pos[:, :, None, :]

        center_feat = self.center_feat_proj(center_feat.transpose(
            2, 1)).transpose(2, 1).contiguous()

        center_proposal_feat, _ = self.transformer_decoder(
            query=center_feat,
            key=None,
            value=mlvl_feats,
            query_pos=center_pos_embed,
            key_padding_mask=None,
            reference_points=center_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=torch.ones(batch, levels, 2).to(center_feat))

        # hard code here. There is only one task head in Waymo.
        outs.update(self(center_proposal_feat.transpose(2, 1).contiguous())[0])

        batch_size = len(batch_data_samples)
        batch_input_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_input_metas.append(metainfo)

        results_list = self.predict_by_feat([outs],
                                            batch_input_metas,
                                            rescale=rescale,
                                            **kwargs)
        return results_list

    def predict_by_feat(self, preds_dicts: Tuple[List[dict]],
                        batch_input_metas: List[dict], *args,
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_input_metas (list[dict]): Meta info of multiple
                inputs.

        Returns:
            list[:obj:`InstanceData`]: Instance prediction
            results of each sample after the post process.
            Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (:obj:`LiDARInstance3DBoxes`): Prediction
                  of bboxes, contains a tensor with shape
                  (num_instances, 7) or (num_instances, 9), and
                  the last 2 dimensions of 9 is
                  velocity.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_heatmap = preds_dict['center_heatmap']
            bs, _, H, W = batch_heatmap.size()

            batch_reg = preds_dict['reg'].transpose(2, 1).contiguous()
            batch_hei = preds_dict['height'].transpose(2, 1).contiguous()

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict['dim']).transpose(
                    2, 1).contiguous()
            else:
                batch_dim = preds_dict['dim'].transpose(2, 1).contiguous()

            batch_rots = preds_dict['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict['rot'][:, 1].unsqueeze(1)
            batch_rot = torch.atan2(batch_rots,
                                    batch_rotc).transpose(2, 1).contiguous()

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(bs, 1, 1).to(batch_heatmap)
            xs = xs.view(1, H, W).repeat(bs, 1, 1).to(batch_heatmap)
            obj_num = preds_dict['order'].shape[1]
            batch_id = np.indices((bs, obj_num))[0]
            batch_id = torch.from_numpy(batch_id).to(preds_dict['order'])

            xs = xs.view(bs, -1, 1)[batch_id,
                                    preds_dict['order']] + batch_reg[:, :, 0:1]
            ys = ys.view(bs, -1, 1)[batch_id,
                                    preds_dict['order']] + batch_reg[:, :, 1:2]

            xs = xs * self.test_cfg['out_size_factor'] * self.test_cfg[
                'voxel_size'][0] + self.bbox_coder.pc_range[0]
            ys = ys * self.test_cfg['out_size_factor'] * self.test_cfg[
                'voxel_size'][1] + self.bbox_coder.pc_range[1]

            final_box_preds = torch.cat(
                [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
            final_scores = preds_dict['scores']
            final_preds = preds_dict['clses']

            # use score threshold
            if self.test_cfg['score_threshold'] is not None:
                thresh_mask = final_scores > self.test_cfg['score_threshold']

            # pose center restriction
            post_center_range = torch.tensor(
                self.bbox_coder.post_center_range, device=batch_heatmap.device)
            mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
            predictions_dicts = []
            for i in range(bs):
                cmask = mask[i, :]
                if self.test_cfg['score_threshold']:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }
                predictions_dicts.append(predictions_dict)

            # NMS
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(bs):
                    boxes3d = predictions_dicts[i]['bboxes']
                    scores = predictions_dicts[i]['scores']
                    labels = predictions_dicts[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                batch_reg_preds = [box['bboxes'] for box in predictions_dicts]
                batch_cls_preds = [box['scores'] for box in predictions_dicts]
                batch_cls_labels = [box['labels'] for box in predictions_dicts]
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels,
                                             batch_input_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            temp_instances = InstanceData()
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = batch_input_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            ret_list.append(temp_instances)
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
