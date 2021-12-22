import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms

from ..builder import HEADS
from .rpn_test_mixin import RPNTestMixin
from .anchor_aas_head import AnchorAASHead

@HEADS.register_module()
class RPNAASHead(RPNTestMixin, AnchorAASHead):
    """RPN head.

        Args:
            in_channels (int): Number of channels in the input feature map.
        """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        super(RPNAASHead, self).__init__(1, in_channels, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(RPNAASHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                if torch.onnx.is_in_onnx_export():
                    # sort op will be converted to TopK in onnx
                    # and k<=3480 in TensorRT
                    _, topk_inds = scores.topk(cfg.nms_pre)
                    scores = scores[topk_inds]
                else:
                    ranked_scores, rank_inds = scores.sort(descending=True)
                    topk_inds = rank_inds[:cfg.nms_pre]
                    scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0),), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        # Skip nonzero op while exporting to ONNX
        if cfg.min_bbox_size > 0 and (not torch.onnx.is_in_onnx_export()):
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # deprecate arguments warning
        if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
            warnings.warn(
                'In rpn_proposal or test_cfg, '
                'nms_thr has been moved to a dict named nms as '
                'iou_threshold, max_num has been renamed as max_per_img, '
                'name of original arguments and the way to specify '
                'iou_threshold of NMS will be deprecated.')
        if 'nms' not in cfg:
            cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
        if 'max_num' in cfg:
            if 'max_per_img' in cfg:
                assert cfg.max_num == cfg.max_per_img, f'You ' \
                                                       f'set max_num and ' \
                                                       f'max_per_img at the same time, but get {cfg.max_num} ' \
                                                       f'and {cfg.max_per_img} respectively' \
                                                       'Please delete max_num which will be deprecated.'
            else:
                cfg.max_per_img = cfg.max_num
        if 'nms_thr' in cfg:
            assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set' \
                                                         f' iou_threshold in nms and ' \
                                                         f'nms_thr at the same time, but get' \
                                                         f' {cfg.nms.iou_threshold} and {cfg.nms_thr}' \
                                                         f' respectively. Please delete the nms_thr ' \
                                                         f'which will be deprecated.'

        dets, keep = batched_nms(proposals, scores, ids, cfg.nms)
        # gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
        return dets[:cfg.max_per_img]
#######################################################################################################
    def batched_nms(self,boxes, scores, idxs, nms_cfg, class_agnostic=False):
        """Performs non-maximum suppression in a batched fashion.

        Modified from https://github.com/pytorch/vision/blob
        /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
        In order to perform NMS independently per class, we add an offset to all
        the boxes. The offset is dependent only on the class idx, and is large
        enough so that boxes from different classes do not overlap.

        Arguments:
            boxes (torch.Tensor): boxes in shape (N, 4).
            scores (torch.Tensor): scores in shape (N, ).
            idxs (torch.Tensor): each index value correspond to a bbox cluster,
                and NMS will not be applied between elements of different idxs,
                shape (N, ).
            nms_cfg (dict): specify nms type and other parameters like iou_thr.
                Possible keys includes the following.

                - iou_thr (float): IoU threshold used for NMS.
                - split_thr (float): threshold number of boxes. In some cases the
                    number of boxes is large (e.g., 200k). To avoid OOM during
                    training, the users could set `split_thr` to a small value.
                    If the number of boxes is greater than the threshold, it will
                    perform NMS on each group of boxes separately and sequentially.
                    Defaults to 10000.
            class_agnostic (bool): if true, nms is class agnostic,
                i.e. IoU thresholding happens over all boxes,
                regardless of the predicted class.

        Returns:
            tuple: kept dets and indice.
        """
        nms_cfg_ = nms_cfg.copy()
        class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
        if class_agnostic:
            boxes_for_nms = boxes
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

        nms_type = nms_cfg_.pop('type', 'nms')
        nms_op = eval(nms_type)

        split_thr = nms_cfg_.pop('split_thr', 10000)
        # Won't split to multiple nms nodes when exporting to onnx
        if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
            dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
            boxes = boxes[keep]
            # -1 indexing works abnormal in TensorRT
            # This assumes `dets` has 5 dimensions where
            # the last dimension is score.
            # TODO: more elegant way to handle the dimension issue.
            # Some type of nms would reweight the score, such as SoftNMS
            scores = dets[:, 4]
        else:
            total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
            # Some type of nms would reweight the score, such as SoftNMS
            scores_after_nms = scores.new_zeros(scores.size())
            for id in torch.unique(idxs):
                mask = (idxs == id).nonzero(as_tuple=False).view(-1)
                dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
                total_mask[mask[keep]] = True
                scores_after_nms[mask[keep]] = dets[:, -1]
            keep = total_mask.nonzero(as_tuple=False).view(-1)
            scores, inds = scores_after_nms[keep].sort(descending=True)
            keep = keep[inds]
            boxes = boxes[keep]

        return torch.cat([boxes, scores[:, None]], -1), keep


    def NMS(self,dets,scores, thresh):

        import numpy as np
        # x1、y1、x2、y2、以及score赋值
        # （x1、y1）（x2、y2）为box的左上和右下角标
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        # 每一个候选框的面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
        order = scores.argsort()[::-1]
        # ::-1表示逆序

        temp = []
        while order.size > 0:
            i = order[0]
            temp.append(i)
            # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
            # 由于numpy的broadcast机制，得到的是向量
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.minimum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.maximum(y2[i], y2[order[1:]])

            # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 计算重叠度IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 找到重叠度不高于阈值的矩形框索引
            inds = np.where(ovr <= thresh)[0]
            # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
            order = order[inds + 1]
        return temp

    def bboxes_diou(self,boxes1, boxes2):
        import numpy as np
        '''
        cal DIOU of two boxes or batch boxes
        :param boxes1:[xmin,ymin,xmax,ymax] or
                    [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
        :param boxes2:[xmin,ymin,xmax,ymax]
        :return:
        '''

        # cal the box's area of boxes1 and boxess
        boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # cal Intersection
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1Area + boxes2Area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        # cal outer boxes
        outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
        outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
        outer = np.maximum(outer_right_down - outer_left_up, 0.0)
        outer_diagonal_line = np.square(outer[..., 0]) + np.square(outer[..., 1])

        # cal center distance
        boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
        boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
        center_dis = np.square(boxes1_center[..., 0] - boxes2_center[..., 0]) + \
                     np.square(boxes1_center[..., 1] - boxes2_center[..., 1])

        # cal diou
        dious = ious - center_dis / outer_diagonal_line

        return dious


