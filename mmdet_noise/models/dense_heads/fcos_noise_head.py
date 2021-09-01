import torch
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, reduce_mean
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads import FCOSHead

INF = 1e8


@HEADS.register_module(force=True)
class FCOSNHead(FCOSHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 adapt_cfg=None,
                 **kwargs):
        super().__init__(
            num_classes,
            in_channels,
            **kwargs)
        if adapt_cfg is None:
            self.adapt = False
            self.adapt_cfg = None
        else:
            self.adapt = True
            self.adapt_cfg = adapt_cfg

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg, **kwargs)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             ext_bboxes=None,
             gt_bboxes_ignore=None,
             adapt=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            ext_bboxes (list[Tensor]): Real ground truth bboxes without noise.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            adapt (None | bool): whether adapt the gt_bboxes and use the adapted
                ones to calculate losses

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        metric_dict = {}
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        if ext_bboxes:
            noise_level, variances = self.noise_level(gt_bboxes, ext_bboxes)
            metric_dict['noise_level'] = noise_level
        adapt = adapt if adapt is not None else self.adapt
        if adapt:
            adapted_gt_bboxes, s = self._adapt_gt_bboxes(all_level_points, bbox_preds, cls_scores, centernesses,
                                                        gt_bboxes, gt_labels, self.adapt_cfg)
            if ext_bboxes:
                adapted_noise_level, _ = self.noise_level(adapted_gt_bboxes, ext_bboxes)
                metric_dict['adapted_noise_level'] = adapted_noise_level
        else:
            adapted_gt_bboxes = gt_bboxes

        labels, bbox_targets = self.get_targets(all_level_points, adapted_gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if len(pos_inds) > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            centerness_denorm = max(
                reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        metric_dict.update(dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
        ))
        return metric_dict

    @torch.no_grad()
    def noise_level(self, gt_bboxes_list, ext_bboxes_list):
        gt_bboxes = torch.cat(gt_bboxes_list)
        ext_bboxes = torch.cat(ext_bboxes_list)
        ious = bbox_overlaps(gt_bboxes, ext_bboxes, is_aligned=True)
        num_gts = torch.tensor(
            len(gt_bboxes), dtype=torch.float, device=gt_bboxes.device)
        num_gts = max(reduce_mean(num_gts), 1.0)
        means = ious.sum() / num_gts
        variances = (ious - means).pow(2).sum() / num_gts
        return means, variances

    @torch.no_grad()
    def _adapt_gt_bboxes(self, points, bbox_preds, cls_scores, centernesses,
                        gt_bboxes_list, gt_labels_list, adapt_cfg):
        assert adapt_cfg is not None
        assert len(points) == len(bbox_preds)
        num_levels = len(bbox_preds)
        num_imgs = bbox_preds[0].size(0)

        flatten_points = torch.cat(points, dim=0)
        bbox_preds_list = []
        scores_list = []
        for img_id in range(num_imgs):
            bbox_preds_per_img = []
            scores_per_img = []
            for i in range(num_levels):
                # bbox_preds
                bbox_preds_per_level = bbox_preds[i][img_id].permute(1, 2, 0).reshape(-1, 4)
                if self.norm_on_bbox:
                    bbox_preds_per_level = self.strides[i] * bbox_preds_per_level
                # cls_scores
                cls_scores_per_level = cls_scores[i][img_id].permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
                # centerness
                centernesses_per_level = centernesses[i][img_id].permute(1, 2, 0).reshape(-1, 1).sigmoid()

                bbox_preds_per_img.append(bbox_preds_per_level)
                scores_per_img.append(cls_scores_per_level * centernesses_per_level)

            flatten_bbox_preds_per_img = torch.cat(bbox_preds_per_img)
            decoded_bbox_preds = distance2bbox(flatten_points, flatten_bbox_preds_per_img)
            flatten_scores_per_img = torch.cat(scores_per_img)
            bbox_preds_list.append(decoded_bbox_preds)
            scores_list.append(flatten_scores_per_img)
        adapted_gt_bboxes, s = multi_apply(
            self._adapt_gt_bboxes_single,
            bbox_preds_list,
            scores_list,
            gt_bboxes_list,
            gt_labels_list,
            adapt_cfg=adapt_cfg)
        return adapted_gt_bboxes, s

    @torch.no_grad()
    def _adapt_gt_bboxes_single(self, bbox_preds, cls_scores, gt_bboxes, gt_labels, adapt_cfg):
        def _get_weight(ious, scores, cfg):
            # bboxes: (N, 4)
            # ious: (N, k)
            # scores: (N, k)
            score_thr = cfg.get('score_thr', 0.05)
            rscores = torch.where(scores < score_thr, torch.zeros_like(scores), scores)
            atype = cfg.get('type', 'step')
            if atype == 'step':
                ratio = cfg.get('ratio', 1.0)
                iou_thr = cfg.get('iou_thr', 0.7)
                weights = (ious > iou_thr) * ratio**2 * rscores
            elif atype == 'exp':
                sigma = cfg.get('sigma', 0.025)
                iou_thr = cfg.get('iou_thr', 0.01)
                weights = (ious > iou_thr) * torch.exp(-(1 - ious)**2 / sigma) * rscores
            else:
                raise KeyError
            topk = cfg.get('topk', None)
            if topk is not None:
                # weights: (N, k)
                topk_weights, topk_idx = torch.topk(weights, k=topk, dim=0)
                new_weights = torch.zeros_like(weights)
                new_weights.scatter_(0, topk_idx, topk_weights)
                return new_weights
            else:
                return weights
        assert bbox_preds.size(0) == cls_scores.size(0)
        n = bbox_preds.size(0)
        # [num_points, num_gts]
        ious = bbox_overlaps(bbox_preds, gt_bboxes)
        scores = torch.gather(cls_scores, 1, gt_labels.unsqueeze(0).expand(n, -1))
        weights = _get_weight(ious, scores, adapt_cfg)
        pred_gt_bboxes = (bbox_preds.unsqueeze(1) * weights.unsqueeze(-1)).sum(0)
        s = weights.sum(0).unsqueeze(-1)
        adapted_gt_bboxes = (gt_bboxes + pred_gt_bboxes) / (1 + s)
        return adapted_gt_bboxes, s

    def forward_adapt_bboxes(self,
                             x,
                             img_metas,
                             gt_bboxes,
                             gt_labels=None,
                             gt_bboxes_ignore=None,
                             proposal_cfg=None,
                             **kwargs):
        """
        forward and adapt bboxes
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        cls_scores, bbox_preds, centernesses = self(x)

        adapted_gt_bboxes, metric_dict = self.adapt_bboxes(cls_scores, bbox_preds, centernesses, img_metas,
                                                           gt_bboxes, gt_labels, gt_bboxes_ignore=gt_bboxes_ignore,
                                                           **kwargs)

        return adapted_gt_bboxes, metric_dict

    def adapt_bboxes(self,
                     cls_scores,
                     bbox_preds,
                     centernesses,
                     img_metas,
                     gt_bboxes,
                     gt_labels=None,
                     gt_bboxes_ignore=None,
                     **kwargs):
        metric_dict = {}

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        adapted_gt_bboxes, _ = self._adapt_gt_bboxes(all_level_points, bbox_preds, cls_scores, centernesses,
                                                    gt_bboxes, gt_labels, self.adapt_cfg)
        if 'ext_bboxes' in kwargs:
            ext_bboxes = kwargs['ext_bboxes']
            noise_level, _ = self.noise_level(gt_bboxes, ext_bboxes)
            adapted_noise_level, _ = self.noise_level(adapted_gt_bboxes, ext_bboxes)
            metric_dict.update(dict(
                noise_level=noise_level,
                adapted_noise_level=adapted_noise_level
            ))

        return adapted_gt_bboxes, metric_dict
