import torch

from mmdet.core import multi_apply, anchor_inside_flags
from mmdet.core.bbox.iou_calculators import bbox_overlaps

from mmdet.models import HEADS
from mmdet.models.dense_heads import RPNHead


@HEADS.register_module(force=True)
class RPNNHead(RPNHead):
    """RPN Noise v2 head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, collect_cfg=None, **kwargs):
        super(RPNNHead, self).__init__(in_channels, **kwargs)
        self.collect_cfg = collect_cfg

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

    def forward_collect_bboxes(self,
                               x,
                               img_metas,
                               gt_bboxes,
                               gt_labels=None,
                               gt_bboxes_ignore=None,
                               collect_cfg=None,
                               **kwargs):
        """
        forward and collect bboxes
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
        cls_scores, bbox_preds = self(x)

        proposal_list = self.collect_bboxes(
            cls_scores,
            bbox_preds,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            cfg=collect_cfg,
            **kwargs
        )

        return proposal_list

    def collect_bboxes(self,
                       cls_scores,
                       bbox_preds,
                       img_metas,
                       gt_bboxes_list,
                       gt_labels_list=None,
                       gt_bboxes_ignore_list=None,
                       cfg=None,
                       **kwargs):
        assert len(cls_scores) == len(bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        
        num_imgs = len(img_metas)
        num_levels = len(cls_scores)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        bbox_preds_list = []
        scores_list = []
        for img_id in range(num_imgs):
            assert len(anchor_list[img_id]) == len(valid_flag_list[img_id])
            anchors_per_img = torch.cat(anchor_list[img_id])
            anchors_valid_per_img = torch.cat(valid_flag_list[img_id])

            bbox_preds_per_img = []
            scores_per_img = []
            for i in range(num_levels):
                bbox_preds_per_level = bbox_preds[i][img_id].permute(1, 2, 0).reshape(-1, 4)
                cls_scores_per_level = cls_scores[i][img_id].permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                if self.use_sigmoid_cls:
                    cls_scores_per_level = cls_scores_per_level.sigmoid()
                else:
                    cls_scores_per_level = cls_scores_per_level.softmax(-1)[..., 0:1]
                
                bbox_preds_per_img.append(bbox_preds_per_level)
                scores_per_img.append(cls_scores_per_level)

            flatten_bbox_preds_per_img = torch.cat(bbox_preds_per_img)
            decoded_bbox_preds = self.bbox_coder.decode(anchors_per_img, flatten_bbox_preds_per_img)
            flatten_scores_per_img = torch.cat(scores_per_img)

            inside_flags = anchor_inside_flags(anchors_per_img, anchors_valid_per_img,
                                               img_metas[img_id]['img_shape'][:2],
                                               self.train_cfg.allowed_border)
            decoded_bbox_preds = decoded_bbox_preds[inside_flags, :]
            flatten_scores_per_img = flatten_scores_per_img[inside_flags, :]
            bbox_preds_list.append(decoded_bbox_preds)
            scores_list.append(flatten_scores_per_img)
        
        cfg = cfg if cfg else self.collect_cfg
        collected_bboxes, _ = multi_apply(
            self._collect_bboxes_single,
            bbox_preds_list,
            scores_list,
            gt_bboxes_list,
            gt_labels_list,
            cfg=cfg,
        )

        return collected_bboxes

    @torch.no_grad()
    def _collect_bboxes_single(self,
                               bbox_preds,
                               cls_scores,
                               gt_bboxes,
                               gt_labels,
                               cfg):
        assert bbox_preds.size(0) == cls_scores.size(0)
        ious = bbox_overlaps(bbox_preds, gt_bboxes)
        scores = cls_scores
        iou_thr = cfg.get('iou_thr', 0.5)
        weights = (ious > iou_thr) * scores
        num_gts = gt_bboxes.size(0)
        num_per_object = cfg.get('num_per_object', 100)
        topk_weights, topk_inds = weights.topk(num_per_object, dim=0)
        expand_bboxes = bbox_preds.unsqueeze(1).expand(-1, num_gts, -1)
        bboxes_candidate = torch.gather(expand_bboxes, dim=0, index=topk_inds.unsqueeze(-1).expand(-1, -1, 4))
        
        flatten_bboxes_candidate = bboxes_candidate.reshape(-1, 4)
        max_per_img = cfg.get('max_per_img', 1000)
        if flatten_bboxes_candidate.size(0) > max_per_img:
            flatten_weights = topk_weights.reshape(-1, 1)
            _, inds = flatten_weights.topk(max_per_img, dim=0)
            flatten_bboxes_candidate = flatten_bboxes_candidate[inds.squeeze(-1), :]
        return flatten_bboxes_candidate, flatten_bboxes_candidate.size(0)


