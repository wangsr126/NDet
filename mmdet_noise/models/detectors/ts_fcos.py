from mmdet.models import DETECTORS
from mmdet.models.detectors import FCOS
import mmcv
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import torch


@DETECTORS.register_module(force=True)
class TSFCOS(FCOS):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 teacher_cfg=None):
        super(TSFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
        tconfig = dict(
            model=dict(
                backbone=backbone,
                neck=neck,
                bbox_head=bbox_head,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                pretrained=pretrained
            ))
        tconfig = mmcv.Config(tconfig)
        cfg = teacher_cfg.cfg
        tconfig.merge_from_dict(cfg)
        tmodel = build_detector(tconfig.model)
        assert 'load_from' in tconfig
        load_checkpoint(tmodel, tconfig.load_from)
        tmodel.cfg = tconfig
        self.tmodel = tmodel

        # freeze teacher
        for p in self.tmodel.parameters():
            p.requires_grad = False
        self.tmodel.eval()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        with torch.no_grad():
            x = self.tmodel.extract_feat(img)
            adapted_gt_bboxes, metric_dict = self.tmodel.bbox_head.forward_adapt_bboxes(
                x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs
            )

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, adapted_gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, **kwargs)
        losses.update(metric_dict)
        return losses
