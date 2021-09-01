import mmcv
import torch
from mmdet.models import DETECTORS
from mmdet.models.detectors import FasterRCNN
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint


@DETECTORS.register_module(force=True)
class TSFasterRCNN(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 teacher_cfg=None):
        super(TSFasterRCNN, self).__init__(
            backbone, rpn_head, roi_head, train_cfg, test_cfg, neck=neck, pretrained=pretrained
        )
        tconfig = dict(
            model=dict(
                backbone=backbone,
                rpn_head=rpn_head,
                roi_head=roi_head,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                neck=neck,
                pretrained=pretrained,
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
                      gt_masks=None,
                      proposals=None,
                      ext_bboxes=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

            ext_bboxes (list[Tensor]): Real ground truth bboxes without noise.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        with torch.no_grad():
            x = self.tmodel.extract_feat(img)
            # proposal_cfg = self.tmodel.train_cfg.get('rpn_proposal',
            #                                          self.tmodel.test_cfg.rpn)
            collect_bboxes = self.tmodel.rpn_head.forward_collect_bboxes(
                x, img_metas, gt_bboxes, gt_labels=gt_labels, gt_bboxes_ignore=gt_bboxes_ignore,
                collect_cfg=None, ext_bboxes=ext_bboxes,
            )
            adapted_gt_bboxes, metric_dict = self.tmodel.roi_head.forward_adapt_bboxes(x, img_metas, collect_bboxes,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 ext_bboxes=ext_bboxes,
                                                 **kwargs)


        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                adapted_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 adapted_gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        losses.update(metric_dict)

        return losses
