import torch

from mmdet.core import reduce_mean, multi_apply
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.roi_heads import StandardRoIHead


@HEADS.register_module(force=True)
class NRoIHead(StandardRoIHead):
    """RoI head with noise adaptation."""
    def __init__(self, *args, adapt_cfg=None, **kwargs):
        super(NRoIHead, self).__init__(*args, **kwargs)
        self.adapt_cfg = adapt_cfg

    def forward_adapt_bboxes(self, 
                             x,
                             img_metas,
                             proposal_list,
                             gt_bboxes,
                             gt_labels,
                             adapt_cfg=None,
                             gt_bboxes_ignore=None,
                             gt_masks=None,
                             **kwargs):
        ori_bboxes_list, ori_scores_list = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=False
        )
        bboxes_list = []
        scores_list = []
        for bboxes, scores in zip(ori_bboxes_list, ori_scores_list):
            bboxes_list.append(bboxes.view(scores.size(0), bboxes.size(1)//4, 4))
            scores_list.append(scores[:, :-1])

        adapt_cfg = adapt_cfg if adapt_cfg else self.adapt_cfg

        adapted_gt_bboxes, metric_dict = self.adapt_bboxes(
            bboxes_list,
            scores_list,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            adapt_cfg=adapt_cfg,
            **kwargs
        )
        return adapted_gt_bboxes, metric_dict

    def adapt_bboxes(self,
                     bboxes,
                     scores,
                     img_metas,
                     gt_bboxes,
                     gt_labels=None,
                     gt_bboxes_ignore=None,
                     adapt_cfg=None,
                     **kwargs):
        metric_dict = {}

        adapted_gt_bboxes, _ = multi_apply(
            self._adapt_gt_bboxes_single,
            bboxes,
            scores,
            gt_bboxes,
            gt_labels,
            adapt_cfg=adapt_cfg
        )

        if 'ext_bboxes' in kwargs:
            ext_bboxes = kwargs['ext_bboxes']
            noise_level, _ = self.noise_level(gt_bboxes, ext_bboxes)
            adapted_noise_level, _ = self.noise_level(adapted_gt_bboxes, ext_bboxes)
            metric_dict.update(dict(
                noise_level=noise_level,
                adapted_noise_level=adapted_noise_level
            ))
        return adapted_gt_bboxes, metric_dict

    @torch.no_grad()
    def _adapt_gt_bboxes_single(self, bboxes, scores, gt_bboxes, gt_labels, adapt_cfg):
        # bboxes: (N, 80, 4)
        # scores: (N, 80)
        # gt_bboxes: (M, 4)
        # gt_labels: (M, )
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
                weights = (ious > iou_thr) * ratio ** 2 * rscores
            elif atype == 'exp':
                sigma = cfg.get('sigma', 0.025)
                iou_thr = cfg.get('iou_thr', 0.01)
                weights = (ious > iou_thr) * torch.exp(-(1 - ious) ** 2 / sigma) * rscores
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
        assert bboxes.size(0) == scores.size(0)
        n = bboxes.size(0)
        if bboxes.ndim == 3:
            bboxes = torch.gather(bboxes, 1, gt_labels.view(1, -1, 1).expand(n, -1, 4))
            ious = bbox_overlaps(bboxes, gt_bboxes.unsqueeze(0).expand(n, -1, -1), is_aligned=True)
        else:
            ious = bbox_overlaps(bboxes, gt_bboxes)
            bboxes = bboxes.unsqueeze(1)
        scores = torch.gather(scores, 1, gt_labels.unsqueeze(0).expand(n, -1))
        weights = _get_weight(ious, scores, adapt_cfg)
        pred_gt_bboxes = (bboxes * weights.unsqueeze(-1)).sum(0)
        s = weights.sum(0).unsqueeze(-1)
        adapted_gt_bboxes = (gt_bboxes + pred_gt_bboxes) / (1 + s)
        return adapted_gt_bboxes, s

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
