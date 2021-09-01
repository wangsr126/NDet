from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations


@PIPELINES.register_module(force=True)
class LoadNAnnotations(LoadAnnotations):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        with_ext_bbox (bool): Whether to parse and load extra bbox annotation.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_ext_bbox=True,
                 *args, **kwargs):
        super(LoadNAnnotations, self).__init__(*args, **kwargs)
        self.with_ext_bbox = with_ext_bbox

    def _load_ext_bboxes(self, results):
        """Private function to load clean bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        # TODO: _gt_bboxes
        results['ext_bboxes'] = ann_info['ext_bboxes'].copy()
        results['bbox_fields'].append('ext_bboxes')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super(LoadNAnnotations, self).__call__(results)
        if self.with_ext_bbox:
            results = self._load_ext_bboxes(results)
        return results
