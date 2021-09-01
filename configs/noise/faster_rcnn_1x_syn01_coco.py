_base_ = [
    'faster_rcnn_r50_fpn_1x_coco.py'
]

data_root = 'data/coco/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/instances_train2017_syn01.json',
    )
)
