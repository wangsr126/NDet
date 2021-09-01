_base_ = [
    'fcos_1x_coco.py'
]

data_root = 'data/coco/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/instances_train2017_real-world_subset.json',
    )
)
