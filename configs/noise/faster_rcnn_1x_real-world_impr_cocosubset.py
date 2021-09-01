_base_ = [
    'faster_rcnn_r50_fpn_1x_coco.py'
]
# model settings
model = dict(
    type='TSFasterRCNN',
    teacher_cfg=dict(
        cfg=dict(
            model=dict(
                type='FasterRCNN',
                rpn_head=dict(
                    type='RPNNHead',
                    collect_cfg=dict(
                        iou_thr=0.5,
                        num_per_object=100,
                        max_per_img=1000)),
                roi_head=dict(
                    type='NRoIHead',
                    adapt_cfg=dict(
                        type='step',
                        score_thr=0.05,
                        iou_thr=0.7,
                        ratio=1.0,
                    )
                ),
                test_cfg=dict(
                    rcnn=None
                ),
            ),
            load_from='work_dirs/faster_rcnn_1x_real-world_cocosubset/epoch_12.pth',
        ),
    )
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNAnnotations', with_bbox=True, with_ext_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='NFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'ext_bboxes']),
]
dataset_type = 'CocoNDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017_real-world_subset.json',
        pipeline=train_pipeline))
