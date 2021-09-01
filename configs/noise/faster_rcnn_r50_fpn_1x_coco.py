_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
