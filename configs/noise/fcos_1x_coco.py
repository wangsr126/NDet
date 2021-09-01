_base_ = [
    '../fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
# optimizer
optimizer = dict(
    lr=0.005, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
