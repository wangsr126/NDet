# Narrowing the Gap: Improved Detector Training with Noisy Location Annotations

This repository is an official implementation of the paper: [Link](https://arxiv.org/abs/2206.05708).
It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Introduction
**TL; DR.** Detectors trained with noisy location annotations would show inferior performance compared to that trained with delicate annotations. However, the noise is inevitable sometimes. Fortunately, our methods can effectively improve such degraded performance.

## Usage
### Installation
Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation and dataset preparation. Note that we use a previous version: mmdet==2.8.0.

### Noisy dataset preparation
The annotations with synthesized noise we used have been uploaded to [link1](https://drive.google.com/file/d/1rTLiEJltAp3iHVMSlV7iFvmGHuvw6Qy8/view?usp=sharing),[link2](https://drive.google.com/file/d/1x8Q8n-pcOKwocKtEeTJc10ax1k10vdgY/view?usp=sharing).
Alternatively, we provide the code for generating such noisy annotations:
```bash
python tools/add_noise.py --data-root ./data/coco --split train2017 --gamma 0.1 -o syn01
```
Then, the annotations in COCO format are presented in `./data/coco/annotations/instances_train2017_syn01.json`.

Furthermore, the annotations that we relabeled with rough bounding boxes are also public at [link](https://drive.google.com/file/d/1qDaRtz7Q1FgLYRCofHdpU123PK3He0k3/view?usp=sharing), with about 12k images and 90k instances involved.

### Training
- *Step1*: To train a detector with noisy annotations:
    ```bash
    # multi-gpu training
    tools/dist_train.sh <CONFIG_FILE> <GPU_NUM>
    ```
    Please refer to [Main Results](#Results) for more details about our `CONFIG_FILE`.

- *Step2*: To train an improved detector with noisy annotations:
    ```
    # multi-gpu training
    tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.teacher_cfg.cfg.load_from=<PRETRAIN_MODEL>
    ```
    Since we adopt Teacher-Student learning paradigm, a pre-trained teacher model is need, which can be obtained easily by the *Step1*, and the path to this model should be provided as `<PRETRAIN_MODEL>`.

## Results
### FCOS

| Detector    | dataset      | noise                | AP   | `<CONFIG_FILE>` |
|------------ | ------------ | -------------------- | ---- | --------------- |
| FCOS        | COCO         | -                    | 38.9 | [CONFIG](configs/noise/fcos_1x_coco.py) |
| FCOS        | COCO         | syn(gamma=0.05) | 37.1 | [CONFIG](configs/noise/fcos_1x_syn005_coco.py) |
| FCOS (impr) | COCO         | syn(gamma=0.05) | 37.9 | [CONFIG](configs/noise/fcos_1x_syn005_impr_coco.py) |
| FCOS        | COCO         | syn(gamma=0.1)  | 33.6 | [CONFIG](configs/noise/fcos_1x_syn01_coco.py) |
| FCOS (impr) | COCO         | syn(gamma=0.1)  | 35.6 | [CONFIG](configs/noise/fcos_1x_syn01_impr_coco.py) |
| FCOS        | COCO subset  | -                    | 22.9 | [CONFIG](configs/noise/fcos_1x_cocosubset.py) |
| FCOS        | COCO subset  | real-world           | 21.0 | [CONFIG](configs/noise/fcos_1x_real-world_cocosubset.py) |
| FCOS (impr) | COCO subset  | real-world           | 21.4 | [CONFIG](configs/noise/fcos_1x_real-world_impr_cocosubset.py) |

### Faster R-CNN
| Detector    | dataset      | noise                | AP   | `<CONFIG_FILE>` |
|------------ | ------------ | -------------------- | ---- | --------------- |
| Faster R-CNN        | COCO         | -                    | 37.8 | [CONFIG](configs/noise/faster_rcnn_1x_coco.py) |
| Faster R-CNN        | COCO         | syn(gamma=0.05) | 36.4 | [CONFIG](configs/noise/faster_rcnn_1x_syn005_coco.py) |
| Faster R-CNN (impr) | COCO         | syn(gamma=0.05) | 36.7 | [CONFIG](configs/noise/faster_rcnn_1x_syn005_impr_coco.py) |
| Faster R-CNN        | COCO         | syn(gamma=0.1)  | 33.7 | [CONFIG](configs/noise/faster_rcnn_1x_syn01_coco.py) |
| Faster R-CNN (impr) | COCO         | syn(gamma=0.1)  | 35.1 | [CONFIG](configs/noise/faster_rcnn_1x_syn01_impr_coco.py) |
| Faster R-CNN        | COCO subset  | -                    | 23.7 | [CONFIG](configs/noise/faster_rcnn_1x_cocosubset.py) |
| Faster R-CNN        | COCO subset  | real-world           | 22.2 | [CONFIG](configs/noise/faster_rcnn_1x_real-world_cocosubset.py) |
| Faster R-CNN (impr) | COCO subset  | real-world           | 22.5 | [CONFIG](configs/noise/faster_rcnn_1x_real-world_impr_cocosubset.py) |

*Caution that all the results above are evaluated on COCO `test-dev` split.*