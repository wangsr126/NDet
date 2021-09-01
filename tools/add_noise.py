from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pylab as plt
import os.path as osp
import json
import argparse
import warnings
from copy import deepcopy
from progressbar import *
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def parse_args():
    parser = argparse.ArgumentParser(description='Add Noise to Coco Annotation')
    parser.add_argument('--data-root', default='data/coco', help='root path to coco dataset')
    parser.add_argument('--split', default='train2017', type=str, help='split of dataset')
    parser.add_argument('--gamma', default='0.1', type=str, help="variance of noise added to bbox, e.g., '0.1'/'0.05'")
    parser.add_argument(
        '--force-replace', action='store_true', help='if set, new annotation will be generated no matter '
                                                     'whether it exists')
    parser.add_argument(
        '-o', '--out_suffix', type=str, default='noise', help='add suffix to the output annotations file name'
    )
    parser.add_argument(
        '--display', action='store_true', help='whether display')
    args = parser.parse_args()

    return args


class MovingAvg(object):
    def __init__(self):
        self.list = []
        self.count = 0

    def add(self, item):
        self.list.append(item)
        self.count += 1

    def clear(self):
        self.list = []
        self.count = 0

    def avg(self):
        if self.count == 0:
            warnings.warn('Count = 0!')
            return 0
        else:
            return sum(self.list) / self.count


def showAnns(anns):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    if len(anns) == 0:
        return 0
    
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for i, ann in enumerate(anns):
        c = (np.array([i*75/255%1, (i+40)*45/255%1, (75-i)*30/255%1]) * 0.6 + 0.4).tolist()

        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

        p = PatchCollection(polygons,
                            facecolor='none',
                            edgecolors=color,
                            linewidths=1)
        ax.add_collection(p)


def add_noise_single(ann, img_info, gamma):
    def _check(bbox, area, img_info):
        x1, y1, w, h = bbox
        inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
        if inter_w * inter_h == 0:
            return False
        if area <= 0 or w < 1 or h < 1:
            return False
        return True

    ann_tmp = deepcopy(ann)
    bbox_reserved = ann_tmp['bbox']

    if ann_tmp.get('ignore', False) or not _check(ann_tmp['bbox'], ann_tmp['area'], img_info):
        # do not add noise for invalid bbox, just copy original bbox
        bbox_tmp = ann_tmp['bbox']
    else:
        # add noise until noised bbox valid
        x1, y1, w, h = ann_tmp['bbox']
        while True:
            # add noise
            _x1 = x1 + np.random.randn(1).item() * w * gamma
            _y1 = y1 + np.random.randn(1).item() * h * gamma
            _x2 = x1 + w + np.random.randn(1).item() * w * gamma
            _y2 = y1 + h + np.random.randn(1).item() * h * gamma
            bbox_tmp = [round(a, 2) for a in [_x1, _y1, _x2 - _x1, _y2 - _y1]]
            if _check(bbox_tmp, ann_tmp['area'], img_info):
                break
            else:
                print("encounter invalid annotation, img_id={}!".format(img_info['id']))

    # original bbox
    ann_tmp['_bbox'] = bbox_reserved
    # noised bbox
    ann_tmp['bbox'] = bbox_tmp
    return ann_tmp


def add_noise(ann_file, ann_mod_file, gamma):

    with open(ann_file, 'r') as f:
        dataset = json.load(f)

    img_info_list = {}
    for img_info in dataset['images']:
        img_info_list[img_info['id']] = img_info

    dataset_mod = deepcopy(dataset)
    dataset_mod['annotations'] = []

    pbar = ProgressBar(maxval=len(dataset['annotations'])).start()
    for i, ann in enumerate(dataset['annotations']):
        image_id = ann['image_id']
        img_info = img_info_list[image_id]

        ann_tmp = add_noise_single(ann, img_info, gamma)

        dataset_mod['annotations'].append(ann_tmp)
        pbar.update(i+1)

    pbar.finish()
    with open(ann_mod_file, 'w') as f:
        json.dump(dataset_mod, f)


def constrast_show(anns, img_root='./data/coco/train2017', cache_root='./cache'):
    coco_list = [COCO(ann) for ann in anns]
    ann_nums = len(coco_list)

    img_ids = coco_list[0].getImgIds()

    for ids in img_ids:
        img = coco_list[0].loadImgs(ids)[0]
        I = io.imread(os.path.join(img_root, img['file_name']))

        for i, coco in enumerate(coco_list):
            ann_ids = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(ann_ids)
            plt.subplot(1, ann_nums, i+1)
            plt.imshow(I)
            plt.axis('off')
            showAnns(anns)

        plt.savefig(os.path.join(cache_root, "display_{}.png".format(ids)), dpi=600)
        plt.clf()


def main():
    args = parse_args()
    gamma = float(args.gamma)
    data_root = args.data_root
    split = args.split
    ann_file = osp.join(data_root, 'annotations', 'instances_{}.json'.format(split))
    suffix = args.suffix
    ann_mod_file = osp.join(data_root, 'annotations', 'instances_{}_{}.json'.format(
        split, suffix))

    if not os.path.isfile(ann_file):
        raise FileNotFoundError('{} not found!'.format(ann_file))
    if os.path.isfile(ann_mod_file) and not args.force_replace:
        warnings.warn("'{}' has already existed, skip!".format(ann_mod_file))
    else:
        add_noise(ann_file, ann_mod_file, gamma)

    if args.display:
        ann_files = [ann_file, ann_mod_file]
        constrast_show(ann_files, img_root=os.path.join(data_root, split))


if __name__ == '__main__':
    main()
