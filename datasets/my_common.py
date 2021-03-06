import os.path as osp
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
import numpy as np
from torch.utils.data import Dataset
import math
import pickle
import random
import cv2
from copy import deepcopy
import scipy
import scipy.misc
import torch
import pycocotools.mask as mask_util
import datasets.my_transforms as MyT
from torch.utils.data import ConcatDataset
from os.path import join

def rot2DPts(x, y, rotMat):
    new_x = rotMat[0, 0] * x + rotMat[0, 1] * y + rotMat[0, 2]
    new_y = rotMat[1, 0] * x + rotMat[1, 1] * y + rotMat[1, 2]
    return new_x, new_y


def smpl_common_transforms(image_set):
    normalize = MyT.Compose([
        MyT.ToTensor(),
        MyT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return MyT.Compose([
            MyT.RandomHorizontalFlip(),
            MyT.RandomSelect(
                MyT.RandomResize(scales, max_size=1333),
                MyT.Compose([
                    MyT.RandomResize([400, 500, 600]),
                    MyT.RandomSizeCrop(384, 600),
                    MyT.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
            MyT.AddSMPLKeys(),
        ])

    if image_set == 'eval':
        return MyT.Compose([
            MyT.RandomResize([800], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


FLOAT_DTYPE = np.float32
INT_DTYPE = np.int64


def get_default(k, num_persons):
    default_shape = {'labels': lambda n: np.ones(n).astype(INT_DTYPE),
                     'kpts3d': lambda n: np.zeros((n, 24, 4), dtype=FLOAT_DTYPE),
                     'pose': lambda n: np.zeros((n, 24, 3), dtype=FLOAT_DTYPE),  # Theta in smpl model
                     'shape': lambda n: np.zeros((n, 10), dtype=FLOAT_DTYPE),  # Beta in smpl model
                     'trans': lambda n: np.zeros((n, 3), dtype=FLOAT_DTYPE),  #
                     'has_smpl': lambda n: np.zeros(n, dtype=INT_DTYPE),
                     'bboxes': lambda n: np.zeros((n, 4), dtype=INT_DTYPE),
                     'kpts2d': lambda n: np.zeros((n, 24, 3), dtype=INT_DTYPE),
                     }
    return default_shape[k](num_persons)


class SMPLDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    # CLASSES = None
    CLASSES = ('Human',)

    def __init__(self,
                 ann_file,
                 img_prefix,
                 transforms,
                 square_bbox=False,
                 max_samples=-1,
                 mosh_path=None,
                 sample_weight=1,
                 **kwargs,
                 ):
        # prefix of images path
        self.img_prefix = img_prefix
        self._transforms = transforms

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file)

        # for i in range(len(self.img_infos)):
        #     img_info = self.img_infos[i]
        #     boxes = img_info['bboxes']
        #     keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        #
        #     if keep.sum() == 0:
        #         print(i)
        #         print(img_info['filename'])

        self.square_bbox = square_bbox
        self.max_samples = max_samples

        # Select a subset for quick validation
        if self.max_samples > 0:
            self.img_infos = random.sample(self.img_infos, max_samples)

        # Mosh dataset for generator
        self.mosh_path = mosh_path
        if self.mosh_path:
            mosh = np.load(mosh_path)
            self.mosh_shape = mosh['shape'].copy()
            self.mosh_pose = mosh['pose'].copy()
            self.mosh_sample_list = range(self.mosh_shape.shape[0])

        # sample weight
        self.sample_weight = sample_weight * np.ones(len(self.img_infos)) / (len(self.img_infos))

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        """
        filename:
        height: 1000
        width: commonly 1002 in h36m
        :param ann_file:
        :return:
        """
        with open(ann_file, 'rb') as f:
            raw_infos = pickle.load(f)
        # for quick eval
        if 'val_p2' in ann_file:
            print('quick eval')
            raw_infos = raw_infos[0:len(raw_infos):100]
        return raw_infos

    def add_essential_keys(self, img_info):
        float_list = ['bboxes', 'kpts3d', 'kpts2d', 'pose', 'shape', 'trans']
        int_list = ['labels', 'has_smpl']
        num_persons = img_info['bboxes'].shape[0] if 'bboxes' in img_info else 1
        # num_persons = img_info['bboxes'].shape[0] if 'bboxes' in img_info else 0

        for k in float_list:
            if k in img_info:
                img_info[k] = img_info[k].astype(np.float32).copy()
            else:
                img_info[k] = get_default(k, num_persons)
        for k in int_list:
            if k in img_info:
                img_info[k] = img_info[k].astype(np.int64).copy()
            else:
                img_info[k] = get_default(k, num_persons)

        # if num_persons == 0: # pose track has zero person image,causing a bug
        #     num_persons = 1
        #     for k in float_list:
        #         img_info[k] = get_default(k, num_persons)
        #     for k in int_list:
        #         img_info[k] = get_default(k, num_persons)

        return img_info

    # # NOT USED
    # def get_ann_info(self, idx):
    #     """
    #     :param idx:
    #     :return:A dict of the following iterms:
    #         bboxes: [x1, y1, x2, y2]
    #         labels: number
    #         kpts3d: (24, 4)
    #         kpts2d: (24, 3)
    #         pose: (72,)
    #         shape: (10,)
    #         cam: (3,) (The trans in SMPL model)
    #     """
    #     # Visualization needed
    #     raw_info = deepcopy(self.img_infos[idx])
    #     if 'bbox' in raw_info:      # H36M style
    #         img_info = {}
    #
    #         if 'bbox' in raw_info:
    #             bbox = raw_info['bbox']
    #             if self.square_bbox:
    #                 center = np.array([int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)])
    #                 bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    #                 half_size = int(math.floor(bbox_size / 2))
    #                 square_bbox = np.array(
    #                     [center[0] - half_size, center[1] - half_size, center[0] + half_size, center[1] + half_size])
    #             else:
    #                 square_bbox = np.array(bbox)
    #             #  Here comes a problem, what if the bbox overflows on the corner?
    #             # They will be args passed to `model.train_forward` if we overwrite `CustomDataset`.
    #             img_info.update({'bboxes': square_bbox.reshape(1, -1).astype(np.float32)})
    #
    #         if 'S' in raw_info:
    #             img_info.update({
    #                 'kpts3d': raw_info['S'][np.newaxis].astype(np.float32),  # (1, 24,4) extra chanel for visibility
    #             })
    #         if 'part' in raw_info:
    #             img_info.update({
    #                 'kpts2d': raw_info['part'][np.newaxis].astype(np.float32),  # (1, 24,3) extra chanel for visibility
    #             })
    #         if 'pose' in raw_info:
    #             img_info.update({
    #                 'pose': raw_info['pose'].reshape(-1, 3)[np.newaxis].astype(np.float32),  # (1, 24, 3)
    #                 'shape': raw_info['shape'][np.newaxis].astype(np.float32),  # (1,10)
    #             })
    #         if 'trans' in raw_info:
    #             img_info.update({
    #                 'trans': raw_info['trans'][np.newaxis].astype(np.float32),  # (1, 3)
    #             })
    #
    #     else:                       # COMMON STYLE
    #         img_info = raw_info
    #
    #     img_info = self.add_essential_keys(img_info)
    #     return img_info

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    @staticmethod
    def annToRLE(ann):
        h, w = ann['height'], ann['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_util.frPyObjects(segm, h, w)
            rle = mask_util.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = mask_util.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def get_image(self, path):
        return Image.open(os.path.join(self.img_prefix, path)).convert('RGB')

    def prepare(self, image, img_info):
        w, h = image.size
        boxes = torch.tensor(img_info['bboxes'], dtype=torch.float32).reshape(-1, 4) \
            if 'bboxes' in img_info else torch.zeros([1, 4], dtype=torch.float32)
        # boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        num_box = boxes.shape[0]

        labels = torch.tensor(img_info['labels'], dtype=torch.int64) \
            if 'labels' in img_info else torch.ones(num_box, dtype=torch.int64)
        kpts_2d = torch.tensor(img_info['kpts2d'], dtype=torch.float32)
        joints_2d = kpts_2d[..., :-1]
        joints_2d_visible = kpts_2d[..., -1]
        kpts_3d = torch.tensor(img_info['kpts3d'], dtype=torch.float32)
        joints_3d = kpts_3d[..., :-1]
        joints_3d_visible = kpts_3d[..., -1]
        iscrowd = torch.zeros(num_box, dtype=torch.int64)
        if 'bboxes_ignore' in img_info:
            iscrowd[img_info['bboxes_ignore']] = 1

        if 'pose' in img_info:
            pose = torch.tensor(img_info['pose'], dtype=torch.float32)
            pose = pose.reshape(num_box, 24, 3)
            shape = torch.tensor(img_info['shape'], dtype=torch.float32)
            has_smpl = torch.tensor(img_info['has_smpl'], dtype=torch.int64)
        else:
            pose = torch.zeros([num_box, 24, 3], dtype=torch.float32)
            shape = torch.zeros([num_box, 10], dtype=torch.float32)
            has_smpl = torch.zeros([num_box], dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area = area.abs()

        trans = torch.tensor(img_info['trans']) if 'trans' in img_info \
            else torch.zeros([num_box, 3], dtype=torch.float32)
        camera = torch.tensor(img_info['camera']) if 'camera' in img_info \
            else torch.zeros([num_box, 3], dtype=torch.float32)

        if 'segmentation' in img_info:
            assert 'COCO' in img_info['filename'], "Only support coco segmentation now"
            raw_mask = np.zeros((h, w), dtype=np.uint8)
            for i, seg in enumerate(img_info['segmentation']):
                ori_mask = mask_util.decode(
                    self.annToRLE({'width': img_info['width'], 'height': img_info['height'], 'segmentation': seg}))
                raw_mask[ori_mask > 0] = i + 1
            # scene = torch.from_numpy(raw_mask)
            scene = torch.from_numpy(raw_mask).long()
        else:
            # scene = torch.zeros([h, w], dtype=torch.uint8)
            scene = torch.zeros([h, w], dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]
        joints_2d, joints_2d_visible = joints_2d[keep], joints_2d_visible[keep]
        joints_3d, joints_3d_visible = joints_3d[keep], joints_3d_visible[keep]
        pose, shape, has_smpl = pose[keep], shape[keep], has_smpl[keep]
        trans, camera = trans[keep], camera[keep]

        size = torch.as_tensor([int(h), int(w)])
        target = dict(boxes=boxes, labels=labels, area=area,
                      iscrowd=iscrowd, orig_size=size, size=size,
                      joints_2d=joints_2d, joints_2d_visible=joints_2d_visible,
                      joints_3d=joints_3d, joints_3d_visible=joints_3d_visible,
                      smpl_pose=pose, smpl_shape=shape, has_smpl=has_smpl,
                      trans=trans, camera=camera,
                      scene=scene
                      )

        # Added for the EVAL on H36M
        imgname = img_info['filename']
        is_h36m_p2 = '.60457274_' in imgname
        is_h36m_p2 = torch.tensor(is_h36m_p2)
        target['is_h36m_p2'] = is_h36m_p2

        return image, target

    def __getitem__(self, idx):
        img_info = deepcopy(self.img_infos[idx])

        try:
            img = self.get_image(osp.join(self.img_prefix, img_info['filename']))
        except:
            print('Load image failed:{}, try another one.'.format(osp.join(self.img_prefix, img_info['filename'])))
            idx = np.random.randint(0, self.__len__())
            return self.__getitem__(idx)

        img_info = self.add_essential_keys(img_info)

        img, target = self.prepare(img, img_info)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def build_smpl_mix_dataset(image_set, args=None):
    data_root = '/mnt/lustre/zengwang/data'
    h36m_root = join(data_root, 'h36m')
    coco_data_root = join(data_root, 'coco')
    lsp_root = join(data_root, 'lsp_dataset_original')
    mpii_root = join(data_root, 'mpii')
    pose_track_root = join(data_root, 'posetrack2018/')
    mpi_inf_3dhp_root = join(data_root, '3dhp')

    train_transform = smpl_common_transforms('train')
    h36m = dict(
        ann_file=join(h36m_root, 'rcnn/train.pkl'),
        img_prefix=join(h36m_root, 'train_set'),
        transforms=train_transform,
        sample_weight=0.6,
    )
    coco = dict(
        ann_file=join(coco_data_root, 'coco-annots/train_densepose_2014_depth_nocrowd.pkl'),
        img_prefix=join(coco_data_root, 'train2014/'),
        transforms=train_transform,
        sample_weight=0.3,
    )
    lsp = dict(
        ann_file=join(lsp_root, 'train.pkl'),
        img_prefix=join(lsp_root, 'images/'),
        transforms=train_transform,
        sample_weight=0.3,
    )
    mpii = dict(
        ann_file=join(mpii_root, 'mpii-rcnn/train.pkl'),
        img_prefix=join(mpii_root, 'images/'),
        transforms=train_transform,
        sample_weight=0.3,
    )
    posetrack = dict(
        ann_file=join(pose_track_root, 'rcnn/train.pkl'),
        img_prefix=join(pose_track_root, 'images/'),
        transforms=train_transform,
        sample_weight=0.3,
    )
    hp3d = dict(
        ann_file=join(mpi_inf_3dhp_root, 'mpi_inf_rcnn/train.pkl'),
        img_prefix=mpi_inf_3dhp_root,
        transforms=train_transform,
        sample_weight=0.1,
    )

    if image_set == 'train':
        if args.stage == 'pretrain':
            train_cfgs = [
                h36m,
                coco,
                lsp,
                mpii,
                hp3d
            ]
        elif args.stage == 'baseline':
            train_cfgs = [
                h36m,
                coco,
                posetrack,
                mpii,
                hp3d
            ]
        else:
            train_cfgs = [
                h36m,
                coco,
                mpii,
                posetrack,
                hp3d
            ]

        datasets = []
        for dataset_cfg in train_cfgs:
            datasets.append(SMPLDataset(**dataset_cfg))
        return ConcatDataset(datasets)
    else:
        # FIXME: not finished yet.
        return None


def build_smpl_eval_dataset(dataset, args=None):
    data_root = '/mnt/lustre/zengwang/data'
    h36m_root = join(data_root, 'h36m')
    panoptic_root = join(data_root, 'panoptic')
    mupots_root = join(data_root, 'mupots-3d')
    eval_transform = smpl_common_transforms('eval')

    eval_dataset_mapper = dict(
        cropped_h36m=dict(
            ann_file=join(h36m_root, 'rcnn/val.pkl'),
            img_prefix=join(h36m_root, 'valid_set'),
            transforms=eval_transform,
        ),
        full_h36m=dict(
            ann_file=join(h36m_root, 'rcnn/val_p2.pkl'),
            img_prefix=join(h36m_root, 'valid_set'),
            transforms=eval_transform,
        ),
        ultimatum=dict(
            ann_file=join(panoptic_root, 'annotations/160422_ultimatum1.pkl'),
            img_prefix=panoptic_root,
            transforms=eval_transform,
        ),
        panoptic=dict(
            ann_file=join(panoptic_root, 'annotations/160422_ultimatum1.pkl'),
            img_prefix=panoptic_root,
            transforms=eval_transform,
        ),
        haggling=dict(
            ann_file=join(panoptic_root, 'annotations/160422_haggling1.pkl'),
            img_prefix=panoptic_root,
            transforms=eval_transform,
        ),
        pizza=dict(
            ann_file=join(panoptic_root, 'annotations/160906_pizza1.pkl'),
            img_prefix=panoptic_root,
            transforms=eval_transform,
        ),
        mafia=dict(
            ann_file=join(panoptic_root, 'annotations/160422_mafia2.pkl'),
            img_prefix=panoptic_root,
            transforms=eval_transform,
        ),
        mupots=dict(
            ann_file=join(mupots_root, 'rcnn/all_sorted.pkl'),
            img_prefix='',
            transforms=smpl_common_transforms('eval'),
        ),
    )

    dataset_cfg = eval_dataset_mapper[dataset]

    return SMPLDataset(**dataset_cfg)

