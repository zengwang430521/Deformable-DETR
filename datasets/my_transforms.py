# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
import numpy as np


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    # merged 24 keypoints
    if "joints_2d" in target:
        joints_2d = target["joints_2d"]
        joints_2d_visible = target["joints_2d_visible"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_joints_2d = joints_2d - torch.as_tensor([j, i])[None, :]
        cropped_joints_2d_visible = (joints_2d_visible *
                                     (cropped_joints_2d < max_size).min(dim=-1)[0])
        target["joints_2d"] = cropped_joints_2d
        target["joints_2d_visible"] = cropped_joints_2d_visible

    if "scene" in target:
        target["scene"] = target['scene'][i:i + h, j:j + w]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    if "joints_2d" in target:
        joints_2d = target["joints_2d"]
        joints_2d_visible = target["joints_2d_visible"]
        joints_2d, joints_2d_visible = flip_kp_2d(joints_2d, joints_2d_visible, img_width=w)
        target["joints_2d"] = joints_2d
        target["joints_2d_visible"] = joints_2d_visible

    if "joints_3d" in target:
        joints_3d = target["joints_3d"]
        joints_3d_visible = target["joints_3d_visible"]
        joints_3d, joints_3d_visible = flip_kp_3d(joints_3d, joints_3d_visible)
        target["joints_3d"] = joints_3d
        target["joints_3d_visible"] = joints_3d_visible

    if "smpl_pose" in target:
        smpl_pose = target["smpl_pose"]
        smpl_pose = flip_smpl_pose(smpl_pose)
        target["smpl_pose"] = smpl_pose

    if "scene" in target:
        target["scene"] = target['scene'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    if "joints_2d" in target:
        joints_2d = target["joints_2d"]
        scaled_joints_2d = joints_2d * torch.as_tensor([ratio_width, ratio_height])
        target["joints_2d"] = scaled_joints_2d

    if "scene" in target:
        scene = target["scene"]
        rescaled_scene = interpolate(scene[None, None, ...].float(), size, mode="nearest")
        rescaled_scene = rescaled_scene.type(torch.uint8)[0, 0]
        target["scene"] = rescaled_scene

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))

    if "scene" in target:
        target["scene"] = torch.nn.functional.pad(target['scene'], (0, padding[0], 0, padding[1]))

    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


#############   Added


class TranKeypoints(object):
    def __init__(self, type='coco'):
        self.type = type

    def __call__(self, img, target):
        target = target.copy()

        if 'keypoints' in target:
            keypoints_17 = target["keypoints"]
            keypoints_24 = coco17_to_superset(keypoints_17)
            joints_2d = keypoints_24[..., :2]
            joints_2d_visible = keypoints_24[..., -1]
            target["joints_2d"] = joints_2d
            target["joints_2d_visible"] = joints_2d_visible

        return img, target


class AddSMPLKeys(object):
    def __call__(self, img, target):
        boxes = target["boxes"]
        num_bbox = boxes.shape[0]

        if "joints_2d" not in target:
            target["joints_2d"] = boxes.new_zeros(num_bbox, 24, 2)
            target["joints_2d_visible"] = boxes.new_zeros(num_bbox, 24)

        if "joints_3d" not in target:
            target["joints_3d"] = boxes.new_zeros(num_bbox, 24, 3)
            target["joints_3d_visible"] = boxes.new_zeros(num_bbox, 24)

        if "smpl_pose" not in target:
            target["smpl_pose"] = boxes.new_zeros(num_bbox, 72)
            target["smpl_shape"] = boxes.new_zeros(num_bbox, 10)
            target["has_smpl"] = boxes.new_zeros(num_bbox).long()

        if "camera" not in target:
            target["camera"] = boxes.new_zeros(num_bbox, 3)

        if "trans" not in target:
            target["trans"] = boxes.new_zeros(num_bbox, 3)

        if "scene" not in target:
            _, h, w = img.shape
            target["scene"] = torch.zeros([h, w], dtype=torch.uint8)

        return img, target


def flip_kp_2d(kp, kp_visible, img_width):
    """
    Flip augmentation for keypoints
    :param kp:
    :param img_width: (int)
    :return:
    """
    flipped_parts = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
    flipped_kp = kp[:, flipped_parts]
    flipped_kp[:, :, 0] = img_width - flipped_kp[:, :, 0]
    flipped_kp_visible = kp_visible[:, flipped_parts]
    return flipped_kp, flipped_kp_visible


def flip_kp_3d(kp, kp_visible):
    """
    Flip augmentation for keypoints
    :param kp:
    :param img_width: (int)
    :return:
    """
    flipped_parts = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
    flipped_kp = kp[:, flipped_parts]
    flipped_kp[:, :, 0] = 0 - flipped_kp[:, :, 0]
    flipped_kp_visible = kp_visible[:, flipped_parts]
    return flipped_kp, flipped_kp_visible


def flip_smpl_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flippedParts = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13,
                    14, 18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33,
                    34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41,
                    45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55,
                    56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
    num = pose.shape[0]
    pose = pose.reshape(num, 72)
    pose = pose[:, flippedParts]
    # we also negate the second and the third dimension of the axis-angle
    pose[:, 1::3] = -pose[:, 1::3]
    pose[:, 2::3] = -pose[:, 2::3]
    pose = pose.reshape(num, 24, 3)
    return pose


def coco17_to_superset(coco_kpts):
    """
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                    'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    0 - Right Ankle
    1 - Right Knee
    2 - Right Hip
    3 - Left Hip
    4 - Left Knee
    5 - Left Ankle
    6 - Right Wrist
    7 - Right Elbow
    8 - Right Shoulder
    9 - Left Shoulder
    10 - Left Elbow
    11 - Left Wrist
    12 - Neck (LSP definition)
    13 - Top of Head (LSP definition)
    14 - Pelvis (MPII definition)
    15 - Thorax (MPII definition)
    16 - Spine (Human3.6M definition)
    17 - Jaw (Human3.6M definition)
    18 - Head (Human3.6M definition)
    19 - Nose
    20 - Left Eye
    21 - Right Eye
    22 - Left Ear
    23 - Right Ear
    :param gt_keypoints: ...x17xM Keypoints tensor or array
    :return super_kpts
    """

    creator_fn = None
    coco_in_superset = [19, 20, 21, 22, 23, 9,  # 5
                        8, 10, 7, 11, 6,  # 10
                        3, 2, 4, 1, 5, 0  # 15
                        ]
    if isinstance(coco_kpts, torch.Tensor):
        creator_fn = torch.zeros
    elif isinstance(coco_kpts, np.ndarray):
        creator_fn = np.zeros
    super_kpts = creator_fn((coco_kpts.shape[:-2]) + (24,) + (coco_kpts.shape[-1],))
    super_kpts[..., coco_in_superset, :] = coco_kpts
    return super_kpts


def coco17to19(coco17pose):
    """
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    coco19_kp_names = ['neck', 'nose', 'hip', 'l_shoulder', 'l_elbow', 'l_wrist',  # 5
                'l_hip', 'l_knee', 'l_ankle', 'r_shoulder', 'r_elbow',  # 10
                'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye', # 15
                'l_ear', 'r_eye', 'r_ear']
    :param coco17pose: 17x3 coco pose np.array
    :return: 19x3 coco19 pose np.array
    """
    coco19pose = np.zeros((19, coco17pose.shape[1]))
    index_array = np.array([1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14])
    coco19pose[index_array] = coco17pose
    coco19pose[0] = (coco17pose[5] + coco17pose[6]) / 2
    coco19pose[2] = (coco17pose[11] + coco17pose[12]) / 2
    coco19pose[-4:] = coco17pose[0]  # Since we have not implement eye and ear yet.
    return coco19pose


def coco19_to_superset(coco19pose):
    """
    coco19_kp_names = ['neck', 'nose', 'hip', 'l_shoulder', 'l_elbow', 'l_wrist',  # 5
                'l_hip', 'l_knee', 'l_ankle', 'r_shoulder', 'r_elbow',  # 10
                'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye', # 15
                'l_ear', 'r_eye', 'r_ear']
    kpts_coco19 = [12, 19, 14, 9, 10, 11,
                    3, 4, 5, 8, 7, #10
                     6, 2, 1, 0, 20, #15
                     22, 21, 23]
    0 - Right Ankle
    1 - Right Knee
    2 - Right Hip
    3 - Left Hip
    4 - Left Knee
    5 - Left Ankle
    6 - Right Wrist
    7 - Right Elbow
    8 - Right Shoulder
    9 - Left Shoulder
    10 - Left Elbow
    11 - Left Wrist
    12 - Neck (LSP definition)
    13 - Top of Head (LSP definition)
    14 - Pelvis (MPII definition)
    15 - Thorax (MPII definition)
    16 - Spine (Human3.6M definition)
    17 - Jaw (Human3.6M definition)
    18 - Head (Human3.6M definition)
    19 - Nose
    20 - Left Eye
    21 - Right Eye
    22 - Left Ear
    23 - Right Ear
    :param coco19pose:
    :return:
    """
    pass
    # superset_names =
    J24_names = ['Right Ankle',
                 'Right Knee',
                 'Right Hip',
                 'Left Hip',
                 'Left Knee',
                 'Left Ankle',
                 'Right Wrist',
                 'Right Elbow',
                 'Right Shoulder',
                 'Left Shoulder',
                 'Left Elbow',
                 'Left Wrist',
                 'Neck (LSP definition)',
                 'Top of Head (LSP definition)',
                 'Pelvis (MPII definition)',
                 'Thorax (MPII definition)',
                 'Spine (Human3.6M definition)',
                 'Jaw (Human3.6M definition)',
                 'Head (Human3.6M definition)',
                 'Nose',
                 'Left Eye',
                 'Right Eye',
                 'Left Ear',
                 'Right Ear']
    coco19_kp_names = ['neck', 'nose', 'hip', 'l_shoulder', 'l_elbow', 'l_wrist',  # 5
                       'l_hip', 'l_knee', 'l_ankle', 'r_shoulder', 'r_elbow',  # 10
                       'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye',  # 15
                       'l_ear', 'r_eye', 'r_ear']

    h36m_names = ['Pelvis (MPII definition)',
                  'Left Hip',
                  'Left Knee',
                  'Left Ankle',
                  'Right Hip',
                  'Right Knee',
                  'Right Ankle',
                  'Spine (Human3.6M definition)',  # To interpolate
                  'Neck (LSP definition)',
                  'Jaw (Human3.6M definition)',  # To interpolate
                  'Head (Human3.6M definition)',  # To interpolate
                  'Left Shoulder',
                  'Left Elbow',
                  'Left Wrist',
                  'Right Shoulder',
                  'Right Elbow',
                  'Right Wrist']
    """
    0: Pelvis (MPII definition)
    1: Left Hip
    2: Left Knee
    3: Left Ankle
    4: Right Hip
    5: Right Knee
    6: Right Ankle
    7: Spine (Human3.6M definition)
    8: Neck (LSP definition)
    9: Jaw (Human3.6M definition)
    10: Head (Human3.6M definition)
    11: Left Shoulder
    12: Left Elbow
    13: Left Wrist
    14: Right Shoulder
    15: Right Elbow
    16: Right Wrist
    """

    superset_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
    kpts_coco19 = [12, 19, 14, 9, 10, 11,
                   3, 4, 5, 8, 7,  # 10
                   6, 2, 1, 0, 20,  # 15
                   22, 21, 23]


def PanopticJ15_to_Superset():
    """
    0 - Right Ankle
    1 - Right Knee
    2 - Right Hip
    3 - Left Hip
    4 - Left Knee
    5 - Left Ankle
    6 - Right Wrist
    7 - Right Elbow
    8 - Right Shoulder
    9 - Left Shoulder
    10 - Left Elbow
    11 - Left Wrist
    12 - Neck (LSP definition)
    13 - Top of Head (LSP definition)
    14 - Pelvis (MPII definition)
    15 - Thorax (MPII definition)
    16 - Spine (Human3.6M definition)
    17 - Jaw (Human3.6M definition)
    18 - Head (Human3.6M definition)
    19 - Nose
    20 - Left Eye
    21 - Right Eye
    22 - Left Ear
    23 - Right Ear
    BoneJointOrder = { [2 1 3] ...   %{headtop, neck, bodyCenter}
                    , [1 4 5 6] ... %{neck, leftShoulder, leftArm, leftWrist}
                    , [3 7 8 9] ...  %{neck, leftHip, leftKnee, leftAnkle}
                    , [1 10 11 12]  ... %{neck, rightShoulder, rightArm, rightWrist}
                    , [3 13 14 15]};    %{neck, rightHip, rightKnee, rightAnkle}
    :return:
    """
    pass
    Panoptic_to_J15 = [12, 13, 14, 9, 10, 11,  # 5s
                       3, 4, 5, 8, 7, 6,  # 11
                       2, 1, 0
                       ]
