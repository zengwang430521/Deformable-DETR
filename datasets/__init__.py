# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .coco import smpl_build as smpl_build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    # FIXME : NOT FINSHED VAL SET YET
    if args.smpl:
        if image_set == 'train':
            from .my_common import build_smpl_mix_dataset
            return build_smpl_mix_dataset(image_set, args)
        else:
            #TODO: make wval dataset compatiable with training process
            if hasattr(args, 'eval_dataset'):
                from .my_common import build_smpl_eval_dataset
                return build_smpl_eval_dataset(args.eval_dataset, args)
            # else:
            #     if args.dataset_file == 'coco':
            #         return build_coco(image_set, args)
            #     if args.dataset_file == 'coco_panoptic':
            #         # to avoid making panopticapi required for coco
            #         from .coco_panoptic import build as build_coco_panoptic
            #         return build_coco_panoptic(image_set, args)

    else:
        if args.dataset_file == 'coco':
            return build_coco(image_set, args)
        if args.dataset_file == 'coco_panoptic':
            # to avoid making panopticapi required for coco
            from .coco_panoptic import build as build_coco_panoptic
            return build_coco_panoptic(image_set, args)
        raise ValueError(f'dataset {args.dataset_file} not supported')
