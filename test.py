from datasets.my_common import SMPLDataset, smpl_common_transforms

WITH_NR = True
FOCAL_LENGTH = 1000

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
square_bbox = False
common_train_cfg = dict(
    img_scale=(832, 512),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0.5,
    # noise_factor=1e-3,  # To avoid color jitter.
    with_mask=False,
    with_crowd=False,
    with_label=True,
    with_kpts2d=True,
    with_kpts3d=True,
    with_pose=True,
    with_shape=True,
    with_trans=True,
    # max_samples=1024
    square_bbox=square_bbox,
    mosh_path='data/CMU_mosh.npz',
    with_nr=WITH_NR,
    use_poly=True,
    # rot_factor=30,
)

# h36m_dataset_type = 'H36MDataset'
# h36m_data_root = 'data/h36m/'
# coco_dataset_type = 'COCOKeypoints'
# coco_data_root = 'data/coco/'
# common_dataset = 'CommonDataset'
# pose_track_root = 'data/posetrack2018/'
# mpi_inf_3dhp_root = 'data/mpi_inf_3dhp/'
# panoptic_root = 'data/Panoptic/'


# mpii_root = '/home/wzeng/mydata/mpii/'
# dataset_cfg = dict(
#         ann_file=mpii_root + 'rcnn/train.pkl',
#         img_prefix=mpii_root + 'images/',
#         transforms=smpl_common_transforms('train'),
#         **common_train_cfg
# )

coco_data_root = '/home/wzeng/mydata/coco/'
dataset_cfg = dict(
    # ann_file=coco_data_root + 'annotations/train_densepose_2014_scene.pkl',
    ann_file=coco_data_root + 'annotations/train_densepose_2014_depth_nocrowd.pkl',
    img_prefix=coco_data_root + 'train2014/',
    # sample_weight=0.3,
    transforms=smpl_common_transforms('train'),
    **common_train_cfg
)

test_dataset = SMPLDataset(**dataset_cfg)

for i in range(100):
    test_item = test_dataset[i]

t = 0