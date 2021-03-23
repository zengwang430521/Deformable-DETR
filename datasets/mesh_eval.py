import torch
import torch
import torch.nn as nn

# from ..registry import LOSSES
# from ..utils.pose_utils import reconstruction_error
from models.smpl.smpl import SMPL, JointMapper
import random
from sdf import SDFLoss
import neural_renderer as nr
import numpy as np
from models.camera import PerspectiveCamera

import os.path as osp
import torch
import torchvision
# from smplx.body_models import SMPL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import abc
import math
import scipy.io as scio
from models.smpl_utils import batch_rodrigues, J24_TO_J14, H36M_TO_J14, J24_TO_H36M
from models.smpl.viz import draw_skeleton, J24_TO_J14, get_bv_verts, plot_pose_H36M
import os
import pickle
from tqdm import tqdm
from models.pose_utils import reconstruction_error

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EvalHandler(object):
    def __init__(self, writer=print, log_every=50, viz_dir='', FOCAL_LENGTH=1000, work_dir=''):
        self.call_cnt = 0
        self.log_every = log_every
        self.writer = writer
        self.viz_dir = viz_dir
        self.work_dir = work_dir
        self.camera = PerspectiveCamera(FOCAL_LENGTH=FOCAL_LENGTH)
        self.FOCAL_LENGTH = FOCAL_LENGTH
        if self.viz_dir:
            from models.smpl.renderer import Renderer
            self.renderer = Renderer(focal_length=FOCAL_LENGTH)
            pass
        else:
            self.renderer = None

    def __call__(self, *args, **kwargs):
        self.call_cnt += 1
        res = self.handle(*args, **kwargs)
        if self.log_every > 0 and (self.call_cnt % self.log_every == 0):
            self.log()
        return res

    def handle(self, *args, **kwargs):
        pass

    def log(self):
        pass

    def finalize(self):
        pass

    def to(self, device):
        self.smpl = self.smpl.to(device)
        self.J_regressor = self.J_regressor.to(device)
        return self

    def get_outputs(self, data_batch, pred_results):
        FOCAL_LENGTH = self.FOCAL_LENGTH

        all_logit = pred_results["pred_logits"]
        prob = all_logit.sigmoid()
        valid = prob[..., 1] > prob[..., 0]

        device = all_logit.device
        samples, targets = data_batch
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # img_size is the shape of the inputed image
        img_size = targets[0]['size'].unsqueeze(0)
        img_size = img_size.flip(-1)  # img_size should be [w, h]

        orig_size = targets[0]['orig_size'].unsqueeze(0)
        orig_size = orig_size.flip(-1)

        all_bboxes = pred_results['pred_boxes']
        all_bboxes = all_bboxes * torch.cat([img_size, img_size], dim=-1)
        valid_bbox = (all_bboxes[..., 2] > 5) * (all_bboxes[..., 3] > 5)
        valid = valid * valid_bbox

        pred_bboxes = all_bboxes[valid]
        batch_size = pred_bboxes.shape[0]

        pred_bboxes[..., 2:] += pred_bboxes[..., :2]
        center_pts = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
        pred_camera = pred_results['pred_camera'][valid]

        # get translation
        crop_translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(pred_camera.device)
        crop_translation[..., :2] = pred_camera[..., 1:]
        # We may detach it.
        bboxes_size = torch.max(torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]),
                                torch.abs(pred_bboxes[..., 1] - pred_bboxes[..., 3]))
        crop_translation[..., 2] = 2 * FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
        depth = 2 * FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
        translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(
            pred_camera.device)
        translation[:, :-1] = depth[:, None] * \
                              (center_pts + pred_camera[:, 1:] *
                               bboxes_size.unsqueeze(-1) - img_size / 2) / FOCAL_LENGTH
        translation[:, -1] = depth
        pred_translation = translation

        # get smpl parameters
        pred_betas = pred_results['pred_smpl_shape'][valid]
        pred_rotmat = pred_results['pred_smpl_pose'][valid]

        # top-K
        K = 10
        num_bbox = pred_bboxes.shape[0]
        if num_bbox > K:
            pred_logit = pred_results["pred_logits"][valid]
            prob = pred_logit.sigmoid()
            topk_values, topk_indexes = torch.topk(prob[:, 1], K)

            pred_logit = pred_logit[topk_indexes]
            pred_betas = pred_betas[topk_indexes]
            pred_rotmat = pred_rotmat[topk_indexes]
            pred_bboxes = pred_bboxes[topk_indexes]
            pred_translation = pred_translation[topk_indexes]
            pred_camera = pred_camera[topk_indexes]

        smpl_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                                pose2rot=False)
        pred_vertices = smpl_output.vertices.clone()
        pred_joints = smpl_output.joints
        return samples, targets, \
               pred_bboxes, pred_rotmat, pred_betas, pred_camera, \
               pred_translation, pred_vertices, pred_joints


class MuPoTSEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        # self.p1_meter = AverageMeter('P1', ':.2f')
        # self.p2_meter = AverageMeter('P2', ':.2f')
        # self.p3_meter = AverageMeter('P3', ':.2f')
        self.stats = list()
        self.mismatch_cnt = 0
        # Initialize SMPL model
        openpose_joints = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                           7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        extra_joints = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27]
        joints = torch.tensor(openpose_joints + extra_joints, dtype=torch.int32)
        joint_mapper = JointMapper(joints)
        # smpl_params = dict(model_path='data/smpl',
        #                    # model_folder='data/smpl',
        #                    joint_mapper=joint_mapper,
        #                    create_glb_pose=True,
        #                    body_pose_param='identity',
        #                    create_body_pose=True,
        #                    create_betas=True,
        #                    # create_trans=True,
        #                    dtype=torch.float32,
        #                    vposer_ckpt=None,
        #                    gender='neutral')
        # self.smpl = SMPL(**smpl_params)

        self.smpl = SMPL('data/smpl')

        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.result_list = list()
        self.result_list_2d = list()
        self.h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]

        self.FOCAL_LENGTH = 1000
        # self.collision_meter = AverageMeter('collision', ':.2f')
        # self.collision_volume = CollisionVolume(self.smpl.faces, grid_size=64).cuda()
        # self.coll_cnt = 0

    def handle(self, data_batch, pred_results):
        # # Evaluate collision metric
        # pred_vertices = pred_results['pred_vertices']
        # pred_translation = pred_results['pred_translation']
        # # cur_collision_volume = self.collision_volume(pred_vertices, pred_translation)
        # # if cur_collision_volume.item() > 0:
        # #     # self.writer(f'Collision found with {cur_collision_volume.item() * 100 } L')
        # #     self.coll_cnt += 1
        # # self.collision_meter.update(cur_collision_volume.item() * 1000.)

        samples, targets, \
        pred_bboxes, pred_rotmat, pred_betas, \
        pred_camera, pred_translation, \
        pred_vertices, pred_joints = self.get_outputs(data_batch, pred_results)

        # img_size is the shape of the inputed image
        img_size = targets[0]['size'].unsqueeze(0)
        img_size = img_size.flip(-1)  # img_size should be [w, h]
        orig_size = targets[0]['orig_size'].unsqueeze(0)
        orig_size = orig_size.flip(-1)

        batch_size = pred_bboxes.shape[0]

        # pred_vertices = pred_results['pred_vertices'].cpu()
        # pred_camera = pred_results['pred_camera'].cpu()
        # pred_translation = pred_results['pred_translation'].cpu()
        # bboxes = pred_results['bboxes'][0][:, :4]
        # img = data_batch['img'].data[0][0].clone()

        J_regressor_batch = self.J_regressor[None, :].expand(batch_size, -1, -1).to(pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        self.result_list.append(
            (pred_keypoints_3d_smpl[:, self.h36m_to_MPI] + pred_translation[:, None]).cpu().numpy())

        batch_size = pred_keypoints_3d_smpl.shape[0]
        rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_keypoints_3d_smpl.device)
        pred_keypoints_2d_smpl = self.camera(pred_keypoints_3d_smpl, batch_size=batch_size, rotation=rotation_Is,
                                             translation=pred_translation,
                                             center=img_size / 2)
        if self.viz_dir:
            # img = img.clone() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
            #     [0.485, 0.456, 0.406]).view(3, 1, 1)
            img = samples.tensors[0]
            img = img.clone() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) \
                  + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img_cv = img.clone().cpu().numpy()
            img_cv = (img_cv * 255).astype(np.uint8).transpose([1, 2, 0]).copy()
            for kpts, bbox in zip(pred_keypoints_2d_smpl.cpu().numpy(), pred_bboxes.cpu().numpy()):
                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                img_cv = draw_skeleton(img_cv, kpts[H36M_TO_J14, :2])
            # img_cv = draw_text(img_cv, {'mismatch': is_mismatch, 'error': str(error_smpl.mean(-1) * 1000)});
            img_cv = (img_cv / 255.)
            # fname = osp.basename(data_batch['img_meta'].data[0][0]['file_name'])
            fname = '{}.jpg'.format(len(self.result_list_2d))
            plt.imsave(osp.join(self.viz_dir, fname), img_cv)

        # scale_factor is the scale factor between input image and origin image in dataset
        # scale_factor = data_batch['img_meta'].data[0][0]['scale_factor']
        scale_factor = img_size.float() / orig_size.float()
        raw_kpts2d = pred_keypoints_2d_smpl / scale_factor

        self.result_list_2d.append(raw_kpts2d[:, self.h36m_to_MPI].cpu().numpy())
        # return {'file_name': data_batch['img_meta'].data[0][0]['file_name'], 'pred_kpts3d': pred_keypoints_3d_smpl}
        return {'pred_kpts3d': pred_keypoints_3d_smpl}

    def log(self):
        # self.writer(f'coll_cnt: {self.coll_cnt} coll {self.collision_meter.avg} L')
        pass

    def finalize(self):
        max_persons = max([i.shape[0] for i in self.result_list])
        result = np.zeros((len(self.result_list), max_persons, 17, 3))
        result_2d = np.zeros((len(self.result_list), max_persons, 17, 2))
        for i, (r, r_2d) in enumerate(zip(self.result_list, self.result_list_2d)):
            result[i, :r.shape[0]] = r
            result_2d[i, :r.shape[0]] = r_2d
        scio.savemat(osp.join(self.work_dir, 'mupots.mat'), {'result': result, 'result_2d': result_2d})


class H36MEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.2f')
        self.p2_meter = AverageMeter('P2', ':.2f')
        self.FOCAL_LENGTH = 1000
        self.smpl = SMPL('data/smpl')

    def handle(self, data_batch, pred_results, use_gt=False):
        samples, targets, \
        pred_bboxes, pred_rotmat, pred_betas, \
        pred_camera, pred_translation, \
        pred_vertices, pred_joints = self.get_outputs(data_batch, pred_results)

        gt_keypoints_3d = targets[0]['joints_3d'].clone().repeat([pred_vertices.shape[0], 1, 1])
        # gt_keypoints_3d = data_batch['gt_kpts3d'].data[0][0].clone().repeat([pred_vertices.shape[0], 1, 1])

        gt_pelvis_smpl = gt_keypoints_3d[:, [14], :].clone()
        gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_J14, :].clone()
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

        # file_name = data_batch['img_meta'].data[0][0]['file_name']

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1)

        mpjpe = float(error_smpl.min() * 1000)
        self.p1_meter.update(mpjpe)

        # if self.pattern in file_name:
        is_p2 = targets[0]['is_h36m_p2'].cpu().item()
        if is_p2:
            # Reconstruction error
            r_error_smpl = reconstruction_error(pred_keypoints_3d_smpl.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                                reduction=None)
            r_error = float(r_error_smpl.min() * 1000)
            self.p2_meter.update(r_error)
        else:
            r_error = -1

        save_pack = {  # 'file_name': file_name,
            'MPJPE': mpjpe,
            'r_error': r_error,
            'pred_rotmat': pred_rotmat,
            'pred_betas': pred_betas,
        }
        return save_pack

    def log(self):
        self.writer(f'p1: {self.p1_meter.avg:.2f}mm, p2: {self.p2_meter.avg:.2f}mm')


class PanopticEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.p1_meter = AverageMeter('P1', ':.2f')
        self.stats = list()
        self.mismatch_cnt = 0
        # Initialize SMPL model
        openpose_joints = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                           7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        extra_joints = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27]
        joints = torch.tensor(openpose_joints + extra_joints, dtype=torch.int32)
        joint_mapper = JointMapper(joints)
        self.smpl = SMPL('data/smpl')

        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.collision_meter = AverageMeter('P3', ':.2f')
        # self.collision_volume = CollisionVolume(self.smpl.faces, grid_size=64).cuda()
        self.coll_cnt = 0
        self.threshold_list = [0.1, 0.15, 0.2]
        self.total_ordinal_cnt = {i: 0 for i in self.threshold_list}
        self.correct_ordinal_cnt = {i: 0 for i in self.threshold_list}

    def handle(self, data_batch, pred_results, use_gt=False):
        # # Evaluate collision metric
        # pred_vertices = pred_results['pred_vertices']
        # pred_translation = pred_results['pred_translation']
        # cur_collision_volume = self.collision_volume(pred_vertices, pred_translation)
        # if cur_collision_volume.item() > 0:
        #     # self.writer(f'Collision found with {cur_collision_volume.item() * 1000} L')
        #     self.coll_cnt += 1
        # self.collision_meter.update(cur_collision_volume.item() * 1000.)

        samples, targets, \
        pred_bboxes, pred_rotmat, pred_betas, \
        pred_camera, pred_translation, \
        pred_vertices, pred_joints = self.get_outputs(data_batch, pred_results)

        # gt_keypoints_3d = data_batch['gt_kpts3d'].data[0][0].clone()
        gt_keypoints_3d = targets[0]['joints_3d'].clone()
        gt_pelvis_smpl = gt_keypoints_3d[:, [14], :].clone()
        visible_kpts = targets[0]['joints_3d_visible'][:, J24_TO_H36M].clone()

        # origin_gt_kpts3d = data_batch['gt_kpts3d'].data[0][0].clone().cpu()
        origin_gt_kpts3d = targets[0]['joints_3d'].clone().cpu()

        origin_gt_kpts3d = origin_gt_kpts3d[:, J24_TO_H36M]
        # origin_gt_kpts3d[:, :, :-1] -= gt_pelvis_smpl
        gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_H36M, :].clone()
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        # pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

        # file_name = data_batch['img_meta'].data[0][0]['file_name']
        # fname = osp.basename(file_name)
        fname = '{}.jpg'.format(self.p1_meter.count)

        # To select closest points
        glb_vis = (visible_kpts.sum(0) >= (visible_kpts.shape[0] - 0.1)).float()[None, :,
                  None]  # To avoid in-accuracy in float point number
        if use_gt:
            paired_idxs = torch.arange(gt_keypoints_3d.shape[0])
        else:
            # def vectorize_distance(x, y):
            #     dist = x[:, np.newaxis, :, :] - y[np.newaxis, :, :, :]
            #     dist = np.linalg.norm(dist, ord=2, axis=-1)
            #     dist = dist.mean(axis=-1)
            #     return dist
            # dist = vectorize_distance((glb_vis * gt_keypoints_3d).numpy(),
            #                           (glb_vis * pred_keypoints_3d_smpl).numpy())
            # paired_idxs = torch.from_numpy(dist.argmin(1))

            def vectorize_distance(x, y):
                dist = x.unsqueeze(1) - y.unsqueeze(0)
                dist = dist.norm(p=2, dim=-1).mean(dim=-1)
                # dist = dist.mean(dim=-2).norm(p=2, dim=-1)
                return dist

            dist = vectorize_distance((glb_vis * gt_keypoints_3d),
                                      (glb_vis * pred_keypoints_3d_smpl))
            paired_idxs = dist.argmin(1)

        is_mismatch = len(set(paired_idxs.tolist())) < len(paired_idxs)
        if is_mismatch:
            self.mismatch_cnt += 1

        selected_prediction = pred_keypoints_3d_smpl[paired_idxs]

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = (torch.sqrt(((selected_prediction - gt_keypoints_3d) ** 2).sum(dim=-1)) * visible_kpts)

        mpjpe = float(error_smpl.mean() * 1000)
        self.p1_meter.update(mpjpe, n=error_smpl.shape[0])

        save_pack = {  # 'file_name': osp.basename(file_name),
            'MPJPE': mpjpe,
            'pred_rotmat': pred_rotmat.cpu(),
            'pred_betas': pred_betas.cpu(),
            'gt_kpts': origin_gt_kpts3d,
            'kpts_paired': selected_prediction,
            'pred_kpts': pred_keypoints_3d_smpl,
        }

        if self.viz_dir and (is_mismatch or error_smpl.mean(-1).min() * 1000 > 200):
            img = samples.tensors[0]
            img = img.clone() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) \
                  + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

            img_cv = img.clone().numpy()
            img_cv = (img_cv * 255).astype(np.uint8).transpose([1, 2, 0]).copy()

            bboxes = pred_bboxes.cpu().numpy()
            paired_idxs = paired_idxs.cpu().numpy()
            for bbox in bboxes[paired_idxs]:
                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            img_cv = draw_text(img_cv, {'mismatch': is_mismatch, 'error': str(error_smpl.mean(-1) * 1000)});
            img_cv = (img_cv / 255.)

            torch.set_printoptions(precision=1)
            img_render = self.renderer([torch.tensor(img_cv.transpose([2, 0, 1]))], [pred_vertices.cpu()],
                                       translation=[pred_translation])

            bv_verts = get_bv_verts(pred_bboxes, pred_vertices, pred_translation,
                                    img.shape, self.FOCAL_LENGTH)
            img_bv = self.renderer([torch.ones_like(img)], [bv_verts],
                                   translation=[torch.zeros(bv_verts.shape[0], 3)])
            img_grid = torchvision.utils.make_grid(torch.tensor(([img_render[0], img_bv[0]])),
                                                   nrow=2).numpy().transpose([1, 2, 0])
            img_grid[img_grid > 1] = 1
            img_grid[img_grid < 0] = 0
            if not os.path.exists(self.viz_dir):
                os.makedirs(self.viz_dir)
            plt.imsave(osp.join(self.viz_dir, fname), img_grid)

        return save_pack

    def log(self):
        self.writer(
            f'p1: {self.p1_meter.avg:.2f}mm, coll_cnt: {self.coll_cnt} coll: {self.collision_meter.avg} L')

    def finalize(self):
        print('Finish:')
        print(f'p1: {self.p1_meter.avg:.2f}mm, coll_cnt: {self.coll_cnt} coll: {self.collision_meter.avg} L')


def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = np.array([0, 0, 255])
    margin = 45
    start_x = 15
    start_y = margin
    for key in sorted(content.keys()):
        text = f"{key}: {content[key]}"
        image = cv2.putText(image, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image


def evaluate_smpl(model, data_loader, args, device):
    model.eval()

    # dump_dir = os.path.join(cfg.work_dir, f'eval_{args.dataset}')
    # os.makedirs(dump_dir, exist_ok=True)
    # if args.viz_dir:
    #     os.makedirs(args.viz_dir, exist_ok=True)

    dump_dir = f'eval_results/eval_{args.eval_dataset}'
    os.makedirs(dump_dir, exist_ok=True)
    viz_dir=''
    # viz_dir = f'eval_results/vis_{args.eval_dataset}'
    # if viz_dir:
    #     os.makedirs(viz_dir, exist_ok=True)

    FOCAL_LENGTH = 1000
    eval_handler_mapper = dict(
        mupots=MuPoTSEvalHandler,
        full_h36m=H36MEvalHandler,
        ultimatum=PanopticEvalHandler,
        haggling=PanopticEvalHandler,
        pizza=PanopticEvalHandler,
        mafia=PanopticEvalHandler,
    )

    eval_handler = eval_handler_mapper[args.eval_dataset](
        writer=tqdm.write, viz_dir=viz_dir,
        FOCAL_LENGTH=FOCAL_LENGTH,
        work_dir='eval_results')
    eval_handler.to(device)

    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(data_loader)):
            samples, targets = data_batch
            samples = samples.to(device)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            pred_results = model(samples)
            save_pack = eval_handler(data_batch, pred_results)
            # save_pack.update({'bbox_results': pred_results['bboxes']})
            # if args.dump_pkl:
            #     with open(osp.join(dump_dir, f"{save_pack['file_name']}.pkl"), 'wb') as f:
            #         pickle.dump(save_pack, f)

            ##TODO: Add mesh saving codes
            # if args.paper_dir:
            #     os.makedirs(args.paper_dir, exist_ok=True)
            #     img = denormalize(data_batch['img'].data[0][0].numpy())
            #     verts = pred_results['pred_vertices'] + pred_results['pred_translation']
            #     dump_folder = osp.join(args.paper_dir, file_name)
            #     os.makedirs(dump_folder, exist_ok=True)
            #     plt.imsave(osp.join(dump_folder, 'img.png'), img)
            #     for obj_i, vert in enumerate(verts):
            #         nr.save_obj(osp.join(dump_folder, f'{obj_i}.obj'), vert,
            #                     torch.tensor(smpl.faces.astype(np.int64)))
            # break
    eval_handler.finalize()
