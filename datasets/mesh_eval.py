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
from smplx.body_models import SMPL
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
            # self.renderer = Renderer(focal_length=FOCAL_LENGTH)
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
        smpl_params = dict(model_path='data/smpl',
                           # model_folder='data/smpl',
                           joint_mapper=joint_mapper,
                           create_glb_pose=True,
                           body_pose_param='identity',
                           create_body_pose=True,
                           create_betas=True,
                           # create_trans=True,
                           dtype=torch.float32,
                           vposer_ckpt=None,
                           gender='neutral')
        self.smpl = SMPL(**smpl_params)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.result_list = list()
        self.result_list_2d = list()
        self.h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]

        # self.collision_meter = AverageMeter('collision', ':.2f')
        # self.collision_volume = CollisionVolume(self.smpl.faces, grid_size=64).cuda()
        # self.coll_cnt = 0

    def handle(self, data_batch, pred_results, use_gt=False):
        # # Evaluate collision metric
        # pred_vertices = pred_results['pred_vertices']
        # pred_translation = pred_results['pred_translation']
        # # cur_collision_volume = self.collision_volume(pred_vertices, pred_translation)
        # # if cur_collision_volume.item() > 0:
        # #     # self.writer(f'Collision found with {cur_collision_volume.item() * 100 } L')
        # #     self.coll_cnt += 1
        # # self.collision_meter.update(cur_collision_volume.item() * 1000.)

        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_camera = pred_results['pred_camera'].cpu()
        pred_translation = pred_results['pred_translation'].cpu()
        bboxes = pred_results['bboxes'][0][:, :4]
        img = data_batch['img'].data[0][0].clone()

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        self.result_list.append(
            (pred_keypoints_3d_smpl[:, self.h36m_to_MPI] + pred_translation[:, None]).numpy())
        batch_size = pred_keypoints_3d_smpl.shape[0]
        img_size = torch.zeros(batch_size, 2).to(pred_keypoints_3d_smpl.device)
        img_size += torch.tensor(img.shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
        rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_keypoints_3d_smpl.device)

        pred_keypoints_2d_smpl = self.camera(pred_keypoints_3d_smpl, batch_size=batch_size, rotation=rotation_Is,
                                             translation=pred_translation,
                                             center=img_size / 2)
        if self.viz_dir:
            img = img.clone() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]).view(3, 1, 1)
            img_cv = img.clone().numpy()
            img_cv = (img_cv * 255).astype(np.uint8).transpose([1, 2, 0]).copy()
            for kpts, bbox in zip(pred_keypoints_2d_smpl.numpy(), bboxes):
                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                img_cv = draw_skeleton(img_cv, kpts[H36M_TO_J14, :2])
            # img_cv = draw_text(img_cv, {'mismatch': is_mismatch, 'error': str(error_smpl.mean(-1) * 1000)});
            img_cv = (img_cv / 255.)
            fname = osp.basename(data_batch['img_meta'].data[0][0]['file_name'])
            plt.imsave(osp.join(self.viz_dir, fname), img_cv)

        scale_factor = data_batch['img_meta'].data[0][0]['scale_factor']
        raw_kpts2d = pred_keypoints_2d_smpl / scale_factor
        self.result_list_2d.append(raw_kpts2d[:, self.h36m_to_MPI])
        return {'file_name': data_batch['img_meta'].data[0][0]['file_name'], 'pred_kpts3d': pred_keypoints_3d_smpl}

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


def evaluate_smpl(model, data_loader, args):
    # dump_dir = os.path.join(cfg.work_dir, f'eval_{args.dataset}')
    # os.makedirs(dump_dir, exist_ok=True)
    # if args.viz_dir:
    #     os.makedirs(args.viz_dir, exist_ok=True)

    dump_dir = f'eval_results/eval_{args.eval_dataset}'
    os.makedirs(dump_dir, exist_ok=True)
    viz_dir = f'eval_results/vis_{args.eval_dataset}'
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    FOCAL_LENGTH = 1000
    eval_handler_mapper = dict(mupots=MuPoTSEvalHandler)
    eval_handler = eval_handler_mapper[args.eval_dataset](
        writer=tqdm.write, viz_dir=viz_dir,
        FOCAL_LENGTH=FOCAL_LENGTH,
        work_dir='eval_results')

    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(data_loader)):
            file_name = data_batch['img_meta'].data[0][0]['file_name']
            try:
                bbox_results, pred_results = model(**data_batch, return_loss=False, use_gt_bboxes=args.use_gt)
                pred_results['bboxes'] = bbox_results
                if args.paper_dir:
                    os.makedirs(args.paper_dir, exist_ok=True)
                    img = denormalize(data_batch['img'].data[0][0].numpy())
                    verts = pred_results['pred_vertices'] + pred_results['pred_translation']
                    dump_folder = osp.join(args.paper_dir, file_name)
                    os.makedirs(dump_folder, exist_ok=True)
                    plt.imsave(osp.join(dump_folder, 'img.png'), img)
                    for obj_i, vert in enumerate(verts):
                        nr.save_obj(osp.join(dump_folder, f'{obj_i}.obj'), vert,
                                    torch.tensor(smpl.faces.astype(np.int64)))

                save_pack = eval_handler(data_batch, pred_results, use_gt=args.use_gt)
                save_pack.update({'bbox_results': pred_results['bboxes']})
                if args.dump_pkl:
                    with open(osp.join(dump_dir, f"{save_pack['file_name']}.pkl"), 'wb') as f:
                        pickle.dump(save_pack, f)
            except Exception as e:
                tqdm.write(f"Fail on {file_name}")
                tqdm.write(str(e))
    eval_handler.finalize()
