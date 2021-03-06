# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
from __future__ import division
import numpy as np
import h5py
from .smpl_utils import rot6d_to_rotmat, batch_rodrigues
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

from .deformable_detr import inverse_sigmoid


class DETRsmpl(nn.Module):
    def __init__(self, detr, freeze_detr=False, head_type='hmr'):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        # self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        # self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
        self.head_type = head_type
        if self.head_type == 'hmr':
            self.smpl_head = HMRHead(hidden_dim)
        elif self.head_type == 'cmr':
            self.smpl_head = CMRHead(hidden_dim)
        else:
            raise KeyError('Unknown smpl head type: {}'.format(self.head_type))

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.detr.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.detr.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.detr.two_stage:
            query_embeds = self.detr.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.detr.transformer(srcs, masks,
                                                                                                            pos,
                                                                                                            query_embeds)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        if self.detr.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # SMPL paras
        ##FIXME There is no intermediate loss for SMPL paras yet
        if self.detr.aux_loss and False:
            smpl_para = self.smpl_head(hs)
            out["pred_smpl_pose"] = smpl_para[0][-1]
            out["pred_smpl_shape"] = smpl_para[1][-1]
            out["pred_camera"] = smpl_para[2][-1]

            for lvl in range(len(out['aux_outputs'])):
                out['aux_outputs'][lvl]["pred_smpl_pose"] = smpl_para[0][lvl]
                out['aux_outputs'][lvl]["pred_smpl_shape"] = smpl_para[1][lvl]
                out['aux_outputs'][lvl]["pred_camera"] = smpl_para[2][lvl]
        else:
            smpl_para = self.smpl_head(hs[-1].unsqueeze(0), outputs_class[-1].unsqueeze(0))
            out["pred_smpl_pose"] = smpl_para[0][-1]
            out["pred_smpl_shape"] = smpl_para[1][-1]
            out["pred_camera"] = smpl_para[2][-1]

        return out



"""
This file contains definitions of layers used as building blocks in SMPLParamRegressor
"""

class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""

    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)


class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""

    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size))

    def forward(self, x):
        return F.relu(x + self.fc_block(x))




"""
Definition of SMPL Parameter Regressor used for regressing the SMPL parameters from the 3D shape
"""


class BaseSMPLHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_num = 10

    def forward(self, x, pred_class):

        stage, bs, num_query, channel = x.shape

        # only the first K bbox for saving memory
        topk_value, topk_idx = torch.topk(pred_class[..., 1], self.max_num, 2)
        idx_stage = torch.arange(stage, device=x.device, dtype=torch.long)[:, None, None]
        idx_bs = torch.arange(bs, device=x.device, dtype=torch.long)[None, :, None]
        idx = (idx_stage * bs * num_query + idx_bs * num_query + topk_idx)
        valid_top = torch.zeros([stage * bs * num_query], dtype=torch.bool, device=x.device)
        valid_top[idx] = 1

        # only foreground bbox
        x = x.flatten(start_dim=0, end_dim=-2)
        pred_class = pred_class.flatten(start_dim=0, end_dim=-2)
        valid_class = pred_class[..., 1] > pred_class[..., 0]

        valid = valid_class * valid_top

        valid[:] = True     # for debug

        num_all = x.shape[0]
        # rotmat = x.new_zeros([num_all, 24, 3, 3])
        rotmat = torch.eye(3, dtype=x.dtype, device=x.device)[None, None, :, :].repeat([num_all, 24, 1, 1])
        betas = x.new_zeros([num_all, 10])
        camera = x.new_zeros([num_all, 3])
        camera[:, 0] = 1

        # forward the head
        rotmat[valid], betas[valid], camera[valid] = self.head_forward(x[valid])


        rotmat = rotmat.view(stage, bs, num_query, 24, 3, 3)
        betas = betas.view(stage, bs, num_query, 10)
        camera = camera.view(stage, bs, num_query, 3)
        return rotmat, betas, camera


class CMRHead(BaseSMPLHead):

    def __init__(self, in_channels, use_cpu_svd=True):
        super().__init__()
        # self.layers = nn.Sequential(FCBlock(in_channels, 1024),
        #                             # FCResBlock(1024, 1024),
        #                             # FCResBlock(1024, 1024),
        #                             nn.Linear(1024, 24 * 3 * 3 + 10))

        self.layers = nn.Sequential(FCBlock(in_channels, in_channels),
                                    FCResBlock(in_channels, in_channels),
                                    nn.Linear(in_channels, 24 * 3 * 3 + 10 + 3))

        self.use_cpu_svd = use_cpu_svd

    def head_forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)
        rotmat = x[:, :24*3*3].view(-1, 24, 3, 3).contiguous()
        betas = x[:, 24*3*3:-3].contiguous()
        camera = x[:, -3:].contiguous()

        rotmat = rotmat.view(-1, 3, 3).contiguous()
        orig_device = rotmat.device
        if self.use_cpu_svd:
            rotmat = rotmat.cpu()
        # U, S, V = batch_svd(rotmat)
        U, S, V = torch.svd(rotmat)

        rotmat = torch.matmul(U, V.transpose(1,2))

        with torch.no_grad():
            det = torch.det(rotmat)
            det = det[:, None, None]
        # det = torch.zeros(rotmat.shape[0], 1, 1).to(rotmat.device)
        # with torch.no_grad():
        #     for i in range(rotmat.shape[0]):
        #         det[i] = torch.det(rotmat[i])

        rotmat = rotmat * det
        rotmat = rotmat.view(batch_size, 24, 3, 3)
        rotmat = rotmat.to(orig_device)
        return rotmat, betas, camera


def batch_svd(A):
    """Wrapper around torch.svd that works when the input is a batch of matrices."""
    U_list = []
    S_list = []
    V_list = []
    for i in range(A.shape[0]):
        U, S, V = torch.svd(A[i])
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    U = torch.stack(U_list, dim=0)
    S = torch.stack(S_list, dim=0)
    V = torch.stack(V_list, dim=0)
    return U, S, V


class HMRHead(BaseSMPLHead):

    def __init__(self, in_channels,
                 init_param_file='data/neutral_smpl_mean_params.h5',
                 implicity_size=-1, iteraration=3):
        super().__init__()

        # Load SMPL mean parameters
        f = h5py.File(init_param_file, 'r')
        init_grot = np.array([np.pi, 0., 0.])
        init_pose = np.hstack([init_grot, f['pose'][3:]])
        init_pose = torch.tensor(init_pose.astype('float32'))
        init_rotmat = batch_rodrigues(init_pose.contiguous().view(-1, 3))
        init_contrep = init_rotmat.view(-1, 3, 3)[:, :, ::2].contiguous().view(-1)

        init_shape = torch.tensor(f['shape'][:].astype('float32'))
        init_cam = torch.tensor([0.9, 0, 0])

        self.register_buffer('init_contrep', init_contrep)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        self.npose = init_rotmat.shape[0] * 6
        # Multiply by 6 as we need to estimate two matrixes on each joins

        self.in_channels = in_channels
        if implicity_size < 0:
            implicity_size = in_channels
        self.implicity_size = implicity_size
        self.iteration = iteraration

        self.fc_blocks = nn.Sequential(
            nn.Linear(self.in_channels + 2 * 72 + 10 + 3, self.implicity_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.implicity_size, self.implicity_size),
            nn.Dropout(),
            nn.Linear(self.implicity_size, self.npose + 10 + 3)
        )
        self.init_weights()

    def head_forward(self, x):
        batch_size = x.shape[0]

        init_pose = self.init_contrep.view(1, -1).expand(batch_size, -1)
        init_shape = self.init_shape.view(1, -1).expand(batch_size, -1)
        init_cam = self.init_cam.view(1, -1).expand(batch_size, -1)

        theta = torch.cat([init_pose, init_shape, init_cam], 1)
        thetas = []
        for _ in range(self.iteration):
            total_inputs = torch.cat([x, theta], 1)
            theta = theta + self.fc_blocks(total_inputs)
            thetas.append(theta)

        pred_pose = theta[:, :self.npose].contiguous()
        pred_betas = theta[:, self.npose:self.npose + 10].contiguous()
        pred_camera = theta[:, self.npose + 10:].contiguous()
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return pred_rotmat, pred_betas, pred_camera

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                nn.init.xavier_uniform_(p)
            if 'bias' in n:
                nn.init.zeros_(p)
