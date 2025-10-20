import os
import sys
import importlib
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import atan2, asin

from method.forward_kinematics import FK
from method.linear_blend_skin import batch_linear_blend_skinning_wo_rootquat
from method.ops import q_mul_q
from method.point_module import sample_and_group, Point_Transformer_Last, Local_op

sys.path.append("./submodules")
try:
    from ChamferDistancePytorch.Adapted_ChamferDistance import ChamferDistance
    ChamDist = ChamferDistance.chamfer_distance()
except Exception as e:
    print("Loading another chamfer distance.", e)
    from ChamferDistancePytorch.chamfer_python import distChamfer_a2b as ChamDist


class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5  # 1/sqrt(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        self.nn1 = nn.Linear(dim, out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, n, 3, h, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0, :, :, :, :], qkv[1, :, :, :, :], qkv[2, :, :, :, :]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(Attention(dim, mlp_dim, heads=heads, dropout=dropout)),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout))),
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, mlp_dim, heads=heads, dropout=dropout),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout))),
                ]))

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)
            x = mlp(x)
        return x


class SpatioEncoder(nn.Module):
    def __init__(self, num_joint, in_channels, token_channels, hidden_channels, kp):
        super(SpatioEncoder, self).__init__()
        self.num_joint = num_joint
        self.token_linear = nn.Linear(in_channels, token_channels)
        self.trans1 = Transformer(token_channels, 1, 4, hidden_channels, 1 - kp)

    def forward(self, pose_t):
        token_q = self.token_linear(pose_t)
        embed_q = self.trans1(token_q)
        return embed_q


class TemporalEncoder(nn.Module):
    def __init__(self, num_frame, token_channels, hidden_channels, kp):
        super(TemporalEncoder, self).__init__()
        self.num_frame = num_frame
        self.token_linear = nn.Linear(token_channels, token_channels)
        self.trans1 = Transformer(token_channels, 1, 4, hidden_channels, 1 - kp)

    def forward(self, pose_t):
        token_q = self.token_linear(pose_t)
        embed_q = self.trans1(token_q)
        return embed_q


class ShapeEncoderPCD(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, kp):
        super(ShapeEncoderPCD, self).__init__()
        self.num_joint = num_joint
        self.token_conv1 = nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(4)
        self.token_conv2 = nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(8)
        self.token_conv3 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(16)
        self.token_conv4 = nn.Conv1d(16, num_joint, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(num_joint)
        self.trans1 = Transformer(token_channels, 1, 2, embed_channels, 1 - kp)

    def forward(self, shape):
        token_s = F.relu(self.bn1(self.token_conv1(shape[:, None])))
        token_s = F.relu(self.bn2(self.token_conv2(token_s)))
        token_s = F.relu(self.bn3(self.token_conv3(token_s)))
        token_s = F.relu(self.bn4(self.token_conv4(token_s)))
        embed_s = self.trans1(token_s)
        return embed_s


class DeltaShapeDecoder(nn.Module):
    """
        Back
        change the four shape-decoder to four per-node encoder;
        process along not only K-joints (spatial) but also T-frames (temporal) dinemsions;
        Both STTrans ***before*** and ***after*** the concatenation;
    """
    def __init__(self, target_num_joint, num_joint, num_frame, token_channels, embed_channels, hidden_channels, kp):
        super(DeltaShapeDecoder, self).__init__()

        self.target_num_joint = target_num_joint
        self.num_joint = num_joint
        self.num_frame = num_frame

        self.quat_encoder = SpatioEncoder(num_joint, 4, token_channels, hidden_channels, kp)
        self.shape_encoder = ShapeEncoderPCD(num_joint, token_channels, embed_channels, kp)

        out_channels = hidden_channels + (2 * embed_channels)
        self.spatial_encoder = SpatioEncoder(num_joint, out_channels, out_channels, out_channels, kp)
        self.temporal_encoder = TemporalEncoder(num_frame, out_channels, out_channels, kp)

        self.joint_linear = nn.Linear(out_channels, out_channels)
        self.joint_acti = nn.ReLU()
        self.joint_drop = nn.Dropout(p=1 - kp)

        self.delta_linear1 = nn.Linear(out_channels, 8)
        self.delta_linear2 = nn.Linear(self.num_joint * 8, 4 * target_num_joint)

    def forward(self, q, shape_encoding_A, shape_encoding_B):
        bs, T, K, _ = q.shape

        q_spatial = q.reshape(-1, K, 4)                # bs*T, K, 4
        q_embed = self.quat_encoder(q_spatial).view(bs, T, K, -1)
        shapeA_embed = self.shape_encoder(shape_encoding_A)
        shapeB_embed = self.shape_encoder(shape_encoding_B)
        x_cat = torch.cat([q_embed, shapeA_embed[:, None].repeat(1, T, 1, 1), shapeB_embed[:, None].repeat(1, T, 1, 1)], dim=-1)

        f = x_cat.shape[-1]
        q_spatial = x_cat.view(-1, K, f)
        q_embed = self.spatial_encoder(q_spatial).view(bs, T, K, -1)
        q_temporal = q_embed.view(bs, T, K, -1).permute(0, 2, 1, 3).contiguous().view(-1, T, f)    # bs*K, T, f
        q_embed = self.temporal_encoder(q_temporal)
        q_embed = q_embed.view(bs, K, T, -1).permute(0, 2, 1, 3).contiguous()   # bs, T, K, f

        x_embed = self.joint_drop(self.joint_acti(self.joint_linear(q_embed)))
        x_embed = self.delta_linear1(x_embed).view(bs, T, -1)
        deltaq_t = self.delta_linear2(x_embed)

        return deltaq_t


def normalize(angles):
    lengths = torch.sqrt(torch.sum(torch.square(angles), dim=-1))
    normalized_angle = angles / lengths[..., None]
    return normalized_angle


class Pct(nn.Module):
    def __init__(self, dropout, output_channels=40):
        super(Pct, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        return x


class Retarget_Model(nn.Module):
    def __init__(self, num_joint=22, num_frame=60, token_channels=64, hidden_channels=256, embed_channels=128, kp=0.8):
        super(Retarget_Model, self).__init__()
        self.num_joint = num_joint
        self.num_frame = num_frame
        self.delta_shape_dec = DeltaShapeDecoder(10, num_joint, num_frame, token_channels, embed_channels, hidden_channels, kp)

    def forward(
        self,
        nameA_lst,
        nameB_lst,
        shape_encoding_A,
        shape_encoding_B,
        seqA_localnorm,         # bs T joints*3+4
        skelA_norm,             # bs T joints*3
        skelB_norm,
        shapeA,                 # bs 66
        shapeB,
        quatA_norm_cp,
        heightA,
        heightB,
        general_info,
        full_mesh_info,
        scale_factor,
    ):
        device = seqA_localnorm.device
        bs, T, K = quatA_norm_cp.size(0), quatA_norm_cp.size(1), quatA_norm_cp.size(2)

        tpose_skelB_norm = torch.reshape(skelB_norm[:, 0, :], [bs, self.num_joint, 3])
        tpose_skelB = tpose_skelB_norm * general_info['local_std'] + general_info['local_mean']
        tpose_skelA_norm = torch.reshape(skelA_norm[:, 0, :], [bs, self.num_joint, 3])
        tpose_skelA = tpose_skelA_norm * general_info['local_std'] + general_info['local_mean']
        quatA_cp = quatA_norm_cp * general_info['quat_std'][None, :] + general_info['quat_mean'][None, :]

        limb_joints = [6, 7, 10, 11, 14, 15, 16, 18, 19, 20]
        delta_quat_shape_norm = self.delta_shape_dec(quatA_norm_cp, shape_encoding_A, shape_encoding_B)
        delta_quat_shape_norm = torch.reshape(delta_quat_shape_norm, [bs, T, 10, 4])
        delta_quat_shape = (delta_quat_shape_norm * general_info['quat_std'][:, limb_joints, :] + general_info['quat_mean'][:, limb_joints, :])
        delta_quat_shape = normalize(delta_quat_shape)
        delta_q_shape = (torch.tensor([1, 0, 0, 0], dtype=torch.float32).cuda(device).repeat(bs, T, self.num_joint, 1))
        delta_q_shape[:, :, limb_joints, :] = delta_quat_shape
        quatB_rt = q_mul_q(quatA_cp, delta_q_shape)

        refA_fk = tpose_skelA[:, None].repeat(1, T, 1, 1).view(-1, tpose_skelA.shape[-2], tpose_skelA.shape[-1])
        refA_vectors = torch.Tensor([[0, 0, 1]]).repeat(K, 1)[None].repeat(bs*T, 1, 1).cuda()
        localA, vecA = FK.run_w_vec(general_info['parents'], refA_fk, refA_vectors, quatA_cp.view(-1, quatA_cp.shape[-2], quatA_cp.shape[-1]))
        localA = localA.view(bs, T, K, -1)
        vecA = vecA.view(bs, T, K, -1)

        refB_fk = tpose_skelB[:, None].repeat(1, T, 1, 1).view(-1, tpose_skelB.shape[-2], tpose_skelB.shape[-1])
        refB_vectors = torch.Tensor([[0, 0, 1]]).repeat(K, 1)[None].repeat(bs*T, 1, 1).cuda()

        localB_rt, vecB_rt = FK.run_w_vec(general_info['parents'], refB_fk, refB_vectors, quatB_rt.view(-1, quatB_rt.shape[-2], quatB_rt.shape[-1]))
        localB_rt = localB_rt.view(bs, T, K, -1)
        vecB_rt = vecB_rt.view(bs, T, K, -1)

        localB_cp, _ = FK.run_w_vec(general_info['parents'], refB_fk, refB_vectors, quatA_cp.view(-1, quatA_cp.shape[-2], quatA_cp.shape[-1]))
        localB_cp = localB_cp.view(bs, T, K, -1)

        globalA_vel = seqA_localnorm[:, :, -4:-1]
        globalA_rot = seqA_localnorm[:, :, -1]
        normalized_vin = torch.cat((torch.divide(globalA_vel, heightA[:, :, None]), globalA_rot[:, :, None]), dim=-1)
        normalized_vout = normalized_vin.clone()
        globalB_vel = normalized_vout[:, :, :-1]
        globalB_rot = normalized_vout[:, :, -1]
        globalB_rt = torch.cat((torch.multiply(globalB_vel, heightB[:, :, None]), globalB_rot[:, :, None]), dim=-1)

        return quatB_rt, localB_rt, globalB_rt, quatA_cp, localB_cp, vecA, vecB_rt


# ------------------------------------- Losses ------------------------------------

def Loss(loss_args, param_dict, full_mesh_info, sample_mesh_info):
    '''
    all losses in one function
    '''
    all_losses_dict = OrderedDict()
    for loss, hyperparam in loss_args.items():
        if hyperparam is not None:
            if 'geometric' in loss:
                all_losses_dict.update(getattr(importlib.import_module('src.' + os.path.basename(__file__).split('.')[0]), loss)(param_dict, full_mesh_info, sample_mesh_info, hyperparam))
            else:
                all_losses_dict.update(getattr(importlib.import_module('src.' + os.path.basename(__file__).split('.')[0]), loss)(param_dict, hyperparam))

    loss = sum(all_losses_dict[key] for key in all_losses_dict)
    return loss, all_losses_dict


def joint_vector_loss(param_dict, hyperparam):
    '''
    constrain the direction of each joint
    '''
    device = param_dict['localB_rt'].device
    attW = torch.ones(param_dict['num_joints']).cuda(device)
    attW[param_dict['attention_list']] = 2
    T, K = param_dict['localB_rt'].shape[1], param_dict['localB_rt'].shape[2]

    joint_vector_loss = torch.sum((torch.multiply(param_dict['temporal_mask'][:, :, None, None], torch.subtract(param_dict['vecA'], param_dict['vecB_rt']))) ** 2, dim=[0, 1, 3])
    joint_vector_loss = torch.sum(joint_vector_loss * attW)
    joint_vector_loss = torch.divide(joint_vector_loss, torch.maximum(torch.sum(param_dict['temporal_mask']), torch.tensor(1).cuda(device)))

    return OrderedDict(joint_vector_loss = joint_vector_loss * hyperparam)


def temporal_loss(param_dict, hyperparam):
    '''
    temporal constraint for the smoothness and consistency of the motion;
    normalized to unit cude to eliminate the scale effect;
    '''
    device = param_dict['localB_rt'].device
    attW = torch.ones(param_dict['num_joints']).cuda(device)
    attW[param_dict['attention_list']] = 2

    # dim1: end point, dim2: start point
    T, K = param_dict['localB_rt'].shape[1], param_dict['localB_rt'].shape[2]

    localB_flow = param_dict['localB_rt'].permute(0, 2, 1, 3)
    motion_bbox_scaleB = (localB_flow.max(dim=2)[0] - localB_flow.min(dim=2)[0]).max(dim=2)[0].detach()
    motion_bbox_scaleB += 0.01
    localB_rt_flow_matrix = (localB_flow[:, :, None].repeat(1, 1, T, 1, 1) - localB_flow[:, :, :, None].repeat(1, 1, 1, T, 1).detach()) / motion_bbox_scaleB[:, :, None, None, None]
    
    localA_flow = param_dict['localA_gt'].permute(0, 2, 1, 3)
    motion_bbox_scaleA = (localA_flow.max(dim=2)[0] - localA_flow.min(dim=2)[0]).max(dim=2)[0].detach()
    motion_bbox_scaleA += 0.01
    localA_gt_flow_matrix = (localA_flow[:, :, None].repeat(1, 1, T, 1, 1) - localA_flow[:, :, :, None].repeat(1, 1, 1, T, 1).detach()) / motion_bbox_scaleA[:, :, None, None, None]

    temporal_mask = param_dict['temporal_mask'][:, None, :, None, None].repeat(1, K, 1, T, 1) * param_dict['temporal_mask'][:, None, None, :, None].repeat(1, K, T, 1, 1)

    local_flow_loss = torch.sum((torch.multiply(temporal_mask, torch.subtract(localB_rt_flow_matrix, localA_gt_flow_matrix))) ** 2, dim=[0, 2, 3, 4])
    local_flow_loss = torch.sum(attW * local_flow_loss)
    local_flow_loss = torch.divide(local_flow_loss, torch.maximum(torch.sum(param_dict['temporal_mask']), torch.tensor(1).cuda(device))) / 60

    return OrderedDict(temporal_loss = local_flow_loss * hyperparam)


def constrain_loss(param_dict, hyperparam):
    '''
    constrain the motion modification
    '''
    device = param_dict['localB_rt'].device
    attW = torch.ones(param_dict['num_joints']).cuda(device)
    attW[param_dict['attention_list']] = 2

    ae_joints_err = torch.sum((torch.multiply(param_dict['temporal_mask'][:, :, None, None], torch.subtract(param_dict['localB_rt'], param_dict['localB_base']))) ** 2, dim=[0, 1, 3])
    local_ae_loss = torch.sum(attW * ae_joints_err)
    local_ae_loss = torch.divide(local_ae_loss, torch.maximum(torch.sum(param_dict['temporal_mask']), torch.tensor(1).cuda(device)))

    quat_ae_loss = torch.sum((torch.multiply(param_dict['temporal_mask'][:, :, None, None], torch.subtract(param_dict['quatB_rt'], param_dict['quatB_base']))) ** 2, dim=[0, 1, 3])
    quat_ae_loss = torch.sum(attW * quat_ae_loss)
    quat_ae_loss = torch.divide(quat_ae_loss, torch.maximum(torch.sum(param_dict['temporal_mask']), torch.tensor(1).cuda(device)))

    return OrderedDict(local_constrain_loss = local_ae_loss * hyperparam, quat_constrain_loss = quat_ae_loss * hyperparam)


def twist_constrain_loss(param_dict, hyperparam):
    '''
    constrain the maximum of the twist, adapted from R2ET
    '''
    device = param_dict['localB_rt'].device
    rads = param_dict['alpha'] / 180.0

    twistB_loss = torch.mean(torch.square(torch.maximum(
        torch.tensor(0).cuda(device),
        torch.abs(euler_y(param_dict['quatB_rt'], param_dict['euler_order'])) - rads * np.pi,
    )))

    return OrderedDict(twist_constrain_loss = twistB_loss * hyperparam)


def new_geometric_loss(param_dict, full_mesh_info, sample_mesh_info, hyperparam):
    '''
    New SDF loss: use SDF for losses
    '''
    device = param_dict['quatB_rt'].device
    bs = param_dict['quatB_rt'].shape[0]
    tposeB_norm = torch.reshape(param_dict['skelB_norm'][:, 0, :], [bs, param_dict['num_joints'], 3])
    tposeB = tposeB_norm * param_dict['local_std'] + param_dict['local_mean']

    # ---------------------------- try to process all meshes together ----------------------------
    scales = []
    centroids = []
    vertices_lsts = []
    vertices_normals_lsts = []
    sk_weights_lsts = []
    for i in range(bs):
        nameB = param_dict['all_names'][param_dict['indexesB'][i]]
        scales.append(full_mesh_info['scale_info'][nameB]['scale'][None, :])
        centroids.append(full_mesh_info['scale_info'][nameB]['centroid'][None, :])
        tmp_lsts = []
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_lefthand_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_righthand_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_leftarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_wo_leftarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_rightarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_wo_rightarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_leftleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_wo_leftleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_rightleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertices'][nameB][sample_mesh_info['resample_wo_rightleg_vertices'][nameB]])
        vertices_lsts.append(torch.cat(tmp_lsts, dim=0)[None, :])
        tmp_lsts = []
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_lefthand_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_righthand_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_leftarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_wo_leftarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_rightarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_wo_rightarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_leftleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_wo_leftleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_rightleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['rest_vertex_normals'][nameB][sample_mesh_info['resample_wo_rightleg_vertices'][nameB]])
        vertices_normals_lsts.append(torch.cat(tmp_lsts, dim=0)[None, :])
        tmp_lsts = []
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_lefthand_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_righthand_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_leftarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_wo_leftarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_rightarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_wo_rightarm_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_leftleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_wo_leftleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_rightleg_vertices'][nameB]])
        tmp_lsts.append(full_mesh_info['skinning_weights'][nameB][sample_mesh_info['resample_wo_rightleg_vertices'][nameB]])
        sk_weights_lsts.append(torch.cat(tmp_lsts, dim=0)[None, :])

    scales = torch.cat(scales, dim=0)
    centroids = torch.cat(centroids, dim=0)
    vertices_lsts = torch.cat(vertices_lsts, dim=0)
    vertices_normals_lsts = torch.cat(vertices_normals_lsts, dim=0)
    sk_weights_lsts = torch.cat(sk_weights_lsts, dim=0)

    sdf_loss = allinone_sdf_loss(
        param_dict['parents'],
        param_dict['quatB_rt'],
        tposeB,
        vertices_lsts,
        sk_weights_lsts,
        vertices_normals_lsts,
        scales,
        centroids,
        hand_num = param_dict['hand_vert_num'],
        limb_num = param_dict['limb_vert_num'],
        wo_limb_num = param_dict['wo_limb_vert_num'],
    )

    assert isinstance(hyperparam, list)
    return OrderedDict(geometric_loss = sdf_loss * hyperparam[0])


def allinone_sdf_loss(
    parents,
    quatB_rt,
    rest_skelB,
    vertices,
    skinB_weights,
    vertices_normals,
    scales,
    centroids,
    ifth=False,
    hand_num=100,
    limb_num=150,
    wo_limb_num=2050,
):
    '''
    avoid interpenetration
    quatB_rt: (T, 22, 4)
    meshB: (vertex_num, 3)
    skinB_weights: (vertex_num, bone_num)
    '''
    bs, T = quatB_rt.shape[0], quatB_rt.shape[1]
    point_number = skinB_weights.shape[1]
    device = quatB_rt.device

    # LBS
    # vertices_lbs_old = linear_blend_skinning(parents, quatB_rt, rest_skelB, meshB, skinB_weights, vertices_normals)
    vertices_lbs, normals_lbs = batch_linear_blend_skinning_wo_rootquat(parents, quatB_rt, rest_skelB, vertices, skinB_weights, vertices_normals)

    arm_threshold = 1.
    leg_threshold = 1.
    total_num = 2* hand_num + 4 * (limb_num + wo_limb_num)      # 1/2: 4500;    1/4: 2250;      2/1: 17600;
    assert total_num == point_number, "the numbers of vertices are wrong"

    # attractive loss
    # lefthand
    vertices_lefthand = vertices_lbs[:, :, :hand_num]
    vertices_wo_leftarm = vertices_lbs[:, :, hand_num*2+limb_num : hand_num*2+limb_num+wo_limb_num].detach()
    normals_wo_leftarm = normals_lbs[:, :, hand_num*2+limb_num : hand_num*2+limb_num+wo_limb_num].detach()
    _, min_query = ChamDist(vertices_lefthand.view(-1, hand_num, 3), vertices_wo_leftarm.view(-1, wo_limb_num, 3))
    expanded_min_indexes = min_query.view(bs, T, hand_num).long().unsqueeze(-1).expand(-1, -1, -1, 3)
    refer_vector_lefthand = torch.gather(vertices_wo_leftarm, dim=2, index=expanded_min_indexes) - vertices_lefthand
    refer_normals_lefthand = torch.gather(normals_wo_leftarm, dim=2, index=expanded_min_indexes)
    sdf_vector_multiply_lefthand = torch.sum(refer_vector_lefthand * refer_normals_lefthand, dim=-1)
    attractive_loss_lefthand = -torch.sum(torch.clamp(sdf_vector_multiply_lefthand, max=0)) / bs / T / hand_num
    # righthand
    vertices_righthand = vertices_lbs[:, :, hand_num : hand_num*2]
    vertices_wo_rightarm = vertices_lbs[:, :, hand_num*2+limb_num*2+wo_limb_num : hand_num*2+limb_num*2+wo_limb_num*2].detach()
    normals_wo_tightarm = normals_lbs[:, :, hand_num*2+limb_num*2+wo_limb_num : hand_num*2+limb_num*2+wo_limb_num*2].detach()
    _, min_query = ChamDist(vertices_righthand.view(-1, hand_num, 3), vertices_wo_rightarm.view(-1, wo_limb_num, 3))
    expanded_min_indexes = min_query.view(bs, T, hand_num).long().unsqueeze(-1).expand(-1, -1, -1, 3)
    refer_vector_righthand = torch.gather(vertices_wo_rightarm, dim=2, index=expanded_min_indexes) - vertices_righthand
    refer_normals_righthand = torch.gather(normals_wo_tightarm, dim=2, index=expanded_min_indexes)
    sdf_vector_multiply_righthand = torch.sum(refer_vector_righthand * refer_normals_righthand, dim=-1)
    attractive_loss_righthand = -torch.sum(torch.clamp(sdf_vector_multiply_righthand, max=0)) / bs / T / hand_num

    # get the distances & find the nearest neighbor
    # leftarm
    # visualize_pointcloud(vertices_leftarm.detach().cpu().numpy(), vertices_wo_leftarm[0].detach().cpu().numpy(), scale=True)
    # mydist1, mydist2, min_query, idx2 = ChamDist(vertices_leftarm, vertices_wo_leftarm)
    vertices_leftarm = vertices_lbs[:, :, hand_num*2 : hand_num*2+limb_num]
    vertices_wo_leftarm = vertices_lbs[:, :, hand_num*2+limb_num : hand_num*2+limb_num+wo_limb_num].detach()
    normals_wo_leftarm = normals_lbs[:, :, hand_num*2+limb_num : hand_num*2+limb_num+wo_limb_num].detach()
    _, min_query = ChamDist(vertices_leftarm.view(-1, limb_num, 3), vertices_wo_leftarm.view(-1, wo_limb_num, 3))
    expanded_min_indexes = min_query.view(bs, T, limb_num).long().unsqueeze(-1).expand(-1, -1, -1, 3)
    refer_vector_leftarm = torch.gather(vertices_wo_leftarm, dim=2, index=expanded_min_indexes) - vertices_leftarm
    refer_normals_leftarm = torch.gather(normals_wo_leftarm, dim=2, index=expanded_min_indexes)
    sdf_vector_multiply_leftarm = torch.sum(refer_vector_leftarm * refer_normals_leftarm, dim=-1)
    # filter all negative SDF values
    if ifth:
        sdf_vector_multiply_leftarm[sdf_vector_multiply_leftarm < arm_threshold] = 0
        filtered_sdfloss_leftarm = torch.sum(sdf_vector_multiply_leftarm) / bs / T / limb_num
    else:
        filtered_sdfloss_leftarm = torch.sum(torch.clamp(sdf_vector_multiply_leftarm, min=0)) / bs / T / limb_num

    # rightarm
    # mydist1, mydist2, min_query, idx2 = ChamDist(vertices_rightarm, vertices_wo_rightarm)
    vertices_rightarm = vertices_lbs[:, :, hand_num*2+limb_num+wo_limb_num : hand_num*2+limb_num*2+wo_limb_num]
    vertices_wo_rightarm = vertices_lbs[:, :, hand_num*2+limb_num*2+wo_limb_num : hand_num*2+limb_num*2+wo_limb_num*2].detach()
    normals_wo_rightarm = normals_lbs[:, :, hand_num*2+limb_num*2+wo_limb_num : hand_num*2+limb_num*2+wo_limb_num*2].detach()
    _, min_query = ChamDist(vertices_rightarm.view(-1, limb_num, 3), vertices_wo_rightarm.view(-1, wo_limb_num, 3))
    expanded_min_indexes = min_query.view(bs, T, limb_num).long().unsqueeze(-1).expand(-1, -1, -1, 3)
    refer_vector_rightarm = torch.gather(vertices_wo_rightarm, dim=2, index=expanded_min_indexes) - vertices_rightarm
    refer_normals_rightarm = torch.gather(normals_wo_rightarm, dim=2, index=expanded_min_indexes)
    sdf_vector_multiply_rightarm = torch.sum(refer_vector_rightarm * refer_normals_rightarm, dim=-1)
    # filter all negative SDF values
    if ifth:
        sdf_vector_multiply_rightarm[sdf_vector_multiply_rightarm < arm_threshold] = 0
        filtered_sdfloss_rightarm = torch.sum(sdf_vector_multiply_rightarm) / bs / T / limb_num
    else:
        filtered_sdfloss_rightarm = torch.sum(torch.clamp(sdf_vector_multiply_rightarm, min=0)) / bs / T / limb_num

    # leftleg
    # mydist1, mydist2, min_query, idx2 = ChamDist(vertices_leftleg, vertices_wo_leftleg)
    vertices_leftleg = vertices_lbs[:, :, hand_num*2+limb_num*2+wo_limb_num*2 : hand_num*2+limb_num*3+wo_limb_num*2]
    vertices_wo_leftleg = vertices_lbs[:, :, hand_num*2+limb_num*3+wo_limb_num*2 : hand_num*2+limb_num*3+wo_limb_num*3].detach()
    normals_wo_leftleg = normals_lbs[:, :, hand_num*2+limb_num*3+wo_limb_num*2 : hand_num*2+limb_num*3+wo_limb_num*3].detach()
    _, min_query = ChamDist(vertices_leftleg.view(-1, limb_num, 3), vertices_wo_leftleg.view(-1, wo_limb_num, 3))
    expanded_min_indexes = min_query.view(bs, T, limb_num).long().unsqueeze(-1).expand(-1, -1, -1, 3)
    refer_vector_leftleg = torch.gather(vertices_wo_leftleg, dim=2, index=expanded_min_indexes) - vertices_leftleg
    refer_normals_leftleg = torch.gather(normals_wo_leftleg, dim=2, index=expanded_min_indexes)
    sdf_vector_multiply_leftleg = torch.sum(refer_vector_leftleg * refer_normals_leftleg, dim=-1)
    # filter all negative SDF values
    if ifth:
        sdf_vector_multiply_leftleg[sdf_vector_multiply_leftleg < leg_threshold] = 0
        filtered_sdfloss_leftleg = torch.sum(sdf_vector_multiply_leftleg) / bs / T / limb_num
    else:
        filtered_sdfloss_leftleg = torch.sum(torch.clamp(sdf_vector_multiply_leftleg, min=0)) / bs / T / limb_num

    # rightleg
    # mydist1, mydist2, min_query, idx2 = ChamDist(vertices_rightleg, vertices_wo_rightleg)
    vertices_rightleg = vertices_lbs[:, :, hand_num*2+limb_num*3+wo_limb_num*3 : hand_num*2+limb_num*4+wo_limb_num*3]
    vertices_wo_rightleg = vertices_lbs[:, :, hand_num*2+limb_num*4+wo_limb_num*3 : hand_num*2+limb_num*4+wo_limb_num*4].detach()
    normals_wo_rightleg = normals_lbs[:, :, hand_num*2+limb_num*4+wo_limb_num*3 : hand_num*2+limb_num*4+wo_limb_num*4].detach()
    _, min_query = ChamDist(vertices_rightleg.view(-1, limb_num, 3), vertices_wo_rightleg.view(-1, wo_limb_num, 3))
    expanded_min_indexes = min_query.view(bs, T, limb_num).long().unsqueeze(-1).expand(-1, -1, -1, 3)
    refer_vector_rightleg = torch.gather(vertices_wo_rightleg, dim=2, index=expanded_min_indexes) - vertices_rightleg
    refer_normals_rightleg = torch.gather(normals_wo_rightleg, dim=2, index=expanded_min_indexes)
    sdf_vector_multiply_rightleg = torch.sum(refer_vector_rightleg * refer_normals_rightleg, dim=-1)
    # filter all negative SDF values
    if ifth:
        sdf_vector_multiply_rightleg[sdf_vector_multiply_rightleg < leg_threshold] = 0
        filtered_sdfloss_rightleg = torch.sum(sdf_vector_multiply_rightleg) / bs / T / limb_num
    else:
        filtered_sdfloss_rightleg = torch.sum(torch.clamp(sdf_vector_multiply_rightleg, min=0)) / bs / T / limb_num

    return filtered_sdfloss_leftarm + filtered_sdfloss_rightarm + filtered_sdfloss_leftleg + filtered_sdfloss_rightleg

# --------------------------- utils ---------------------------

def get_bounding_boxes(vertices):
    num_people = vertices.shape[0]
    boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
    for i in range(num_people):
        boxes[i, 0, :] = vertices[i].min(dim=0)[0]
        boxes[i, 1, :] = vertices[i].max(dim=0)[0]
    return boxes


def euler_y(angles, order="yzx"):
    q = normalize(angles)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    if order == "xyz":
        ex = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        ey = asin(torch.clamp(2 * (q0 * q2 - q3 * q1), -1, 1))
        ez = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        return torch.stack(values=[ex, ez], dim=-1)[:, :, 1:]
    elif order == "yzx":
        ex = atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
        ey = atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
        ez = asin(torch.clamp(2 * (q1 * q2 + q3 * q0), -1, 1))
        return ey[:, :, 1:]
    else:
        raise Exception("Unknown Euler order!")


def euler_rot(angles):
    q = normalize(angles)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    ex = atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
    ey = atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
    ez = asin(torch.clamp(2 * (q1 * q2 + q3 * q0), -1, 1))

    rotx = ex[..., :]
    roty = ey[..., :]
    rotz = ez[..., :]

    rot = torch.stack([rotx, roty, rotz], dim=-1)
    return rot
