import sys
import torch
import torch.nn as nn
import trimesh
import numpy as np
from scipy.spatial import distance

from sdf import SDF, SDF2
from src.linear_blend_skin import linear_blend_skinning, linear_blend_skinning_old
from src.utils import get_bounding_boxes
from src.forward_kinematics import FK

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("./submodules")
try:
    from ChamferDistancePytorch.Adapted_ChamferDistance import dist_chamfer_3D as dist_chamfer_3D_Mine
    cham3D = dist_chamfer_3D_Mine.chamfer_3DDist()
except:
    print("Loading another chamfer distance.")
    from ChamferDistancePytorch.Adapted_ChamferDistance import distChamfer_a2b as cham3D


def penetrate_r2et(skelB, parents, nameB, quatB_rt, full_mesh_info):
    rest_skelB = skelB.reshape(-1, 3)
    meshB = full_mesh_info['rest_vertices'][nameB]
    skinB_weights = full_mesh_info['skinning_weights'][nameB]
    get_sdf = SDF()
    vertices_lbs = linear_blend_skinning(parents, quatB_rt, rest_skelB, meshB, skinB_weights)

    scale_factor = 0.2
    boxes = get_bounding_boxes(vertices_lbs)
    boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
    boxes_scale = ((1 + scale_factor) * 0.5 * (boxes[:, 1] - boxes[:, 0]).max(dim=-1)[0][:, None, None])
    vertices_centered = vertices_lbs - boxes_center
    vertices_centered_scaled = vertices_centered / boxes_scale
    assert vertices_centered_scaled.min() >= -1
    assert vertices_centered_scaled.max() <= 1

    T = vertices_lbs.shape[0]
    body_vertices_lst = full_mesh_info['body_vertices'][nameB]
    head_vertices_lst = full_mesh_info['head_vertices'][nameB]
    leftarm_vertices_lst = full_mesh_info['leftarm_vertices'][nameB]
    rightarm_vertices_lst = full_mesh_info['rightarm_vertices'][nameB]
    leftleg_vertices_lst = full_mesh_info['leftleg_vertices'][nameB]
    rightleg_vertices_lst = full_mesh_info['rightleg_vertices'][nameB]

    vertices_centered_scaled_la = vertices_centered_scaled[:, leftarm_vertices_lst, :]
    vertices_centered_scaled_ra = vertices_centered_scaled[:, rightarm_vertices_lst, :]
    vertices_centered_scaled_ll = vertices_centered_scaled[:, leftleg_vertices_lst, :]
    vertices_centered_scaled_rl = vertices_centered_scaled[:, rightleg_vertices_lst, :]

    vert_num_all = 0
    pene_num_all = 0

    for t in range(T):
        body_vertices = (vertices_centered_scaled[t, body_vertices_lst, :].cpu().numpy())
        body_point_cloud = trimesh.points.PointCloud(vertices=body_vertices)
        body_mesh = body_point_cloud.convex_hull
        if False:
            body_mesh.show()
        body_vertices = torch.from_numpy(np.array(body_mesh.vertices).astype(np.single)).cuda(quatB_rt.device)
        body_faces = torch.from_numpy(np.array(body_mesh.faces).astype(np.int32)).cuda(quatB_rt.device)
        body_sdf = get_sdf(body_faces, body_vertices[None,], grid_size=32)

        head_vertices = (vertices_centered_scaled[t, head_vertices_lst, :].cpu().numpy())
        head_point_cloud = trimesh.points.PointCloud(vertices=head_vertices)
        head_mesh = head_point_cloud.convex_hull
        head_vertices = torch.from_numpy(np.array(head_mesh.vertices).astype(np.single)).cuda(quatB_rt.device)
        head_faces = torch.from_numpy(np.array(head_mesh.faces).astype(np.int32)).cuda(quatB_rt.device)
        head_sdf = get_sdf(head_faces, head_vertices[None,], grid_size=32)

        total_sdf = body_sdf + head_sdf

        vertices_local_la = vertices_centered_scaled_la[t, :, :]
        vert_num = vertices_local_la.shape[0]
        vertices_grid_la = vertices_local_la.view(1, -1, 1, 1, 3)
        sdf_val_la = nn.functional.grid_sample(total_sdf[0][None, None], vertices_grid_la, align_corners=False).view(-1)
        pene_num_la = (sdf_val_la > 0).sum()
        vert_num_all += vert_num
        pene_num_all += pene_num_la.item()

        vertices_local_ra = vertices_centered_scaled_ra[t, :, :]
        vert_num = vertices_local_ra.shape[0]
        vertices_grid_ra = vertices_local_ra.view(1, -1, 1, 1, 3)
        sdf_val_ra = nn.functional.grid_sample(total_sdf[0][None, None], vertices_grid_ra, align_corners=False).view(-1)
        pene_num_ra = (sdf_val_ra > 0).sum()
        vert_num_all += vert_num
        pene_num_all += pene_num_ra.item()

        vertices_local_ll = vertices_centered_scaled_ll[t, :, :]
        vert_num = vertices_local_ll.shape[0]
        vertices_grid_ll = vertices_local_ll.view(1, -1, 1, 1, 3)
        sdf_val_ll = nn.functional.grid_sample(total_sdf[0][None, None], vertices_grid_ll, align_corners=False).view(-1)
        pene_num_ll = (sdf_val_ll > 0).sum()
        vert_num_all += vert_num
        pene_num_all += pene_num_ll.item()

        vertices_local_rl = vertices_centered_scaled_rl[t, :, :]
        vert_num = vertices_local_rl.shape[0]
        vertices_grid_rl = vertices_local_rl.view(1, -1, 1, 1, 3)
        sdf_val_rl = nn.functional.grid_sample(total_sdf[0][None, None], vertices_grid_rl, align_corners=False).view(-1)
        pene_num_rl = (sdf_val_rl > 0).sum()
        vert_num_all += vert_num
        pene_num_all += pene_num_rl.item()
    
    return vert_num_all, pene_num_all


def penetrate_1(skelB, parents, nameB, quatB_rt, full_mesh_info):
    '''
    This is different from the geometric loss, all vertices are taken into account.
    '''
    rest_skelB = skelB.reshape(-1, 3)
    meshB = full_mesh_info['rest_vertices'][nameB]
    skinB_weights = full_mesh_info['skinning_weights'][nameB]
    vertices_normals = full_mesh_info['rest_vertex_normals'][nameB]
    vertices_lbs, normals_lbs = linear_blend_skinning(parents, quatB_rt, rest_skelB, meshB, skinB_weights, vertices_normals)

    vertices_lbs_centered = vertices_lbs - full_mesh_info['scale_info'][nameB]['centroid']
    vertices_lbs_centered_scaled = vertices_lbs_centered * 2 / full_mesh_info['scale_info'][nameB]['scale']

    leftarm_vertices_lst = full_mesh_info['leftarm_vertices'][nameB]
    rightarm_vertices_lst = full_mesh_info['rightarm_vertices'][nameB]
    leftleg_vertices_lst = full_mesh_info['leftleg_vertices'][nameB]
    rightleg_vertices_lst = full_mesh_info['rightleg_vertices'][nameB]
    wo_leftarm_vertices_lst = full_mesh_info['wo_leftarm_vertices'][nameB]
    wo_rightarm_vertices_lst = full_mesh_info['wo_rightarm_vertices'][nameB]
    wo_leftleg_vertices_lst = full_mesh_info['wo_leftleg_vertices'][nameB]
    wo_rightleg_vertices_lst = full_mesh_info['wo_rightleg_vertices'][nameB]

    vertices_leftarm = vertices_lbs_centered_scaled[:, leftarm_vertices_lst, :]
    vertices_wo_leftarm = vertices_lbs_centered_scaled[:, wo_leftarm_vertices_lst, :]
    normals_wo_leftarm = normals_lbs[:, wo_leftarm_vertices_lst, :]

    vertices_rightarm = vertices_lbs_centered_scaled[:, rightarm_vertices_lst, :]
    vertices_wo_rightarm = vertices_lbs_centered_scaled[:, wo_rightarm_vertices_lst, :]
    normals_wo_rightarm = normals_lbs[:, wo_rightarm_vertices_lst, :]

    vertices_leftleg = vertices_lbs_centered_scaled[:, leftleg_vertices_lst, :]
    vertices_wo_leftleg = vertices_lbs_centered_scaled[:, wo_leftleg_vertices_lst, :]
    normals_wo_leftleg = normals_lbs[:, wo_leftleg_vertices_lst, :]

    vertices_rightleg = vertices_lbs_centered_scaled[:, rightleg_vertices_lst, :]
    vertices_wo_rightleg = vertices_lbs_centered_scaled[:, wo_rightleg_vertices_lst, :]
    normals_wo_rightleg = normals_lbs[:, wo_rightleg_vertices_lst, :]

    vert_num = 0
    pene_num = 0

    # leftarm
    _, min_query = cham3D(vertices_leftarm, vertices_wo_leftarm)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_leftarm = torch.gather(vertices_wo_leftarm, dim=1, index=expanded_min_indexes) - vertices_leftarm
    refer_normals_leftarm = torch.gather(normals_wo_leftarm, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_leftarm = torch.sum(refer_vector_leftarm * refer_normals_leftarm, dim=-1)
    pene_num += (sdf_vector_multiply_leftarm > 0).sum()
    vert_num += vertices_leftarm.shape[0] * vertices_leftarm.shape[1]

    # rightarm
    _, min_query = cham3D(vertices_rightarm, vertices_wo_rightarm)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_rightarm = torch.gather(vertices_wo_rightarm, dim=1, index=expanded_min_indexes) - vertices_rightarm
    refer_normals_rightarm = torch.gather(normals_wo_rightarm, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_rightarm = torch.sum(refer_vector_rightarm * refer_normals_rightarm, dim=-1)
    pene_num += (sdf_vector_multiply_rightarm > 0).sum()
    vert_num += vertices_rightarm.shape[0] * vertices_rightarm.shape[1]
    # leftleg
    _, min_query = cham3D(vertices_leftleg, vertices_wo_leftleg)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_leftleg = torch.gather(vertices_wo_leftleg, dim=1, index=expanded_min_indexes) - vertices_leftleg
    refer_normals_leftleg = torch.gather(normals_wo_leftleg, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_leftleg = torch.sum(refer_vector_leftleg * refer_normals_leftleg, dim=-1)
    pene_num += (sdf_vector_multiply_leftleg > 0).sum()
    vert_num += vertices_leftleg.shape[0] * vertices_leftleg.shape[1]
    # rightleg
    _, min_query = cham3D(vertices_rightleg, vertices_wo_rightleg)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_rightleg = torch.gather(vertices_wo_rightleg, dim=1, index=expanded_min_indexes) - vertices_rightleg
    refer_normals_rightleg = torch.gather(normals_wo_rightleg, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_rightleg = torch.sum(refer_vector_rightleg * refer_normals_rightleg, dim=-1)
    pene_num += (sdf_vector_multiply_rightleg > 0).sum()
    vert_num += vertices_rightleg.shape[0] * vertices_rightleg.shape[1]

    return vert_num, pene_num


def penetrate_2(skelB, parents, nameB, quatB_rt, full_mesh_info):
    rest_skelB = skelB.reshape(-1, 3)
    meshB = full_mesh_info['rest_vertices'][nameB]
    skinB_weights = full_mesh_info['skinning_weights'][nameB]
    vertices_normals = full_mesh_info['rest_vertex_normals'][nameB]

    # lbs the mesh of the full body
    vertices_lbs, normals_lbs = linear_blend_skinning(parents, quatB_rt, rest_skelB, meshB, skinB_weights, vertices_normals)

    # lbs the filled mesh for each wo_limb
    vertices_wo_leftarm, normals_wo_leftarm = linear_blend_skinning(
        parents, quatB_rt, rest_skelB,
        full_mesh_info['rest_vertices_wo_leftarm'][nameB],
        full_mesh_info['skinning_weights_wo_leftarm'][nameB],
        full_mesh_info['rest_normals_wo_leftarm'][nameB],
    )
    vertices_wo_rightarm, normals_wo_rightarm = linear_blend_skinning(
        parents, quatB_rt, rest_skelB,
        full_mesh_info['rest_vertices_wo_rightarm'][nameB],
        full_mesh_info['skinning_weights_wo_rightarm'][nameB],
        full_mesh_info['rest_normals_wo_rightarm'][nameB],
    )
    vertices_wo_leftleg, normals_wo_leftleg = linear_blend_skinning(
        parents, quatB_rt, rest_skelB,
        full_mesh_info['rest_vertices_wo_leftleg'][nameB],
        full_mesh_info['skinning_weights_wo_leftleg'][nameB],
        full_mesh_info['rest_normals_wo_leftleg'][nameB],
    )
    vertices_wo_rightleg, normals_wo_rightleg = linear_blend_skinning(
        parents, quatB_rt, rest_skelB,
        full_mesh_info['rest_vertices_wo_rightleg'][nameB],
        full_mesh_info['skinning_weights_wo_rightleg'][nameB],
        full_mesh_info['rest_normals_wo_rightleg'][nameB],
    )

    # move centroid to origin, and scale
    vertices_lbs_centered = vertices_lbs - full_mesh_info['scale_info'][nameB]['centroid']
    vertices_lbs_centered_scaled = vertices_lbs_centered * 2 / full_mesh_info['scale_info'][nameB]['scale']
    vertices_wo_leftarm = vertices_wo_leftarm - full_mesh_info['scale_info'][nameB]['centroid']
    vertices_wo_leftarm = vertices_wo_leftarm * 2 / full_mesh_info['scale_info'][nameB]['scale']
    vertices_wo_rightarm = vertices_wo_rightarm - full_mesh_info['scale_info'][nameB]['centroid']
    vertices_wo_rightarm = vertices_wo_rightarm * 2 / full_mesh_info['scale_info'][nameB]['scale']
    vertices_wo_leftleg = vertices_wo_leftleg - full_mesh_info['scale_info'][nameB]['centroid']
    vertices_wo_leftleg = vertices_wo_leftleg * 2 / full_mesh_info['scale_info'][nameB]['scale']
    vertices_wo_rightleg = vertices_wo_rightleg - full_mesh_info['scale_info'][nameB]['centroid']
    vertices_wo_rightleg = vertices_wo_rightleg * 2 / full_mesh_info['scale_info'][nameB]['scale']

    # slice the mesh of each limb
    leftarm_vertices_lst = full_mesh_info['leftarm_vertices'][nameB]
    rightarm_vertices_lst = full_mesh_info['rightarm_vertices'][nameB]
    leftleg_vertices_lst = full_mesh_info['leftleg_vertices'][nameB]
    rightleg_vertices_lst = full_mesh_info['rightleg_vertices'][nameB]
    vertices_leftarm = vertices_lbs_centered_scaled[:, leftarm_vertices_lst, :]
    vertices_rightarm = vertices_lbs_centered_scaled[:, rightarm_vertices_lst, :]
    vertices_leftleg = vertices_lbs_centered_scaled[:, leftleg_vertices_lst, :]
    vertices_rightleg = vertices_lbs_centered_scaled[:, rightleg_vertices_lst, :]

    vert_num = 0
    pene_num = 0

    # leftarm
    _, min_query = cham3D(vertices_leftarm, vertices_wo_leftarm)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_leftarm = torch.gather(vertices_wo_leftarm, dim=1, index=expanded_min_indexes) - vertices_leftarm
    refer_normals_leftarm = torch.gather(normals_wo_leftarm, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_leftarm = torch.sum(refer_vector_leftarm * refer_normals_leftarm, dim=-1)
    pene_num += (sdf_vector_multiply_leftarm > 0).sum()
    vert_num += vertices_leftarm.shape[0] * vertices_leftarm.shape[1]

    # rightarm
    _, min_query = cham3D(vertices_rightarm, vertices_wo_rightarm)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_rightarm = torch.gather(vertices_wo_rightarm, dim=1, index=expanded_min_indexes) - vertices_rightarm
    refer_normals_rightarm = torch.gather(normals_wo_rightarm, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_rightarm = torch.sum(refer_vector_rightarm * refer_normals_rightarm, dim=-1)
    pene_num += (sdf_vector_multiply_rightarm > 0).sum()
    vert_num += vertices_rightarm.shape[0] * vertices_rightarm.shape[1]
    # leftleg
    _, min_query = cham3D(vertices_leftleg, vertices_wo_leftleg)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_leftleg = torch.gather(vertices_wo_leftleg, dim=1, index=expanded_min_indexes) - vertices_leftleg
    refer_normals_leftleg = torch.gather(normals_wo_leftleg, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_leftleg = torch.sum(refer_vector_leftleg * refer_normals_leftleg, dim=-1)
    pene_num += (sdf_vector_multiply_leftleg > 0).sum()
    vert_num += vertices_leftleg.shape[0] * vertices_leftleg.shape[1]
    # rightleg
    _, min_query = cham3D(vertices_rightleg, vertices_wo_rightleg)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_rightleg = torch.gather(vertices_wo_rightleg, dim=1, index=expanded_min_indexes) - vertices_rightleg
    refer_normals_rightleg = torch.gather(normals_wo_rightleg, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_rightleg = torch.sum(refer_vector_rightleg * refer_normals_rightleg, dim=-1)
    pene_num += (sdf_vector_multiply_rightleg > 0).sum()
    vert_num += vertices_rightleg.shape[0] * vertices_rightleg.shape[1]

    return vert_num, pene_num


def curvature_1(parents, skel, quat):
    # xyz: meter; v: m/s; a: m2/s
    T, K = quat.shape[0], quat.shape[1]
    joint_path = FK.run(parents, skel, quat) / 100.
    # dx = joint_path[1:] - joint_path[:-1]
    v = (joint_path[1:] - joint_path[:-1]) * 60
    dv = v[1:] - v[:-1]
    a = torch.norm(dv * 60, dim=-1).mean(dim=0)
    return a[None]


def curvature_2(parents, skel, quat):
    T, K = quat.shape[0], quat.shape[1]
    joint_path = FK.run(parents, skel, quat) / 100.
    v = (joint_path[1:] - joint_path[:-1]) * 60  # velocity (frames to seconds)
    a = (v[1:] - v[:-1]) * 60  # acceleration (frames to seconds)
    # True curvature: kappa = ||v x a|| / ||v||^3
    v_mid = v[1:]  # shape (T-2, K, 3)
    a_mid = a      # shape (T-2, K, 3)
    cross = torch.cross(v_mid, a_mid, dim=-1)
    v_norm = torch.norm(v_mid, dim=-1)
    cross_norm = torch.norm(cross, dim=-1)
    # Avoid division by zero
    eps = 1e-8
    curvature = cross_norm / (v_norm ** 3 + eps)
    mean_curvature = curvature.mean(dim=0)
    return mean_curvature[None]
