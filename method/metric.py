import sys
import torch

from method.linear_blend_skin import linear_blend_skinning
from method.forward_kinematics import FK

sys.path.append("./submodules")
try:
    from ChamferDistancePytorch.Adapted_ChamferDistance import ChamferDistance
    ChamDist = ChamferDistance.chamfer_distance()
except Exception as e:
    print("Loading another chamfer distance.", e)
    from ChamferDistancePytorch.chamfer_python import distChamfer_a2b as ChamDist


def penetrate_1(skelB, parents, nameB, quatB_rt, full_mesh_info):
    '''
    This is different from the geometric loss, all vertices are taken into account;
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
    _, min_query = ChamDist(vertices_leftarm, vertices_wo_leftarm)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_leftarm = torch.gather(vertices_wo_leftarm, dim=1, index=expanded_min_indexes) - vertices_leftarm
    refer_normals_leftarm = torch.gather(normals_wo_leftarm, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_leftarm = torch.sum(refer_vector_leftarm * refer_normals_leftarm, dim=-1)
    pene_num += (sdf_vector_multiply_leftarm > 0).sum()
    vert_num += vertices_leftarm.shape[0] * vertices_leftarm.shape[1]

    # rightarm
    _, min_query = ChamDist(vertices_rightarm, vertices_wo_rightarm)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_rightarm = torch.gather(vertices_wo_rightarm, dim=1, index=expanded_min_indexes) - vertices_rightarm
    refer_normals_rightarm = torch.gather(normals_wo_rightarm, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_rightarm = torch.sum(refer_vector_rightarm * refer_normals_rightarm, dim=-1)
    pene_num += (sdf_vector_multiply_rightarm > 0).sum()
    vert_num += vertices_rightarm.shape[0] * vertices_rightarm.shape[1]
    # leftleg
    _, min_query = ChamDist(vertices_leftleg, vertices_wo_leftleg)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_leftleg = torch.gather(vertices_wo_leftleg, dim=1, index=expanded_min_indexes) - vertices_leftleg
    refer_normals_leftleg = torch.gather(normals_wo_leftleg, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_leftleg = torch.sum(refer_vector_leftleg * refer_normals_leftleg, dim=-1)
    pene_num += (sdf_vector_multiply_leftleg > 0).sum()
    vert_num += vertices_leftleg.shape[0] * vertices_leftleg.shape[1]
    # rightleg
    _, min_query = ChamDist(vertices_rightleg, vertices_wo_rightleg)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_rightleg = torch.gather(vertices_wo_rightleg, dim=1, index=expanded_min_indexes) - vertices_rightleg
    refer_normals_rightleg = torch.gather(normals_wo_rightleg, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_rightleg = torch.sum(refer_vector_rightleg * refer_normals_rightleg, dim=-1)
    pene_num += (sdf_vector_multiply_rightleg > 0).sum()
    vert_num += vertices_rightleg.shape[0] * vertices_rightleg.shape[1]

    return vert_num, pene_num


def penetrate_2(skelB, parents, nameB, quatB_rt, full_mesh_info):
    '''
    calculate the penetration rate based on mesh with the small hole filled;
    the mesh completion is complicated, and the result is basically proportional to penetrate_1;
    '''
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
    _, min_query = ChamDist(vertices_leftarm, vertices_wo_leftarm)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_leftarm = torch.gather(vertices_wo_leftarm, dim=1, index=expanded_min_indexes) - vertices_leftarm
    refer_normals_leftarm = torch.gather(normals_wo_leftarm, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_leftarm = torch.sum(refer_vector_leftarm * refer_normals_leftarm, dim=-1)
    pene_num += (sdf_vector_multiply_leftarm > 0).sum()
    vert_num += vertices_leftarm.shape[0] * vertices_leftarm.shape[1]

    # rightarm
    _, min_query = ChamDist(vertices_rightarm, vertices_wo_rightarm)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_rightarm = torch.gather(vertices_wo_rightarm, dim=1, index=expanded_min_indexes) - vertices_rightarm
    refer_normals_rightarm = torch.gather(normals_wo_rightarm, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_rightarm = torch.sum(refer_vector_rightarm * refer_normals_rightarm, dim=-1)
    pene_num += (sdf_vector_multiply_rightarm > 0).sum()
    vert_num += vertices_rightarm.shape[0] * vertices_rightarm.shape[1]
    # leftleg
    _, min_query = ChamDist(vertices_leftleg, vertices_wo_leftleg)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_leftleg = torch.gather(vertices_wo_leftleg, dim=1, index=expanded_min_indexes) - vertices_leftleg
    refer_normals_leftleg = torch.gather(normals_wo_leftleg, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_leftleg = torch.sum(refer_vector_leftleg * refer_normals_leftleg, dim=-1)
    pene_num += (sdf_vector_multiply_leftleg > 0).sum()
    vert_num += vertices_leftleg.shape[0] * vertices_leftleg.shape[1]
    # rightleg
    _, min_query = ChamDist(vertices_rightleg, vertices_wo_rightleg)
    min_query = min_query.long()
    expanded_min_indexes = min_query.unsqueeze(-1).expand(-1, -1, 3)
    refer_vector_rightleg = torch.gather(vertices_wo_rightleg, dim=1, index=expanded_min_indexes) - vertices_rightleg
    refer_normals_rightleg = torch.gather(normals_wo_rightleg, dim=1, index=expanded_min_indexes)
    sdf_vector_multiply_rightleg = torch.sum(refer_vector_rightleg * refer_normals_rightleg, dim=-1)
    pene_num += (sdf_vector_multiply_rightleg > 0).sum()
    vert_num += vertices_rightleg.shape[0] * vertices_rightleg.shape[1]

    return vert_num, pene_num


def curvature_1(parents, skel, quat):
    '''
    Modified version of curvature, use accumulated acceleration instead, might be more suitable for motion;
    force ~ acceleration;
    '''
    # xyz: meter; v: m/s; a: m2/s
    T, K = quat.shape[0], quat.shape[1]
    joint_path = FK.run(parents, skel, quat) / 100.
    # dx = joint_path[1:] - joint_path[:-1]
    v = (joint_path[1:] - joint_path[:-1]) * 60
    dv = v[1:] - v[:-1]
    a = torch.norm(dv * 60, dim=-1).mean(dim=0)
    return a[None]


def curvature_2(parents, skel, quat):
    '''
    Original version of curvature;
    '''
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
