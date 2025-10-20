import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def transforms_rotations(rotations):
    q_length = torch.sqrt(torch.sum(torch.square(rotations), dim=-1))
    qw = rotations[..., 0] / q_length
    qx = rotations[..., 1] / q_length
    qy = rotations[..., 2] / q_length
    qz = rotations[..., 3] / q_length
    """Unit quaternion based rotation matrix computation"""
    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)
    dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)
    dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)
    m = torch.stack([dim0, dim1, dim2], dim=-2)

    return m


def repr6d2mat(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat


def viz_mesh(vertices, normals, with_normals=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='black', s=1)
    if with_normals:
        ax.quiver(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            normals[:, 0], normals[:, 1], normals[:, 2], 
            length=0.05, color='blue'
        )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # ax.set_box_aspect([1, 1, 1])
    plt.show()


def viz_dynamic_mesh(vertices, normals, with_normals=True):
    T = vertices.shape[0]
    # Initialize the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Initial scatter plot for the first frame
    scatter_plot = ax.scatter(vertices[0, :, 0], vertices[0, :, 1], vertices[0, :, 2], color='black', s=1)

    # Initial quiver plot for the first frame (normals)
    if with_normals:
        quiver_plot = ax.quiver(
            vertices[0, :, 0], vertices[0, :, 1], vertices[0, :, 2], 
            normals[0, :, 0], normals[0, :, 1], normals[0, :, 2], 
            length=0.05, color='blue'
        )

    # Update function for each frame
    def update(frame):
        # Clear current plot
        ax.collections.clear()  # Removes the scatter plot
        ax.quiver(*quiver_plot.remove())  # Removes the quiver plot

        # Update scatter plot with new vertices
        ax.scatter(vertices[frame, :, 0], vertices[frame, :, 1], vertices[frame, :, 2], color='black', s=1)
        
        # Update quiver plot with new normals
        if with_normals:
            ax.quiver(
                vertices[frame, :, 0], vertices[frame, :, 1], vertices[frame, :, 2], 
                normals[frame, :, 0], normals[frame, :, 1], normals[frame, :, 2], 
                length=0.05, color='blue'
            )
        
        return ax

    # Create animation
    ani = FuncAnimation(fig, update, frames=T, interval=50)

    # Display the animation
    plt.show()


def linear_blend_skinning(parents, quat, rest_skel, mesh_vertices, skin_weights, vertices_normals=None):
    '''
    quat: (T, 22, 4)
    bone_position: (T, 22, 3)
    rest_skel: (22, 3)
    mesh_vertices: (vertex_num, 3)    rest pose
    skin_weights: (vertex_num, 22)
    '''
    T, K = quat.shape[0], quat.shape[1]
    device = quat.device
    rot_mat = transforms_rotations(quat)    # (T, K, 3, 3)
    rot_mat = torch.cat([
        rot_mat, rest_skel[None, :, :, None].repeat(T, 1, 1, 1),
    ], dim=-1)  # (T, K, 3, 4)
    rot_mat = torch.cat([
        rot_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(T, K, 1, 1).cuda(device),
    ], dim=-2)  # rot_mat: (T, K, 4, 4)
    rest_pose_mat = torch.cat([
        torch.eye(3, dtype=torch.float).repeat(K, 1, 1).cuda(device), rest_skel.unsqueeze(-1),
    ], dim=-1)  # rest_pose_mat: (K, 3, 4)
    rest_pose_mat = torch.cat([
        rest_pose_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(K, 1, 1).cuda(device),
    ], dim=-2)  # rest_pose_mat: (K, 4, 4)

    bone_matrix_rest = rest_pose_mat.clone()
    bone_matrix_rot = rot_mat.clone()
    bone_matrix_rest_lst = list(torch.split(bone_matrix_rest, 1, dim=0))    # K * (1, 4, 4)
    bone_matrix_rot_lst = list(torch.split(bone_matrix_rot, 1, dim=1))      # K * (T, 1, 4, 4)

    # forward from root to each nodes
    for i in range(1, K):
        bone_matrix_rest_lst[i] = torch.matmul(bone_matrix_rest_lst[parents[i]][:, :, :], bone_matrix_rest_lst[i][:, :, :])
        bone_matrix_rot_lst[i] = torch.matmul(bone_matrix_rot_lst[parents[i]][:, :, :, :], bone_matrix_rot_lst[i][:, :, :, :])
    bone_matrix_rest = torch.cat(bone_matrix_rest_lst, dim=0)   # (K, 4, 4)
    bone_matrix_rot = torch.cat(bone_matrix_rot_lst, dim=1)     # (T, K, 4, 4)
    bone_matrix_rest_inverse = bone_matrix_rest.inverse().repeat(T, 1, 1, 1)
    bone_matrix_world = torch.einsum('tjmn,tjnk->tjmk', bone_matrix_rot, bone_matrix_rest_inverse)

    rest_root_x, rest_root_y, rest_root_z = (rest_skel[0, 0], rest_skel[0, 1], rest_skel[0, 2])
    root_recover = (torch.tensor([rest_root_x, rest_root_y, rest_root_z], dtype=torch.float).unsqueeze(-1).cuda(quat.device))
    root_recover_matrix = torch.cat([
        torch.eye(3, dtype=torch.float).cuda(device),
        root_recover,
    ], dim=-1)  # (3, 4)
    root_recover_matrix = torch.cat([
        root_recover_matrix,
        torch.tensor([0, 0, 0, 1], dtype=torch.float).unsqueeze(0).cuda(device),
    ], dim=0)   # (4, 4)
    root_recover_matrix = root_recover_matrix.repeat(T, K, 1, 1)  # (T, K, 4, 4)
    bone_matrix_world_tran_recover = torch.einsum('tjmn,tjnk->tjmk', bone_matrix_world, root_recover_matrix)
    bone_matrix_world_tran_recover_inv = torch.linalg.inv(bone_matrix_world_tran_recover[:, :, :3, :3])
    bone_matrix_world_tran_recover_inv_transpose = torch.permute(bone_matrix_world_tran_recover_inv, (0, 1, 3, 2))

    # homogeneous coordinates
    N = mesh_vertices.shape[0]
    verts_rest = torch.cat([
        mesh_vertices,
        torch.ones((N, 1), dtype=torch.float).cuda(device),
    ], dim=-1)  # (N, 4)

    verts_lbs = torch.zeros((T, N, 4)).cuda(device)
    normals_lbs = torch.zeros((T, N, 3)).cuda(device)
    # from time0 to timeT
    for i in range(T):
        # from root node to leaf nodes
        for j in range(K):
            weight = skin_weights[:, j].unsqueeze(1)  # (K, 1)
            # point LBS
            tfs = bone_matrix_world_tran_recover[i, j, :, :]  # (4，4)
            verts_lbs[i, :, :] += weight * tfs.matmul(verts_rest.transpose(0, 1)).transpose(0, 1)
            # normal LBS
            if vertices_normals is not None:
                tfs_inv_transpose = bone_matrix_world_tran_recover_inv_transpose[i, j, :, :]  # (3，3)
                normals_lbs[i, :, :] += weight * tfs_inv_transpose.matmul(vertices_normals.transpose(0, 1)).transpose(0, 1)

    verts_lbs = verts_lbs[:, :, :3]
    # norm the normals_lbs
    normals_lbs_normed = normals_lbs / torch.norm(normals_lbs, dim=-1, keepdim=True)

    # For DEBUG: viz and check
    # viz_mesh(torch.div(verts_lbs[-1], 100.).detach().cpu().numpy(), normals_lbs_normed[-1].detach().cpu().numpy())
    # viz_dynamic_mesh(torch.div(verts_lbs, 100.).detach().cpu().numpy(), normals_lbs_normed.detach().cpu().numpy())

    if vertices_normals is None:
        return verts_lbs
    else:
        return verts_lbs, normals_lbs_normed


def batch_linear_blend_skinning_wo_rootquat(parents, quat, rest_skel, mesh_vertices, skin_weights, vertices_normals=None):
    '''
    quat: (bs, T, 22, 4)
    bone_position: (bs, T, 22, 3)
    rest_skel: (bs, 22, 3)
    mesh_vertices: (bs, vertex_num, 3)    rest pose
    skin_weights: (bs, vertex_num, 22)
    normals: (bs, vertex_num, 22)
    '''
    bs, T, K = quat.shape[0], quat.shape[1], quat.shape[2]
    device = quat.device
    # delete root quat
    rest_skel[:, 0] *= 0

    rot_mat = transforms_rotations(quat)    # (bs, T, K, 3, 3)
    rot_mat = torch.cat([
        rot_mat, rest_skel[:, None, :, :, None].repeat(1, T, 1, 1, 1),
    ], dim=-1)  # (bs, T, K, 3, 4)
    rot_mat = torch.cat([
        rot_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(bs, T, K, 1, 1).cuda(device),
    ], dim=-2)  # rot_mat: (bs, T, K, 4, 4)
    rest_pose_mat = torch.cat([
        torch.eye(3, dtype=torch.float).repeat(bs, K, 1, 1).cuda(device), rest_skel.unsqueeze(-1),
    ], dim=-1)  # rest_pose_mat: (bs, K, 3, 4)
    rest_pose_mat = torch.cat([
        rest_pose_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(bs, K, 1, 1).cuda(device),
    ], dim=-2)  # rest_pose_mat: (bs, K, 4, 4)

    bone_matrix_rest = rest_pose_mat.clone()
    bone_matrix_rot = rot_mat.clone()
    bone_matrix_rest_lst = list(torch.split(bone_matrix_rest, 1, dim=1))    # K * (bs, 1, 4, 4)
    bone_matrix_rot_lst = list(torch.split(bone_matrix_rot, 1, dim=2))      # K * (bs, T, 1, 4, 4)

    # forward from root to each nodes
    for i in range(1, K):
        bone_matrix_rest_lst[i] = torch.matmul(bone_matrix_rest_lst[parents[i]][:, :, :, :], bone_matrix_rest_lst[i][:, :, :, :])
        bone_matrix_rot_lst[i] = torch.matmul(bone_matrix_rot_lst[parents[i]][:, :, :, :, :], bone_matrix_rot_lst[i][:, :, :, :, :])
    bone_matrix_rest = torch.cat(bone_matrix_rest_lst, dim=1)   # (bs, K, 4, 4)
    bone_matrix_rot = torch.cat(bone_matrix_rot_lst, dim=2)     # (bs, T, K, 4, 4)
    bone_matrix_rest_inverse = bone_matrix_rest.inverse()[:, None].repeat(1, T, 1, 1, 1)
    bone_matrix_world = torch.einsum('btjmn,btjnk->btjmk', bone_matrix_rot, bone_matrix_rest_inverse)

    rest_root_x, rest_root_y, rest_root_z = (rest_skel[:, 0, 0], rest_skel[:, 0, 1], rest_skel[:, 0, 2])
    root_recover = (torch.cat([rest_root_x[:, None], rest_root_y[:, None], rest_root_z[:, None]], dim=-1).unsqueeze(-1).cuda(quat.device))  # (bs, 3, 1)
    root_recover_matrix = torch.cat([
        torch.eye(3, dtype=torch.float).repeat(bs, 1, 1).cuda(device),
        root_recover,
    ], dim=-1)  # (bs, 3, 4)
    root_recover_matrix = torch.cat([
        root_recover_matrix,
        torch.tensor([0, 0, 0, 1], dtype=torch.float).unsqueeze(0).repeat(bs, 1, 1).cuda(device),
    ], dim=1)   # (bs, 4, 4)
    root_recover_matrix = root_recover_matrix[:, None, None].repeat(1, T, K, 1, 1)  # (bs, T, K, 4, 4)
    bone_matrix_world_tran_recover = torch.einsum('btjmn,btjnk->btjmk', bone_matrix_world, root_recover_matrix)
    bone_matrix_world_tran_recover_inv = torch.linalg.inv(bone_matrix_world_tran_recover[:, :, :, :3, :3])
    bone_matrix_world_tran_recover_inv_transpose = torch.permute(bone_matrix_world_tran_recover_inv, (0, 1, 2, 4, 3))   # (bs, T, K, 3, 3)

    # homogeneous coordinates
    N = mesh_vertices.shape[1]
    verts_rest = torch.cat([
        mesh_vertices,
        torch.ones((bs, N, 1), dtype=torch.float).cuda(device),
    ], dim=-1)  # (bs, N, 4)

    verts_lbs = torch.zeros((bs, T, N, 4)).cuda(device)
    normals_lbs = torch.zeros((bs, T, N, 3)).cuda(device)
    # from time0 to timeT
    for i in range(T):
        # from root node to leaf nodes
        for j in range(K):
            weight = skin_weights[:, :, j].unsqueeze(2)  # (bs, N, 1)
            # point LBS
            tfs = bone_matrix_world_tran_recover[:, i, j, :, :]  # (bs, 4，4)
            verts_lbs[:, i, :, :] += weight * tfs.matmul(verts_rest.transpose(1, 2)).transpose(1, 2)
            # normal LBS
            if vertices_normals is not None:
                tfs_inv_transpose = bone_matrix_world_tran_recover_inv_transpose[:, i, j, :, :]  # (bs, 3，3)
                normals_lbs[:, i, :, :] += weight * tfs_inv_transpose.matmul(vertices_normals.transpose(1, 2)).transpose(1, 2)

    verts_lbs = verts_lbs[:, :, :, :3]
    # norm the normals_lbs
    normals_lbs_normed = normals_lbs / torch.norm(normals_lbs, dim=-1, keepdim=True)

    # For DEBUG: viz and check
    # viz_mesh(torch.div(verts_lbs[-1], 100.).detach().cpu().numpy(), normals_lbs_normed[-1].detach().cpu().numpy())
    # viz_dynamic_mesh(torch.div(verts_lbs, 100.).detach().cpu().numpy(), normals_lbs_normed.detach().cpu().numpy())

    if vertices_normals is None:
        return verts_lbs
    else:
        return verts_lbs, normals_lbs_normed


def linear_blend_skinning_wo_rootquat(parents, quat, rest_skel, mesh_vertices, skin_weights, vertices_normals=None):
    '''
    quat: (T, 22, 4)
    bone_position: (T, 22, 3)
    rest_skel: (22, 3)
    mesh_vertices: (vertex_num, 3)    rest pose
    skin_weights: (vertex_num, 22)
    '''
    T, K = quat.shape[0], quat.shape[1]
    device = quat.device
    # delete root quat
    rest_skel[0] *= 0

    rot_mat = transforms_rotations(quat)    # (T, K, 3, 3)
    rot_mat = torch.cat([
        rot_mat, rest_skel[None, :, :, None].repeat(T, 1, 1, 1),
    ], dim=-1)  # (T, K, 3, 4)
    rot_mat = torch.cat([
        rot_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(T, K, 1, 1).cuda(device),
    ], dim=-2)  # rot_mat: (T, K, 4, 4)
    rest_pose_mat = torch.cat([
        torch.eye(3, dtype=torch.float).repeat(K, 1, 1).cuda(device), rest_skel.unsqueeze(-1),
    ], dim=-1)  # rest_pose_mat: (K, 3, 4)
    rest_pose_mat = torch.cat([
        rest_pose_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(K, 1, 1).cuda(device),
    ], dim=-2)  # rest_pose_mat: (K, 4, 4)

    bone_matrix_rest = rest_pose_mat.clone()
    bone_matrix_rot = rot_mat.clone()
    bone_matrix_rest_lst = list(torch.split(bone_matrix_rest, 1, dim=0))    # K * (1, 4, 4)
    bone_matrix_rot_lst = list(torch.split(bone_matrix_rot, 1, dim=1))      # K * (T, 1, 4, 4)

    # forward from root to each nodes
    for i in range(1, K):
        bone_matrix_rest_lst[i] = torch.matmul(bone_matrix_rest_lst[parents[i]][:, :, :], bone_matrix_rest_lst[i][:, :, :])
        bone_matrix_rot_lst[i] = torch.matmul(bone_matrix_rot_lst[parents[i]][:, :, :, :], bone_matrix_rot_lst[i][:, :, :, :])
    bone_matrix_rest = torch.cat(bone_matrix_rest_lst, dim=0)   # (K, 4, 4)
    bone_matrix_rot = torch.cat(bone_matrix_rot_lst, dim=1)     # (T, K, 4, 4)
    bone_matrix_rest_inverse = bone_matrix_rest.inverse().repeat(T, 1, 1, 1)
    bone_matrix_world = torch.einsum('tjmn,tjnk->tjmk', bone_matrix_rot, bone_matrix_rest_inverse)

    rest_root_x, rest_root_y, rest_root_z = (rest_skel[0, 0], rest_skel[0, 1], rest_skel[0, 2])
    root_recover = (torch.tensor([rest_root_x, rest_root_y, rest_root_z], dtype=torch.float).unsqueeze(-1).cuda(quat.device))
    root_recover_matrix = torch.cat([
        torch.eye(3, dtype=torch.float).cuda(device),
        root_recover,
    ], dim=-1)  # (3, 4)
    root_recover_matrix = torch.cat([
        root_recover_matrix,
        torch.tensor([0, 0, 0, 1], dtype=torch.float).unsqueeze(0).cuda(device),
    ], dim=0)   # (4, 4)
    root_recover_matrix = root_recover_matrix.repeat(T, K, 1, 1)  # (T, K, 4, 4)
    bone_matrix_world_tran_recover = torch.einsum('tjmn,tjnk->tjmk', bone_matrix_world, root_recover_matrix)
    bone_matrix_world_tran_recover_inv = torch.linalg.inv(bone_matrix_world_tran_recover[:, :, :3, :3])
    bone_matrix_world_tran_recover_inv_transpose = torch.permute(bone_matrix_world_tran_recover_inv, (0, 1, 3, 2))

    # homogeneous coordinates
    N = mesh_vertices.shape[0]
    verts_rest = torch.cat([
        mesh_vertices,
        torch.ones((N, 1), dtype=torch.float).cuda(device),
    ], dim=-1)  # (N, 4)

    verts_lbs = torch.zeros((T, N, 4)).cuda(device)
    normals_lbs = torch.zeros((T, N, 3)).cuda(device)
    # from time0 to timeT
    for i in range(T):
        # from root node to leaf nodes
        for j in range(K):
            weight = skin_weights[:, j].unsqueeze(1)  # (K, 1)
            # point LBS
            tfs = bone_matrix_world_tran_recover[i, j, :, :]  # (4，4)
            verts_lbs[i, :, :] += weight * tfs.matmul(verts_rest.transpose(0, 1)).transpose(0, 1)
            # normal LBS
            if vertices_normals is not None:
                tfs_inv_transpose = bone_matrix_world_tran_recover_inv_transpose[i, j, :, :]  # (3，3)
                normals_lbs[i, :, :] += weight * tfs_inv_transpose.matmul(vertices_normals.transpose(0, 1)).transpose(0, 1)

    verts_lbs = verts_lbs[:, :, :3]
    # norm the normals_lbs
    normals_lbs_normed = normals_lbs / torch.norm(normals_lbs, dim=-1, keepdim=True)

    # For DEBUG: viz and check
    # viz_mesh(torch.div(verts_lbs[-1], 100.).detach().cpu().numpy(), normals_lbs_normed[-1].detach().cpu().numpy())
    # viz_dynamic_mesh(torch.div(verts_lbs, 100.).detach().cpu().numpy(), normals_lbs_normed.detach().cpu().numpy())

    if vertices_normals is None:
        return verts_lbs
    else:
        return verts_lbs, normals_lbs_normed


def linear_blend_skinning_old(parents, quat, rest_skel, mesh_vertices, skin_weights):
    '''
    quat: (T, 22, 4)
    bone_position: (T, 22, 3)
    rest_skel: (22, 3)
    mesh_vertices: (vertex_num, 3)    rest pose
    skin_weights: (vertex_num, 22)
    '''
    rot_mat = transforms_rotations(quat)    # (T, 22, 3, 3)
    # rot_mat = torch.cat([rot_mat, torch.zeros((rot_mat.shape[0], rot_mat.shape[1], rot_mat.shape[2], 1), dtype=torch.float).cuda(quat.device)], dim=-1)
    rot_mat = torch.cat([rot_mat, rest_skel[None, :, :, None].repeat(quat.shape[0], 1, 1, 1)], dim=-1)  # (T, 22, 3, 4)
    rot_mat = torch.cat([
        rot_mat,
        torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(rot_mat.shape[0], rot_mat.shape[1], 1, 1).cuda(quat.device),
    ], dim=-2,)                                       # rot_mat: (T, 22, 4, 4)

    rest_pose_mat = torch.cat([
        torch.eye(3, dtype=torch.float).repeat(rest_skel.shape[0], 1, 1).cuda(quat.device),
        rest_skel.unsqueeze(-1),
    ], dim=-1,)
    rest_pose_mat = torch.cat([
        rest_pose_mat,
        torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(rest_pose_mat.shape[0], 1, 1).cuda(quat.device),
    ], dim=-2,)  # rest_pose_mat: (22, 4, 4)

    num_bone = rest_skel.shape[0]
    bone_matrix_rest = rest_pose_mat.clone()
    bone_matrix_rot = rot_mat.clone()

    # ******************************************** fail to compute gradient **************************************************
    # for i in range(1, num_bone):
    #     bone_matrix_rest[i, :, :] = torch.matmul(bone_matrix_rest[parents[i],:,:], bone_matrix_rest[i,:,:])
    #     bone_matrix_rot[:, i, :, :] = torch.matmul(bone_matrix_rot[:,parents[i],:,:], bone_matrix_rot[:,i,:,:])

    bone_matrix_rest_lst = list(torch.split(bone_matrix_rest, 1, dim=0))
    bone_matrix_rot_lst = list(torch.split(bone_matrix_rot, 1, dim=1))

    for i in range(1, num_bone):
        bone_matrix_rest_lst[i] = torch.matmul(
            bone_matrix_rest_lst[parents[i]][:, :, :],
            bone_matrix_rest_lst[i][:, :, :]
        )
        bone_matrix_rot_lst[i] = torch.matmul(
            bone_matrix_rot_lst[parents[i]][:, :, :, :],
            bone_matrix_rot_lst[i][:, :, :, :],
        )

    bone_matrix_rest = torch.cat(bone_matrix_rest_lst, dim=0)
    bone_matrix_rot = torch.cat(bone_matrix_rot_lst, dim=1)

    bone_matrix_rest_inverse = bone_matrix_rest.inverse().repeat(bone_matrix_rot.shape[0], 1, 1, 1)
    bone_matrix_word = torch.einsum('tjmn,tjnk->tjmk', bone_matrix_rot, bone_matrix_rest_inverse)

    rest_root_x, rest_root_y, rest_root_z = (
        rest_skel[0, 0],
        rest_skel[0, 1],
        rest_skel[0, 2],
    )
    root_recover = (
        torch.tensor([rest_root_x, rest_root_y, rest_root_z], dtype=torch.float).unsqueeze(-1).cuda(quat.device)
    )
    root_recover_matrix = torch.cat(
        [torch.eye(3, dtype=torch.float).cuda(quat.device), root_recover], dim=-1
    )
    root_recover_matrix = torch.cat([
        root_recover_matrix,
        torch.tensor([0, 0, 0, 1], dtype=torch.float).unsqueeze(0).cuda(quat.device),
    ], dim=0,)  # (4,4)
    root_recover_matrix = root_recover_matrix.repeat(
        bone_matrix_word.shape[0], bone_matrix_word.shape[1], 1, 1
    )  # (T, 22, 4, 4)

    bone_matrix_word_tran_recover = torch.einsum('tjmn,tjnk->tjmk', bone_matrix_word, root_recover_matrix)

    T = quat.shape[0]
    verts_rest = torch.cat([
        mesh_vertices,
        torch.ones((mesh_vertices.shape[0], 1), dtype=torch.float).cuda(quat.device),
    ], dim=-1,)  # (vertex_num, 4)
    verts_lbs = torch.zeros((T, mesh_vertices.shape[0], 4)).cuda(quat.device)
    for i in range(T):
        for j in range(num_bone):
            tfs = bone_matrix_word_tran_recover[i, j, :, :]  # (4，4)
            weight = skin_weights[:, j].unsqueeze(1)  # (vertex_num, 1)
            verts_lbs[i, :, :] += weight * tfs.matmul(verts_rest.transpose(0, 1)).transpose(0, 1)

    verts_lbs = verts_lbs[:, :, :3]

    return verts_lbs


def lbs_sceneflow(parents, quat, rest_skel, mesh_vertices, skin_weights):
    """
    unused
    """
    verts_lbs = linear_blend_skinning(parents, quat, rest_skel, mesh_vertices, skin_weights)
    
    flow = []

    return flow

