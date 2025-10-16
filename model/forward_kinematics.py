from tkinter import N
from tracemalloc import Statistic
import torch
import numpy as np


class FK_NP(object):
    def __init__(self):
        pass

    @staticmethod
    def transforms_multiply(t0s, t1s):
        return np.matmul(t0s, t1s)

    @staticmethod
    def transforms_blank(rotations):
        diagonal = np.diag(np.array([1.0, 1.0, 1.0, 1.0]))[None, None, :, :]
        ts = np.tile(diagonal, (int(rotations.shape[0]), int(rotations.shape[1]), 1, 1))
        return ts

    @staticmethod
    def transforms_rotations(rotations):
        q_length = np.sqrt(np.sum(np.square(rotations), axis=-1))
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
        dim0 = np.stack([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)
        dim1 = np.stack([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)
        dim2 = np.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)
        m = np.stack([dim0, dim1, dim2], axis=-2)

        return m

    @staticmethod
    def transforms_local(positions, rotations):
        transforms = FK_NP.transforms_rotations(rotations)  # bs num_joint 3 3
        transforms = np.concatenate([transforms, positions[:, :, :, None]], axis=-1)  # bs num_joint 3 4
        zeros = np.zeros([int(transforms.shape[0]), int(transforms.shape[1]), 1, 3])
        ones = np.ones([int(transforms.shape[0]), int(transforms.shape[1]), 1, 1])
        zerosones = np.concatenate([zeros, ones], axis=-1)
        transforms = np.concatenate([transforms, zerosones], axis=-2)  # bs num_joint 4 4

        return transforms

    @staticmethod
    def transforms_global(parents, positions, rotations):
        locals = FK_NP.transforms_local(positions, rotations)  # bs num_joint 4 4
        globals = FK_NP.transforms_blank(rotations)  # bs num_joint 4 4
        globals = np.concatenate([locals[:, 0:1], globals[:, 1:]], axis=1)
        # globals = np.split(globals, int(globals.shape[1]), dim=1)
        globals = np.split(globals, 22, axis=1)
        globals = list(globals)
        for i in range(1, positions.shape[1]):
            globals[i] = FK_NP.transforms_multiply(globals[parents[i]][:, 0], locals[:, i])[:, None, :, :]

        return np.concatenate(globals, axis=1)

    @staticmethod
    def run(parents, positions, rotations):
        # positions: bs num_joint 3     rotations: bs numjoint 4
        positions = FK_NP.transforms_global(parents, positions, rotations)[:, :, :, 3]

        return positions[:, :, :3] / positions[:, :, 3, None]


class FK(object):
    def __init__(self):
        pass

    @staticmethod
    def transforms_multiply(t0s, t1s):

        return torch.matmul(t0s, t1s)

    @staticmethod
    def transforms_blank(rotations):
        diagonal = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=rotations.dtype))[
            None, None, :, :
        ].cuda(rotations.device)
        ts = torch.tile(
            diagonal, [int(rotations.shape[0]), int(rotations.shape[1]), 1, 1]
        )

        return ts

    @staticmethod
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

    @staticmethod
    def transforms_local(positions, rotations):
        transforms = FK.transforms_rotations(rotations)  # bs num_joint 3 3
        transforms = torch.cat([transforms, positions[:, :, :, None]], dim=-1)  # bs num_joint 3 4
        zeros = torch.zeros([int(transforms.shape[0]), int(transforms.shape[1]), 1, 3], dtype=positions.dtype).cuda(positions.device)
        ones = torch.ones([int(transforms.shape[0]), int(transforms.shape[1]), 1, 1], dtype=positions.dtype).cuda(positions.device)
        zerosones = torch.cat([zeros, ones], dim=-1)
        transforms = torch.cat([transforms, zerosones], dim=-2)  # bs num_joint 4 4

        return transforms

    @staticmethod
    def transforms_global(parents, positions, rotations):
        locals = FK.transforms_local(positions, rotations)  # bs num_joint 4 4
        globals = FK.transforms_blank(rotations)  # bs num_joint 4 4

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1)
        # globals = torch.split(globals, int(globals.shape[1]), dim=1)
        globals = torch.split(globals, 1, dim=1)
        globals = list(globals)

        for i in range(1, positions.shape[1]):
            globals[i] = FK.transforms_multiply(
                globals[parents[i]][:, 0], locals[:, i]
            )[:, None, :, :]

        return torch.cat(globals, dim=1)

    @staticmethod
    def run(parents, positions, rotations):
        # positions: bs num_joint 3     rotations: bs numjoint 4
        positions = FK.transforms_global(parents, positions, rotations)[:, :, :, 3]

        return positions[:, :, :3] / positions[:, :, 3, None]

    @staticmethod
    def run_w_vec(parents, positions, vectors, rotations):

        locals = FK.transforms_local(positions, rotations)  # bs num_joint 4 4
        globals = FK.transforms_blank(rotations)  # bs num_joint 4 4

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1)
        # globals = torch.split(globals, int(globals.shape[1]), dim=1)
        globals = torch.split(globals, 1, dim=1)
        globals = list(globals)

        K = positions.shape[1]

        for i in range(1, K):
            globals[i] = FK.transforms_multiply(globals[parents[i]][:, 0], locals[:, i])[:, None, :, :]

        positions = torch.cat(globals, dim=1)[:, :, :, 3]

        ##### Warp vectors, only rotation, no translation
        # homo_vectors = torch.cat([vectors, torch.ones_like(vectors[:, :, 0, None])], dim=-1)
        # warped_vectors = [globals[i] @ homo_vectors[:, i][:, None, :, None] for i in range(K)]
        warped_vectors = [globals[i][:, :, :3, :3] @ vectors[:, i][:, None, :, None] for i in range(K)]
        warped_vectors = torch.cat(warped_vectors, dim=1).squeeze()

        return positions[:, :, :3] / positions[:, :, 3, None], warped_vectors


class FK6D(object):
    def __init__(self):
        pass

    @staticmethod
    def transforms_multiply(t0s, t1s):

        return torch.matmul(t0s, t1s)

    @staticmethod
    def transforms_blank(rotations):
        diagonal = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))[
            None, None, :, :
        ].cuda(rotations.device)
        ts = torch.tile(
            diagonal, [int(rotations.shape[0]), int(rotations.shape[1]), 1, 1]
        )

        return ts

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def transforms_local(positions, rotations):
        transforms = FK.repr6d2mat(rotations)  # bs num_joint 3 3
        transforms = torch.cat(
            [transforms, positions[:, :, :, None]], dim=-1
        )  # bs num_joint 3 4
        zeros = torch.zeros(
            [int(transforms.shape[0]), int(transforms.shape[1]), 1, 3]
        ).cuda(positions.device)
        ones = torch.ones(
            [int(transforms.shape[0]), int(transforms.shape[1]), 1, 1]
        ).cuda(positions.device)
        zerosones = torch.cat([zeros, ones], dim=-1)
        transforms = torch.cat([transforms, zerosones], dim=-2)  # bs num_joint 4 4

        return transforms

    @staticmethod
    def transforms_global(parents, positions, rotations):
        locals = FK.transforms_local(positions, rotations)  # bs num_joint 4 4
        globals = FK.transforms_blank(rotations)  # bs num_joint 4 4

        globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1)
        # globals = torch.split(globals, int(globals.shape[1]), dim=1)
        globals = torch.split(globals, 1, dim=1)
        globals = list(globals)

        for i in range(1, positions.shape[1]):
            globals[i] = FK.transforms_multiply(
                globals[parents[i]][:, 0], locals[:, i]
            )[:, None, :, :]

        return torch.cat(globals, dim=1)

    @staticmethod
    def run(parents, positions, rotations):
        # positions: bs num_joint 3     rotations: bs numjoint 6
        positions = FK.transforms_global(parents, positions, rotations)[:, :, :, 3]

        return positions[:, :, :3] / positions[:, :, 3, None]


if __name__ == '__main__':
    import numpy as np
    import torch

    parents = torch.from_numpy(
        np.array(
            [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
        )
    ).cuda()
    posi = np.full([1, 22, 3], 0.5)
    rot = np.full([1, 22, 4], 0.2)
    posi = torch.from_numpy(posi).cuda()
    rot = torch.from_numpy(rot).cuda()
    fk = FK()
    fk.run(parents, posi, rot)
