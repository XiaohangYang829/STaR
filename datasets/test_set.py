import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
from collections import OrderedDict
from torch.utils.data import Dataset

sys.path.append("./submodules")
import BVH as BVH
import Animation as Animation

sys.path.append("./method")
from utils import get_height


class Feeder(Dataset):
    def __init__(
        self, test_pairs, test_data_path, stats_path, shape_path, min_steps, max_steps, is_h36m=False
    ):
        with open(test_pairs, "r") as json_file:
            self.test_pairs = json.load(json_file)

        self.test_data_path = test_data_path
        self.stats_path = stats_path
        self.shape_path = shape_path
        self.min_steps = min_steps
        self.max_steps = max_steps

        # save joints list
        self.parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])
        self.leftarm_bone_lst = np.array([15, 16, 17])      # delete 14 to avoid penetration by LBS
        self.wo_leftarm_bone_lst = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21])
        self.rightarm_bone_lst = np.array([19, 20, 21])     # delete 18 to avoid penetration by LBS
        self.wo_rightarm_bone_lst = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        self.leftleg_bone_lst = np.array([7, 8, 9])         # delete 6 to avoid penetration by LBS
        self.wo_leftleg_bone_lst = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        self.rightleg_bone_lst = np.array([11, 12, 13])     # delete 10 to avoid penetration by LBS
        self.wo_rightleg_bone_lst = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21])
        self.body_bone_lst = np.array([0, 1, 2, 3, 5])

        self.load_data()
        self.shape_dic = {}
        file_names = listdir(shape_path)
        self.shape_mean = np.load(join(stats_path, "mixamo_shape_mean_xyz.npy"))
        self.shape_std = np.load(join(stats_path, "mixamo_shape_std_xyz.npy"))
        for shape_name in file_names:
            fbx_file = np.load(join(shape_path, shape_name), allow_pickle=True)
            full_width = fbx_file['full_width'].astype(np.single)
            joint_shape = fbx_file['joint_shape'].astype(np.single)
            shape_vecotr = np.divide(joint_shape, full_width[None, :])
            self.shape_dic[shape_name.split('.')[0]] = shape_vecotr.reshape(-1)

    def viz(self):
        param = self.testlocal_norm[0][0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([0, 200])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter(param[:, 0], param[:, 2], param[:, 1])
        for i in range(1, len(self.parents)):
            ax.plot(param[[i, self.parents[i]], 0], param[[i, self.parents[i]], 2], param[[i, self.parents[i]], 1])
        plt.show()

    def load_data(self):

        test_data = np.load(self.test_data_path, allow_pickle=True).item()
        self.testlocal_norm = test_data['input_local']
        self.testglobal = test_data['input_global']
        self.testoutseq = test_data['target_data']
        self.input_joints = test_data['input_joints']
        self.input_animation = test_data['input_animation']
        self.target_joints = test_data['target_joints']
        self.target_animation = test_data['target_animation']
        self.testskel_norm = test_data['target_skel']
        self.inpskel_norm = test_data['input_skel']
        self.from_name = test_data['from_name']
        self.to_name = test_data['to_name']
        self.from_shape_name = test_data['from_shape_name']
        self.to_shape_name = test_data['to_shape_name']
        self.target_quat_norm = test_data['target_quat']
        self.input_quat_norm = test_data['input_quat']
        self.input_gt = test_data['input_gt']
        self.target_gt = test_data['target_gt']

        self.local_mean = np.load(join(self.stats_path, "mixamo_local_motion_mean.npy"))
        self.local_std = np.load(join(self.stats_path, "mixamo_local_motion_std.npy"))
        self.quat_mean = np.load(join(self.stats_path, "mixamo_quat_mean.npy"))
        self.quat_std = np.load(join(self.stats_path, "mixamo_quat_std.npy"))
        self.global_mean = np.load(join(self.stats_path, "mixamo_global_motion_mean.npy"))
        self.global_std = np.load(join(self.stats_path, "mixamo_global_motion_std.npy"))
        self.local_std[self.local_std == 0] = 1

        for i in range(len(self.testlocal_norm)):
            self.testlocal_norm[i] = (self.testlocal_norm[i] - self.local_mean) / self.local_std
            self.testskel_norm[i] = (self.testskel_norm[i] - self.local_mean) / self.local_std
            self.inpskel_norm[i] = (self.inpskel_norm[i] - self.local_mean) / self.local_std
            self.input_quat_norm[i] = (self.input_quat_norm[i] - self.quat_mean) / self.quat_std
            self.target_quat_norm[i] = (self.target_quat_norm[i] - self.quat_mean) / self.quat_std

    def __len__(self):
        return len(self.testlocal_norm)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        temporal_mask = np.zeros((self.max_steps,), dtype="float32")
        heightA = np.zeros((1,), dtype="float32")
        heightB = np.zeros((1,), dtype="float32")

        localA_norm_batch = self.testlocal_norm[index][: self.max_steps].reshape([self.max_steps, -1])
        globalA_batch = self.testglobal[index][: self.max_steps].reshape([self.max_steps, -1])
        seqA_localnorm = np.concatenate((localA_norm_batch, globalA_batch), axis=-1)
        skelB_norm = self.testskel_norm[index][: self.max_steps].reshape([self.max_steps, -1])
        skelA_norm = self.inpskel_norm[index][: self.max_steps].reshape([self.max_steps, -1])

        step = self.max_steps
        temporal_mask[:step] = 1.0

        local_mean = self.local_mean.reshape((1, 1, -1))
        local_std = self.local_std.reshape((1, 1, -1))

        """ Height ratio """
        # Input sequence (de-normalize)
        inp_skel = seqA_localnorm[0, :-4].copy() * local_std + local_mean
        inp_skel = inp_skel.reshape([22, 3])

        # Ground truth
        gt = self.testoutseq[index][: self.max_steps, :].copy()
        out_skel = gt[0, :-4].reshape([22, 3])

        inp_height = get_height(inp_skel) / 100
        out_height = get_height(out_skel) / 100

        heightA[0] = inp_height
        heightB[0] = out_height

        inp_shape = self.shape_dic[self.from_shape_name[index]]
        tgt_shape = self.shape_dic[self.to_shape_name[index]]

        target_quat_norm = self.target_quat_norm[index][: self.max_steps]
        input_quat_norm = self.input_quat_norm[index][: self.max_steps]

        input_gt = self.input_gt[index][: self.max_steps]
        tgt_gt = self.target_gt[index][: self.max_steps]

        return (
            index,
            seqA_localnorm,
            skelA_norm,
            skelB_norm,
            temporal_mask,
            heightA,
            heightB,
            gt,
            self.from_name[index],
            self.to_name[index],
            self.to_shape_name[index],
            np.array(self.input_joints[index]),
            np.array(self.target_joints[index]),
            inp_shape,
            tgt_shape,
            input_quat_norm,
            target_quat_norm,
            input_gt,
            tgt_gt,
        )
