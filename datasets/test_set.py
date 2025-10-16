import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from os import listdir, makedirs
from os.path import exists, join
from scipy import stats
from collections import OrderedDict
from torch.utils.data import Dataset

sys.path.append("./submodules")
import BVH as BVH
import Animation as Animation
from Pivots import Pivots
from Quaternions import Quaternions

sys.path.append("./src")
from utils import softmin, get_height
from ops import qlinear, q_mul_q, q_div_q


class Feeder(Dataset):
    def __init__(
        self, test_pairs, source_path, q_path, stats_path, shape_path, min_steps, max_steps, is_h36m=False
    ):
        with open(test_pairs, "r") as json_file:
            self.test_pairs = json.load(json_file)

        # store base config
        self.source_path = source_path
        self.q_path = q_path
        self.stats_path = stats_path
        self.shape_path = shape_path
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.is_h36m = is_h36m

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
        input_local = []
        input_global = []
        input_joints = []
        input_animation = []
        input_skel = []
        input_quat = []

        target_data = []
        target_joints = []
        target_animation = []
        target_skel = []
        target_quat = []
        animation_gt = []

        from_name = []
        to_name = []
        from_shape_name = []
        to_shape_name = []

        input_gt = []
        target_gt = []
        inp_bvh_path = []
        tgt_bvh_path = []
        tgt_shape_path = []

        test_types = []

        joints_list = [
            "Spine",
            "Spine1",
            "Spine2",
            "Neck",
            "Head",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "LeftToeBase",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
            "RightToeBase",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
        ]

        test_dic = {
            'km_kc': self.test_pairs['test_pairs_kk'],
            'km_nc': self.test_pairs['test_pairs_ku'],
            'nm_kc': self.test_pairs['test_pairs_uk'],
            'nm_nc': self.test_pairs['test_pairs_uu'],
        }

        total_count = 0

        for test_type, pair_list in test_dic.items():
            for dic in pair_list:
                try:
                    # some errors in character names: Aj and AJ
                    inp = dic['source_character'].replace(' ', '_').replace('Aj', 'AJ')
                    tgt = dic['terget_character'].replace(' ', '_').replace('Aj', 'AJ')

                    # prepare filenames
                    inp_fbx_file = join(self.source_path, inp, dic['source_motion_file'])
                    inp_bvh_file = inp_fbx_file.replace('fbx', 'bvh')
                    inp_skel_file = join(self.q_path, inp, dic['source_motion_file']).replace('.fbx', '_skel.npy')
                    inp_seq_file = inp_skel_file.replace('_skel', '_seq')
                    inp_quat_file = inp_skel_file.replace('_skel', '_quat')
                    inp_shape_file = join(self.shape_path, inp + '.npz')
                    
                    tgt_fbx_file = join(self.source_path, tgt, dic['target_motion_file'])
                    tgt_bvh_file = tgt_fbx_file.replace('fbx', 'bvh')
                    tgt_skel_file = join(self.q_path, tgt, dic['target_motion_file']).replace('.fbx', '_skel.npy')
                    tgt_seq_file = tgt_skel_file.replace('_skel', '_seq')
                    tgt_quat_file = tgt_skel_file.replace('_skel', '_quat')
                    tgt_shape_file = join(self.shape_path, tgt + '.npz')

                    # load, quat: (T, 22, 4), seq: (T, 74), skel: (T, 22, 3)
                    inpquat = np.load(inp_quat_file)
                    inseq = np.load(inp_seq_file)
                    inpskel = np.load(inp_skel_file)
                    tgtquat = np.load(tgt_quat_file)
                    tgtseq = np.load(tgt_seq_file)
                    tgtskel = np.load(tgt_skel_file)

                    if inseq.shape[0] < self.min_steps:
                        continue
                    inp_gt = inseq[:, :-4].copy()
                    tgt_gt = tgtseq[:, :-4].copy()
                    inp_gt = torch.from_numpy(inp_gt)[None, :]
                    tgt_gt = torch.from_numpy(tgt_gt)[None, :]

                    if tgtskel.shape[0] >= self.min_steps + 1:
                        if not ("Claire" in inp and "Warrok" in tgt):
                            total_count += 1

                    inpanim, inpname, inpftime = BVH.load(inp_bvh_file)
                    tgtanim, tgtname, tgtftime = BVH.load(tgt_bvh_file)
                    gtanim = tgtanim.copy()

                    # get the joints processed by network
                    ibvh_file = (open(inp_bvh_file).read().split("JOINT"))
                    ibvh_joints = [f.split("\n")[0].split(":")[-1].split(" ")[-1] for f in ibvh_file[1:]]
                    ito_keep = [0]
                    for jname in joints_list:
                        for k in range(len(ibvh_joints)):
                            if jname == ibvh_joints[k][-len(jname) :]:
                                ito_keep.append(k + 1)
                                break
                    tbvh_file = (open(tgt_bvh_file).read().split("JOINT"))
                    tbvh_joints = [f.split("\n")[0].split(":")[-1].split(" ")[-1] for f in tbvh_file[1:]]
                    tto_keep = [0]
                    for jname in joints_list:
                        for k in range(len(tbvh_joints)):
                            if jname == tbvh_joints[k][-len(jname) :]:
                                tto_keep.append(k + 1)
                                break

                    # delete all rotations ???
                    tgtanim.rotations.qs[...] = tgtanim.orients.qs[None]
                    if not self.is_h36m:
                        """Copy joints we don't predict"""
                        cinames = []
                        for jname in inpname:
                            cinames.append(jname.split(":")[-1])

                        ctnames = []
                        for jname in tgtname:
                            ctnames.append(jname.split(":")[-1])

                        for jname in cinames:
                            if jname in ctnames:
                                idxt = ctnames.index(jname)
                                idxi = cinames.index(jname)
                                tgtanim.rotations[:, idxt] = inpanim.rotations[:, idxi].copy()

                        tgtanim.positions[:, 0] = inpanim.positions[:, 0].copy()

                    # Put the skels at the same height as the sequence
                    """Subtract lowers point in first timestep for floor contact"""
                    floor_diff = inseq[0, 1:-8:3].min() - tgtseq[0, 1:-8:3].min()
                    tgtseq[:, 1:-8:3] += floor_diff
                    tgtskel[:, 0, 1] = tgtseq[:, 1].copy()
                    offset = inseq[:, -8:-4]
                    inseq = np.reshape(inseq[:, :-8], [inseq.shape[0], -1, 3])
                    num_samples = inseq.shape[0] // self.max_steps

                    # cut the motion sequence by 120 frmes
                    for s in range(num_samples):
                        input_joints.append(ito_keep)
                        target_joints.append(tto_keep)
                        input_animation.append(OrderedDict(
                            animation = inpanim.copy()[s * self.max_steps : (s + 1) * self.max_steps], 
                            name = inpname, 
                            ftime = inpftime,
                        ))
                        target_animation.append(OrderedDict(
                            animation = tgtanim.copy()[s * self.max_steps : (s + 1) * self.max_steps], 
                            name = tgtname, 
                            ftime = tgtftime,
                        ))
                        animation_gt.append([gtanim.copy()[s * self.max_steps : (s + 1) * self.max_steps], tgtname, tgtftime])
                        input_local.append(inseq[s * self.max_steps : (s + 1) * self.max_steps])
                        input_global.append(offset[s * self.max_steps : (s + 1) * self.max_steps])
                        target_data.append(tgtseq[s * self.max_steps : (s + 1) * self.max_steps, :-4])
                        target_skel.append(tgtskel[s * self.max_steps : (s + 1) * self.max_steps])
                        input_skel.append(inpskel[s * self.max_steps : (s + 1) * self.max_steps])
                        from_name.append(test_type.split('_')[0] + '_' + inp)
                        from_shape_name.append(inp)
                        to_name.append(test_type.split('_')[1] + '_' + tgt)
                        to_shape_name.append(tgt)

                        input_gt.append(inp_gt[0, s * self.max_steps : (s + 1) * self.max_steps])
                        target_gt.append(tgt_gt[0, s * self.max_steps : (s + 1) * self.max_steps])
                        # rest_vertices_list.append(rest_vertices)
                        # skinning_weights_list.append(skinning_weights)
                        inp_bvh_path.append(inp_bvh_file)
                        tgt_bvh_path.append(tgt_bvh_file)
                        tgt_shape_path.append(tgt_shape_file)

                        target_quat.append(tgtquat[s * self.max_steps : (s + 1) * self.max_steps])
                        input_quat.append(inpquat[s * self.max_steps : (s + 1) * self.max_steps])
                        test_types.append(test_type)

                    # cut the motion from the other side
                    if not inseq.shape[0] % self.max_steps == 0:
                        input_joints.append(ito_keep)
                        target_joints.append(tto_keep)
                        input_animation.append(OrderedDict(
                            animation = inpanim.copy()[-self.max_steps:], 
                            name = inpname, 
                            ftime = inpftime,
                        ))
                        target_animation.append([tgtanim.copy()[-self.max_steps:], tgtname, tgtftime])
                        animation_gt.append([gtanim.copy()[-self.max_steps:], tgtname, tgtftime])
                        input_local.append(inseq[-self.max_steps:])
                        input_global.append(offset[-self.max_steps:])
                        target_data.append(tgtseq[-self.max_steps:, :-4])
                        target_skel.append(tgtskel[-self.max_steps:])
                        input_skel.append(inpskel[-self.max_steps:])
                        target_quat.append(tgtquat[-self.max_steps:])
                        input_quat.append(inpquat[-self.max_steps:])
                        from_name.append(test_type.split('_')[0] + '_' + inp)
                        from_shape_name.append(inp)
                        to_name.append(test_type.split('_')[1] + '_' + tgt)
                        to_shape_name.append(tgt)

                        input_gt.append(inp_gt[0, -self.max_steps:])
                        target_gt.append(tgt_gt[0, -self.max_steps:])
                        # rest_vertices_list.append(rest_vertices)
                        # skinning_weights_list.append(skinning_weights)
                        inp_bvh_path.append(inp_bvh_file)
                        tgt_bvh_path.append(tgt_bvh_file)
                        tgt_shape_path.append(tgt_shape_file)

                        test_types.append(test_type)

                # leave out the wrong pairs
                except Exception as e:
                    print(dic, e)

        self.testlocal_norm = input_local.copy()
        self.testglobal = input_global
        self.testoutseq = target_data
        self.input_joints = input_joints
        self.input_animation = input_animation
        self.target_joints = target_joints
        self.target_animation = target_animation
        self.testskel_norm = target_skel.copy()
        self.inpskel_norm = input_skel.copy()
        self.animation_gt = animation_gt
        self.from_name = from_name
        self.to_name = to_name
        self.from_shape_name = from_shape_name
        self.to_shape_name = to_shape_name

        self.target_quat_norm = target_quat.copy()
        self.input_quat_norm = input_quat.copy()

        self.input_gt = input_gt
        self.target_gt = target_gt
        self.inp_bvh_path = inp_bvh_path
        self.tgt_bvh_path = tgt_bvh_path
        self.tgt_shape_path = tgt_shape_path

        self.test_types = test_types

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
