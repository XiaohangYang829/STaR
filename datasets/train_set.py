import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from os import listdir, makedirs
from os.path import exists, join

sys.path.append("./method")
from utils import get_height_from_skel
from forward_kinematics import FK_NP
from utils import put_in_world


class Feeder(Dataset):
    def __init__(self, source_path, q_path, stats_path, shape_path, max_length, is_val=False):
        self.source_path = source_path
        self.q_path = q_path
        self.stats_path = stats_path
        self.max_length = max_length
        self.parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])
        self.val_ratio = 0.1
        self.is_val = is_val

        self.leftarm_bone_lst = np.array([15, 16, 17])      # delete 14 to avoid penetration by LBS
        self.wo_leftarm_bone_lst = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21])
        self.rightarm_bone_lst = np.array([19, 20, 21])     # delete 18 to avoid penetration by LBS
        self.wo_rightarm_bone_lst = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        self.leftleg_bone_lst = np.array([7, 8, 9])         # delete 6 to avoid penetration by LBS
        self.wo_leftleg_bone_lst = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        self.rightleg_bone_lst = np.array([11, 12, 13])     # delete 10 to avoid penetration by LBS
        self.wo_rightleg_bone_lst = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21])
        self.body_bone_lst = np.array([0, 1, 2, 3, 5])
        self.foot_bone_lst = np.array([9, 13])
        self.leftfoot_bone_lst = np.array([9])
        self.rightfoot_bone_lst = np.array([13])

        self.shape_dic = {}
        shape_lst = []
        file_names = listdir(shape_path)
        for shape_name in file_names:
            fbx_file = np.load(join(shape_path, shape_name))
            full_width = fbx_file['full_width'].astype(np.single)
            joint_shape = fbx_file['joint_shape'].astype(np.single)
            shape_vecotr = np.divide(joint_shape, full_width[None, :])
            self.shape_dic[shape_name.split('.')[0]] = shape_vecotr
            shape_lst.append(shape_vecotr[:, :])

        shape_array = np.concatenate(shape_lst, axis=0)
        self.shape_mean = shape_array.mean(axis=0)
        self.shape_std = shape_array.std(axis=0)
        self.load_data()

    def load_data(self):
        all_local = []
        all_global = []
        all_skel = []
        all_names = []
        t_skel = []
        all_quats = []
        seq_names = []
        seq_length = []

        folders = [
            f
            for f in listdir(self.q_path)
            if not f.startswith(".") and not f.endswith("py") and not f.endswith(".npz")
        ]
        for folder_name in folders:
            files = [
                f
                for f in listdir(join(self.q_path, folder_name))
                if not f.startswith(".") and f.endswith("_seq.npy")
            ]
            for cfile in files:
                file_name = cfile[:-8]
                # Skel parameters
                positions = np.load(join(self.q_path, folder_name, file_name + "_skel.npy")).astype(np.float32)

                # After processed (Maybe, last 4 elements are dummy values)
                sequence = np.load(join(self.q_path, folder_name, file_name + "_seq.npy")).astype(np.float32)

                # Processed global positions (#frames, 4)
                offset = sequence[:, -8:-4]

                # Processed local positions (#frames, #joints, 3)
                sequence = np.reshape(sequence[:, :-8], [sequence.shape[0], -1, 3])
                positions[:, 0, :] = sequence[:, 0, :]  # root joint

                all_local.append(sequence)
                all_global.append(offset)
                all_skel.append(positions)
                all_names.append(folder_name)
                seq_names.append(file_name)
                seq_length.append(sequence.shape[0])

                # ground truth quat (#frames, #joints, 4)
                quat = np.load(join(self.q_path, folder_name, file_name + "_quat.npy")).astype(np.float32)
                all_quats.append(quat)

        # Joint positions before processed
        train_skel = all_skel       # N T J 3

        # After processed, relative position
        train_local = all_local     # N T J 3
        train_global = all_global   # N T 4

        # T-pose (real position)
        for tt in train_skel:
            t_skel.append(tt[0:1])

        # Total training samples
        all_frames = np.concatenate(train_local)
        ntotal_samples = all_frames.shape[0]
        ntotal_sequences = len(train_local)
        print("Number of sequences: " + str(ntotal_sequences))

        # ============================= Data Normalize ============================= #
        '''Calculate total mean and std'''
        allframes_n_skel = np.concatenate(train_local + t_skel)     # TODO: why add tpose_skel here?
        local_mean = allframes_n_skel.mean(axis=0)[None, :]
        global_mean = np.concatenate(train_global).mean(axis=0)[None, :]
        local_std = allframes_n_skel.std(axis=0)[None, :]
        global_std = np.concatenate(train_global).std(axis=0)[None, :]

        allframes_quat = np.concatenate(all_quats)
        quat_mean = allframes_quat.mean(axis=0)[None, :]  # 1 J 4
        quat_std = allframes_quat.std(axis=0)[None, :]

        '''Calculate the mean and std for each character'''
        mean_dict = {}
        std_dict = {}
        for i in range(ntotal_sequences):
            if all_names[i] not in mean_dict:
                mean_dict[all_names[i]] = []
                std_dict[all_names[i]] = []
            mean_dict[all_names[i]].append(all_quats[i].mean(axis=0)[None, :])
            std_dict[all_names[i]].append(all_quats[i].std(axis=0)[None, :])
        character_mean_dict = {}
        character_std_dict = {}
        for character in mean_dict.keys():
            character_mean_dict[character] = np.concatenate(mean_dict[character]).mean(axis=0)
            character_std_dict[character] = np.concatenate(std_dict[character]).std(axis=0)
        self.character_keys = character_mean_dict.keys()
        self.character_mean_dict = character_mean_dict
        self.character_std_dict = character_std_dict

        '''Save the data stats'''
        if not exists(self.stats_path):
            makedirs(self.stats_path)
        np.save(join(self.stats_path, "mixamo_local_motion_mean.npy"), local_mean)
        np.save(join(self.stats_path, "mixamo_local_motion_std.npy"), local_std)
        np.save(join(self.stats_path, "mixamo_global_motion_mean.npy"), global_mean)
        np.save(join(self.stats_path, "mixamo_global_motion_std.npy"), global_std)
        np.save(join(self.stats_path, "mixamo_shape_mean_xyz.npy"), self.shape_mean)
        np.save(join(self.stats_path, "mixamo_shape_std_xyz.npy"), self.shape_std)
        np.save(join(self.stats_path, "mixamo_quat_mean.npy"), quat_mean)
        np.save(join(self.stats_path, "mixamo_quat_std.npy"), quat_std)

        '''Normalize the data'''
        self.num_joint = all_local[0].shape[-2]
        self.T = all_local[0].shape[0]
        local_std[local_std == 0] = 1

        train_local_norm = train_local.copy()
        train_skel_norm = train_skel.copy()
        all_quats_norm = all_quats.copy()
        for i in range(len(train_local)):
            train_local_norm[i] = (train_local[i] - local_mean) / local_std
            train_global[i] = train_global[i]
            train_skel_norm[i] = (train_skel[i] - local_mean) / local_std
            all_quats_norm[i] = (all_quats[i] - quat_mean) / quat_std

        self.train_local_norm = train_local_norm
        self.train_global = train_global
        self.train_skel_norm = train_skel_norm
        self.local_mean = local_mean
        self.local_std = local_std
        self.quat_mean = quat_mean
        self.quat_std = quat_std
        self.global_mean = global_mean
        self.global_std = global_std
        self.all_names = all_names
        self.seq_names = seq_names
        self.all_quats_norm = all_quats_norm
        self.seq_length = seq_length

    def __len__(self):
        if self.is_val:
            return len(self.train_skel_norm) - int(len(self.train_skel_norm) * 0.9)
        else:
            return int(len(self.train_skel_norm) * 0.9)

    def __iter__(self):
        return self

    def __getitem__(self, indexA):
        if self.is_val:
            indexA = indexA + int(len(self.train_skel_norm) * 0.9)

        local_norm_i = self.train_local_norm[indexA]
        global_i = self.train_global[indexA]
        skel_norm_i = self.train_skel_norm[indexA]
        quat_norm_i = self.all_quats_norm[indexA]

        n_joints = local_norm_i.shape[1]
        max_len = self.max_length

        temporal_mask = torch.zeros((max_len,), dtype=torch.float32)
        heightA = torch.zeros((1,), dtype=torch.float32)
        heightB = torch.zeros((1,), dtype=torch.float32)

        low = 0
        high = local_norm_i.shape[0] - max_len
        if low >= high:
            stidx = 0
        else:
            stidx = np.random.randint(low=low, high=high)

        # ---------------------------------- Character A ----------------------------------------------------

        cropped_localA_norm = local_norm_i[stidx : (stidx + max_len)]
        temporal_mask[: np.min([max_len, cropped_localA_norm.shape[0]])] = 1.0
        # add zeros if the length is shorter than max length (60)
        if cropped_localA_norm.shape[0] < max_len:
            cropped_localA_norm = np.concatenate((cropped_localA_norm, np.zeros((max_len - cropped_localA_norm.shape[0], n_joints, 3))))
        cropped_globalA = global_i[stidx : (stidx + max_len)]
        if cropped_globalA.shape[0] < max_len:
            cropped_globalA = np.concatenate((cropped_globalA, np.zeros((max_len - cropped_globalA.shape[0], 4))))
        cropped_skelA_norm = skel_norm_i[stidx : (stidx + max_len)]
        if cropped_skelA_norm.shape[0] < max_len:
            cropped_skelA_norm = np.concatenate((cropped_skelA_norm, np.zeros((max_len - cropped_skelA_norm.shape[0], n_joints, 3))))
        cropped_quatA_norm = quat_norm_i[stidx : (stidx + max_len)]
        if cropped_quatA_norm.shape[0] < max_len:
            # zeros for quaternions is (1,0,0,0)
            zeros = np.zeros((max_len - cropped_quatA_norm.shape[0], n_joints, 4))
            zeros[:, :, 0] = 1.0
            cropped_quatA_norm = np.concatenate((cropped_quatA_norm, zeros))

        # ---------------------------------- Character B ----------------------------------------------------
        indexB = np.random.randint(len(self.train_skel_norm))

        cropped_skelB_norm = self.train_skel_norm[indexB][0:max_len]
        if cropped_skelB_norm.shape[0] < max_len:
            cropped_skelB_norm = np.concatenate((cropped_skelB_norm, np.zeros((max_len - cropped_skelB_norm.shape[0], n_joints, 3))))

        joints_A_norm = cropped_skelA_norm[0].copy()
        joints_A_norm = joints_A_norm[None]
        joints_A = (joints_A_norm * self.local_std) + self.local_mean
        height_A = get_height_from_skel(joints_A[0])
        height_A = height_A / 100

        joints_B_norm = cropped_skelB_norm[0].copy()
        joints_B_norm = joints_B_norm[None]
        joints_B = joints_B_norm * self.local_std + self.local_mean
        height_B = get_height_from_skel(joints_B[0])
        height_B = height_B / 100

        if np.random.binomial(1, p=0.5):
            cropped_skelB_norm = cropped_skelA_norm.copy()
            heightA[0] = height_A
            heightB[0] = height_A
            indexB = indexA
        else:
            heightA[0] = height_A
            heightB[0] = height_B

        localA_norm = cropped_localA_norm.reshape((max_len, -1))
        globalA = cropped_globalA.reshape((max_len, -1))
        seqA_localnorm = np.concatenate((localA_norm, globalA), axis=-1).astype(np.float32)
        skelA_norm = cropped_skelA_norm.reshape((max_len, -1)).astype(np.float32)
        quatA_norm = cropped_quatA_norm.astype(np.float32)

        localB_norm = cropped_localA_norm.reshape((max_len, -1))
        globalB = cropped_globalA.reshape((max_len, -1))
        seqB_localnorm = np.concatenate((localB_norm, globalB), axis=-1).astype(np.float32)
        skelB_norm = cropped_skelB_norm.reshape((max_len, -1)).astype(np.float32)

        shapeA = self.shape_dic[self.all_names[indexA]].reshape(-1)
        shapeB = self.shape_dic[self.all_names[indexB]].reshape(-1)

        return (
            indexA,
            indexB,
            seqA_localnorm,
            skelA_norm,
            seqB_localnorm,
            skelB_norm,
            temporal_mask,
            heightA,
            heightB,
            shapeA,
            shapeB,
            quatA_norm,
        )
