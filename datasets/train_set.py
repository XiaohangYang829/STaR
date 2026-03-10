import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from os import listdir, makedirs
from os.path import exists, join, splitext

from method.utils import get_height_from_skel
from method.forward_kinematics import FK_NP
from method.utils import put_in_world
from datasets.utils.quaternion import quaternion_to_cont6d_np

# Mixamo skeleton parent indices (22 joints)
_MIXAMO_PARENTS = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])
_N_JOINTS = 22


def visualize_mixamo_mesh(npz_path: str, save_path: str = './mixamo_mesh.png', debug: bool = False):
    """
    Visualise a Mixamo character mesh loaded from a shape .npz file.

    Produces a 2-panel figure:
      Left  — mesh rendered as a translucent trisurf with skeleton overlay.
      Right — vertex colour map where each vertex is coloured by its dominant joint.

    Args:
        npz_path  : path to a shape .npz file (e.g. datasets/mixamo/shape/Mutant.npz)
        save_path : where to save the PNG (only used when debug=False)
        debug     : if True, display interactively instead of saving
    """
    import matplotlib
    if debug:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D        # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm

    data = np.load(npz_path, allow_pickle=True)
    v       = data['rest_vertices'].astype(np.float32)    # (V, 3)
    f       = data['rest_faces'].astype(np.int64)         # (F, 3)
    offsets = data['skeleton'].astype(np.float32)         # (22, 3) — bone offsets from parent
    w       = data['skinning_weights'].astype(np.float32) # (V, 22)
    dominant = np.argmax(w, axis=1)                       # (V,)

    # Convert bone offsets → absolute joint positions by accumulating along parent chain
    parents = _MIXAMO_PARENTS
    J = np.zeros_like(offsets)
    for j in range(_N_JOINTS):
        p = parents[j]
        J[j] = offsets[j] if p < 0 else J[p] + offsets[j]

    # Shift both mesh and skeleton so root joint (J[0]) is at origin
    root_pos = J[0].copy()
    J = J - root_pos

    v_n = v / 100.0               # normalised vertices
    J_n = J / 100.0               # normalised joint positions

    cmap = cm.get_cmap('tab20', _N_JOINTS)
    j_colors = [cmap(j) for j in range(_N_JOINTS)]

    fig = plt.figure(figsize=(16, 8))
    char_name = npz_path.rsplit('/', 1)[-1].replace('.npz', '')
    fig.suptitle(f'Mixamo character: {char_name}', fontsize=12)

    # ── Left: mesh + joints ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection='3d')
    step = max(1, len(f) // 2000)
    tris_xzy = v_n[f[::step]][:, :, [0, 2, 1]]     # X, Z(depth), Y(up)
    poly = Poly3DCollection(tris_xzy, alpha=0.10, edgecolor='none', facecolor='steelblue')
    ax1.add_collection3d(poly)

    for j in range(_N_JOINTS):
        p = parents[j]
        if p >= 0:
            ax1.plot([J_n[p, 0], J_n[j, 0]], [J_n[p, 2], J_n[j, 2]], [J_n[p, 1], J_n[j, 1]],
                     'k-', linewidth=1.5)
    ax1.scatter(J_n[:, 0], J_n[:, 2], J_n[:, 1],
                c=[j_colors[j] for j in range(_N_JOINTS)], s=60, zorder=5)
    for j in range(_N_JOINTS):
        ax1.text(J_n[j, 0], J_n[j, 2], J_n[j, 1], str(j), fontsize=6)

    ax1.view_init(elev=10, azim=-80)
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)
    ax1.set_xlabel('X'); ax1.set_ylabel('Z'); ax1.set_zlabel('Y (up)')
    ax1.set_title(f'T-pose mesh + skeleton (22 joints)')

    # ── Right: dominant-joint vertex colouring ─────────────────────────────────
    ax2 = fig.add_subplot(122, projection='3d')
    vc = np.array([j_colors[d] for d in dominant])
    ax2.scatter(v_n[::4, 0], v_n[::4, 2], v_n[::4, 1],
                c=vc[::4], s=1, alpha=0.5)
    ax2.view_init(elev=10, azim=-80)
    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
    ax2.set_xlabel('X'); ax2.set_ylabel('Z'); ax2.set_zlabel('Y (up)')
    ax2.set_title('Dominant joint per vertex')

    plt.tight_layout()
    if debug:
        plt.show()
    else:
        makedirs(save_path.rsplit('/', 1)[0] if '/' in save_path else '.', exist_ok=True)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f'[Mixamo visualisation] saved to {save_path}')


class Feeder(Dataset):
    def __init__(self, source_path, train_data_path, q_path, stats_path, shape_path, max_length, is_val=False, use_rot6d=False):
        self.source_path = source_path
        self.train_data_path = train_data_path
        self.q_path = q_path
        self.stats_path = stats_path
        self.max_length = max_length
        self.parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])
        self.val_ratio = 0.1
        self.is_val = is_val
        self.use_rot6d = use_rot6d

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
        # Derive separate cache paths for quat and rot6d so both can coexist
        if self.train_data_path is not None and self.use_rot6d:
            base, ext = splitext(self.train_data_path)
            cache_path = base + '_rot6d' + ext  # e.g. train_data_rot6d.npy
        else:
            cache_path = self.train_data_path

        if cache_path is not None and exists(cache_path):
            train_data = np.load(cache_path, allow_pickle=True).item()
            self.train_local_norm = train_data['train_local_norm']
            self.train_global = train_data['train_global']
            self.train_skel_norm = train_data['train_skel_norm']
            self.all_quats_norm = train_data['all_quats_norm']
            self.local_mean = train_data['local_mean']
            self.local_std = train_data['local_std']
            self.quat_mean = train_data['quat_mean']
            self.quat_std = train_data['quat_std']
            self.global_mean = train_data['global_mean']
            self.global_std = train_data['global_std']
            self.all_names = train_data['all_names']
            self.seq_names = train_data['seq_names']
            self.seq_length = train_data['seq_length']
            self.num_joint = train_data['num_joint']
            self.T = train_data['T']
            self.character_mean_dict = train_data['character_mean_dict']
            self.character_std_dict = train_data['character_std_dict']
            self.character_keys = self.character_mean_dict.keys()
            print("Loaded train data from cache: {}".format(cache_path))
            return

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

        # Convert to rot6d if requested
        if self.use_rot6d:
            all_rots = [quaternion_to_cont6d_np(q) for q in all_quats]  # each (T, J, 6)
        else:
            all_rots = all_quats  # each (T, J, 4)

        rot_type = 'rot6d' if self.use_rot6d else 'quat'
        allframes_rot = np.concatenate(all_rots)
        quat_mean = allframes_rot.mean(axis=0)[None, :]  # 1 J (4 or 6)
        quat_std = allframes_rot.std(axis=0)[None, :]
        quat_std[quat_std == 0] = 1

        '''Calculate the mean and std for each character'''
        mean_dict = {}
        std_dict = {}
        for i in range(ntotal_sequences):
            if all_names[i] not in mean_dict:
                mean_dict[all_names[i]] = []
                std_dict[all_names[i]] = []
            mean_dict[all_names[i]].append(all_rots[i].mean(axis=0)[None, :])
            std_dict[all_names[i]].append(all_rots[i].std(axis=0)[None, :])
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
        np.save(join(self.stats_path, f"mixamo_{rot_type}_local_motion_mean.npy"), local_mean)
        np.save(join(self.stats_path, f"mixamo_{rot_type}_local_motion_std.npy"), local_std)
        np.save(join(self.stats_path, f"mixamo_{rot_type}_global_motion_mean.npy"), global_mean)
        np.save(join(self.stats_path, f"mixamo_{rot_type}_global_motion_std.npy"), global_std)
        np.save(join(self.stats_path, f"mixamo_{rot_type}_shape_mean_xyz.npy"), self.shape_mean)
        np.save(join(self.stats_path, f"mixamo_{rot_type}_shape_std_xyz.npy"), self.shape_std)
        np.save(join(self.stats_path, f"mixamo_{rot_type}_mean.npy"), quat_mean)
        np.save(join(self.stats_path, f"mixamo_{rot_type}_std.npy"), quat_std)

        '''Normalize the data'''
        self.num_joint = all_local[0].shape[-2]
        self.T = all_local[0].shape[0]
        local_std[local_std == 0] = 1

        train_local_norm = train_local.copy()
        train_skel_norm = train_skel.copy()
        all_quats_norm = all_rots.copy()
        for i in range(len(train_local)):
            train_local_norm[i] = (train_local[i] - local_mean) / local_std
            train_global[i] = train_global[i]
            train_skel_norm[i] = (train_skel[i] - local_mean) / local_std
            all_quats_norm[i] = (all_rots[i] - quat_mean) / quat_std

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

        if cache_path is not None:
            train_data = dict(
                train_local_norm=train_local_norm,
                train_global=train_global,
                train_skel_norm=train_skel_norm,
                all_quats_norm=all_quats_norm,
                local_mean=local_mean,
                local_std=local_std,
                quat_mean=quat_mean,
                quat_std=quat_std,
                global_mean=global_mean,
                global_std=global_std,
                all_names=all_names,
                seq_names=seq_names,
                seq_length=seq_length,
                num_joint=self.num_joint,
                T=self.T,
                character_mean_dict=character_mean_dict,
                character_std_dict=character_std_dict,
                use_rot6d=self.use_rot6d,
            )
            np.save(cache_path, train_data)
            print("Saved train data cache to: {}".format(cache_path))

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

        temporal_mask = np.zeros((max_len,), dtype=np.float32)
        heightA = np.zeros((1,), dtype=np.float32)
        heightB = np.zeros((1,), dtype=np.float32)

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
            rot_dim = 6 if self.use_rot6d else 4
            pad = np.zeros((max_len - cropped_quatA_norm.shape[0], n_joints, rot_dim))
            if not self.use_rot6d:
                pad[:, :, 0] = 1.0  # identity quaternion (1,0,0,0)
            cropped_quatA_norm = np.concatenate((cropped_quatA_norm, pad))

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

        if False:
            # quatA should be unit quaternions (raw, NOT normalized) — shape (T, J, 4)
            from datasets.utils.quaternion import cont6d_to_quaternion
            quatA = quatA_norm * self.quat_std + self.quat_mean
            quatA_6d = quaternion_to_cont6d_np(quatA)                          # (T, J, 6) numpy
            quatA_rt = cont6d_to_quaternion(torch.from_numpy(quatA_6d).float()) # (T, J, 4) torch
            quatA_rt = quatA_rt.numpy()
            # handle double-cover: q and -q are the same rotation
            signs = np.sign((quatA * quatA_rt).sum(axis=-1, keepdims=True))
            quatA_rt_aligned = quatA_rt * signs
            err = np.abs(quatA - quatA_rt_aligned).max()
            print(f"Round-trip max error: {err:.2e}")  # should be ~1e-6 or less
            assert err < 1e-4, f"Round-trip error too large: {err}"

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


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True, help='path to a shape .npz file, e.g. datasets/mixamo/shape/Mutant.npz')
    p.add_argument('--save', default='./mixamo_mesh.png')
    p.add_argument('--debug', action='store_true', help='show interactively instead of saving')
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    print(f'Keys: {list(data.keys())}')
    print(f'Vertices: {data["rest_vertices"].shape}  Faces: {data["rest_faces"].shape}  '
          f'Weights: {data["skinning_weights"].shape}  Skeleton: {data["skeleton"].shape}')
    visualize_mixamo_mesh(args.npz, save_path=args.save, debug=args.debug)
