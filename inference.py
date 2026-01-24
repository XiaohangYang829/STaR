import os
import time
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from os import makedirs
from os.path import exists, join
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
from PIL import Image
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import TexturesUV

# Set CUDA architecture based on available GPUs
if torch.cuda.is_available():
    # Get all available GPUs
    num_gpus = torch.cuda.device_count()
    architectures = set()

    for i in range(num_gpus):
        device = torch.cuda.get_device_name(i)
        if 'A100' in device:
            architectures.add("8.0")
        elif '4090' in device:
            architectures.add("8.9")
        elif '2080' in device or '2080 Ti' in device:
            architectures.add("7.5")
        else:
            architectures.add("7.5")  # Default for unrecognized GPUs

    # Combine all architectures with PTX for forward compatibility
    arch_list = "+".join([f"{arch}+PTX" for arch in sorted(architectures)])
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
    print(f"Compiling for CUDA architectures: {arch_list}")

else:
    os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5+PTX"

from method.render_utils import DiffRender, render
from method.forward_kinematics import FK
from method.utils import import_str, init_seed, random_point_select_torch


def get_parser():
    parser = argparse.ArgumentParser(description='motion retargeting')
    parser.add_argument('--config', default='./config/config.yaml', help='path to the configuration file')
    return parser


def render_video(motion_name, nameA, nameB, video, video_path):
    """ Function to render and save a single video. """
    fig, ax = plt.subplots(figsize=(6, 6))
    frame_image = ax.imshow(video[0])  # Initialize with the first frame
    ax.axis('off')  # Hide axes

    def update(frame):
        frame_image.set_array(video[frame])
        return [frame_image]

    ani = FuncAnimation(fig, update, frames=len(video), interval=50)
    output_path = os.path.join(video_path, f"{motion_name}_from_{nameA}_to_{nameB}.mp4")
    ani.save(output_path, writer='ffmpeg', fps=20)
    plt.close(fig)
    print(f"Rendered video: {output_path}")


def inference(
    encoder,
    retarget_net,
    dataloader,
    arg,
    mesh_file_dic,
    rest_vertices_dic,
    rest_faces_dic,
    rest_vertex_normals_dic,
    skinning_weights_dic,
    texture_dic,
    scale_info,
    inference_info,
    render_mode = 'ours',  # 'ours', 'copy', 'source'
    render_direction = 'default', # 'default', 'rotate'
    parallel_render = False,
):
    if render_direction == 'default':
        R, T = look_at_view_transform(dist=250, elev=0, azim=0, at=((0, 10, 0),), device="cuda")
    elif render_direction == 'rotate':
        R, T = look_at_view_transform(dist=250, elev=0, azim=90, at=((0, 10, 0),), device="cuda")
    Render = DiffRender(R, T, image_size=640, sigma=1e-6, device="cuda")

    pbar = tqdm(total=len(dataloader), ncols=140)
    train_stage = 2
    parents = torch.LongTensor([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]).cuda()

    for batch_idx, (
        indexes,
        motion_name,
        seqA_localnorm,
        skelA_norm,
        skelB_norm,
        temporal_mask,
        heightA,
        heightB,
        seqB_localnorm,
        source_name,
        target_name,
        target_char_name,
        inp_joints,
        tgt_joints,
        shapeA,
        shapeB,
        quatA_norm_cp,
        quatB_norm,
        inp_gt,
        tgt_gt,
    ) in enumerate(dataloader):

        bs, T, K = quatA_norm_cp.shape[0], quatA_norm_cp.shape[1], quatA_norm_cp.shape[2]
        seqA_localnorm = seqA_localnorm.float().cuda()
        skelA_norm = skelA_norm.float().cuda()
        seqB_localnorm = seqB_localnorm.float().cuda()
        skelB_norm = skelB_norm.float().cuda()
        temporal_mask = temporal_mask.float().cuda()
        heightA = heightA.float().cuda()
        heightB = heightB.float().cuda()
        quatA_norm_cp = quatA_norm_cp.float().cuda()
        quatB_norm_gt = quatA_norm_cp.float().cuda()
        shapeA = shapeA.float().cuda()
        shapeB = shapeB.float().cuda()

        pbar.set_description("Inference Step %i" % (batch_idx))
        start_time = time.time()

        scale_factor = 0.007
        nameA_lst = source_name
        nameB_lst = target_name
        pointsA = [
            random_point_select_torch(torch.Tensor(rest_vertices_dic[name]).cuda()) - \
            scale_info[name]['centroid'] for name in nameA_lst
        ]
        sampling_pointsA = torch.cat(pointsA, dim=0) * scale_factor
        sampling_pointsA = sampling_pointsA.permute(0, 2, 1)
        pointsB = [
            random_point_select_torch(torch.Tensor(rest_vertices_dic[name]).cuda()) - \
            scale_info[name]['centroid'] for name in nameB_lst
        ]
        sampling_pointsB = torch.cat(pointsB, dim=0) * scale_factor
        sampling_pointsB = sampling_pointsB.permute(0, 2, 1)
        shape_encoding_A = encoder(sampling_pointsA).detach()
        shape_encoding_B = encoder(sampling_pointsB).detach()
        torch.cuda.empty_cache()

        quatB_rt, localB_rt, globalB_rt, quatB_base, localB_base, _, _ = retarget_net(
            nameA_lst,
            nameB_lst,
            shape_encoding_A,
            shape_encoding_B,
            seqA_localnorm,
            skelA_norm,
            skelB_norm,
            shapeA,
            shapeB,
            quatA_norm_cp,
            heightA,
            heightB,
            inference_info,
            None,
            scale_factor,
        )

        # motion discriminator
        bs = localB_rt.shape[0]

        # render source
        geometric_dict = OrderedDict(
            rest_vertices_dict = rest_vertices_dic,
            skinning_weights_dict = skinning_weights_dic,
            rest_vertex_normals_dict = rest_vertex_normals_dic,
            rest_faces_dict = rest_faces_dic,
            texture_dict = texture_dic,
        )

        video_lst = []
        bs, t, k = quatA_norm_cp.shape[0], quatA_norm_cp.shape[1], quatA_norm_cp.shape[2]
        skelA = skelA_norm.view(bs, t, k, -1) * inference_info['local_std'] + inference_info['local_mean']
        skelB = skelB_norm.view(bs, t, k, -1) * inference_info['local_std'] + inference_info['local_mean']
        quatA = quatA_norm_cp * inference_info['quat_std'] + inference_info['quat_mean']    # denorm
        center = None
        for i in range(bs):
            if render_mode == 'ours':
                video, center = render(Render, skelB[i, 0], parents, nameB_lst[i], quatB_rt[i], geometric_dict, center)
            elif render_mode == 'copy':
                video, center = render(Render, skelB[i, 0], parents, nameB_lst[i], quatA[i], geometric_dict, center)
            elif render_mode == 'source':
                video, center = render(Render, skelA[i, 0], parents, nameA_lst[i], quatA[i], geometric_dict, center)
            video_lst.append(video)
            if not parallel_render:
                render_video(motion_name[i], nameA_lst[i], nameB_lst[i], video, arg.video_path + render_mode + '/')

        if parallel_render:
            num_workers = min(mp.cpu_count(), bs) // 2
            with mp.Pool(processes=num_workers) as pool:
                pool.starmap(render_video, [(motion_name[i], nameA_lst[i], nameB_lst[i], video_lst[i], arg.video_path + render_mode + '/') for i in range(len(video_lst))])

        end_time = time.time()
        pbar.set_postfix(time=end_time-start_time)
        pbar.update(1)
    pbar.close()

    return None


def main(arg):

    if not exists(arg.work_dir):
        makedirs(arg.work_dir)
    os.makedirs(arg.video_path, exist_ok=True)

    encoder = import_str(arg.model_path + '.' + arg.encoding_model)(**arg.encoding_model_args).cuda()
    encoder = nn.DataParallel(encoder, device_ids=arg.device)
    assert arg.encoding_model_weights is not None
    encoder.load_state_dict(torch.load(arg.encoding_model_weights))
    encoder.eval()

    retarget_net = import_str(arg.model_path + '.' + arg.model_name)(**arg.model_args).cuda()
    retarget_net = nn.DataParallel(retarget_net, device_ids=arg.device)
    retarget_net.load_state_dict(torch.load(join(arg.work_dir, arg.test_weights)))
    retarget_net.eval()

    inference_feeder = import_str(arg.inference_feeder)(**arg.inference_feeder_args)
    print('Inference Set: {}'.format(len(inference_feeder)))
    inference_info = {
        "global_mean": torch.from_numpy(inference_feeder.global_mean).cuda(arg.device[0]),
        "global_std": torch.from_numpy(inference_feeder.global_std).cuda(arg.device[0]),
        "local_mean": torch.from_numpy(inference_feeder.local_mean).cuda(arg.device[0]),
        "local_std": torch.from_numpy(inference_feeder.local_std).cuda(arg.device[0]),
        "quat_mean": torch.from_numpy(inference_feeder.quat_mean).cuda(arg.device[0]),
        "quat_std": torch.from_numpy(inference_feeder.quat_std).cuda(arg.device[0]),
        "parents": torch.from_numpy(inference_feeder.parents).cuda(arg.device[0]),
        "all_names": inference_feeder.to_name,
    }
    inference_loader = torch.utils.data.DataLoader(dataset=inference_feeder, batch_size=arg.test_batch_size, num_workers=8, shuffle=False)

    mesh_file_dict = {}
    file_names = os.listdir(arg.mesh_path)
    for mesh_name in file_names:
        mesh_file_dict[mesh_name.split('.')[0]] = np.load(os.path.join(arg.mesh_path, mesh_name))
    scale_info = OrderedDict()
    characters = os.listdir(arg.scale_info_path)
    characters.sort()
    for character in characters:
        scale_info_path = join(arg.scale_info_path, character, 'scale_info.npy')
        scale_info[character] = np.load(scale_info_path, allow_pickle=True).item()
        scale_info[character]['scale'] = torch.tensor(scale_info[character]['scale'], dtype=torch.float32).unsqueeze(0).cuda()
        scale_info[character]['centroid'] = torch.Tensor(scale_info[character]['centroid']).cuda()
    print('Scale info loaded')

    mesh_file_dic = {}
    file_names = os.listdir(arg.mesh_path)
    for mesh_name in file_names:
        mesh_file_dic[mesh_name.split('.')[0]] = np.load(join(arg.mesh_path, mesh_name), allow_pickle=True)

    rest_vertices_dic = {}
    rest_faces_dic = {}
    rest_vertex_normals_dic = {}
    skinning_weights_dic = {}
    texture_dic = {}
    uv_coords_dic = {}
    mesh_file_lst = list(mesh_file_dic.keys())
    mesh_file_lst.sort()
    for mesh_name in mesh_file_lst:
        fbx_data = mesh_file_dic[mesh_name]
        rest_vertices_dic[mesh_name] = torch.Tensor(fbx_data['rest_vertices']).cuda()
        rest_faces_dic[mesh_name] = torch.LongTensor(fbx_data['rest_faces']).cuda()
        rest_vertex_normals_dic[mesh_name] = torch.Tensor(fbx_data['rest_vertex_normals']).cuda()
        skinning_weights_dic[mesh_name] = torch.Tensor(fbx_data['skinning_weights']).cuda()
        try:
            uv_coords_dic[mesh_name] = torch.Tensor(fbx_data['uv_coords']).cuda()
            with Image.open(os.path.join(arg.texture_path, "{}_diffuse.png".format(mesh_name))) as image:
                cuda_image = torch.from_numpy(np.asarray(image.convert("RGB")).astype(np.float32) / 255.).cuda()
            texture = TexturesUV(maps=cuda_image[None], faces_uvs=rest_faces_dic[mesh_name][None], verts_uvs=uv_coords_dic[mesh_name][None])
            texture_dic[mesh_name] = texture
        except:
            N = rest_vertices_dic[mesh_name].shape[0]
            color = torch.tensor([[[0.6, 0.6, 0.6]]])
            texture_image = color.expand(256, 256, 3).cuda()
            verts_uvs = torch.rand(N, 2).cuda()
            texture = TexturesUV(maps=texture_image[None], faces_uvs=rest_faces_dic[mesh_name][None], verts_uvs=verts_uvs[None])
            texture_dic[mesh_name] = texture
        print('Mesh of {} loaded'.format(mesh_name))

    with torch.no_grad():
        inference(
            encoder,
            retarget_net,
            inference_loader,
            arg,
            mesh_file_dic,
            rest_vertices_dic,
            rest_faces_dic,
            rest_vertex_normals_dic,
            skinning_weights_dic,
            texture_dic,
            scale_info,
            inference_info,
        )


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    init_seed(arg.seed)
    main(arg)
