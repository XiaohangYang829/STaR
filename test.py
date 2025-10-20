import os
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from os.path import exists, join
from tqdm import tqdm
from collections import OrderedDict

if torch.cuda.is_available():
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
            architectures.add("7.5")
    arch_list = "+".join([f"{arch}+PTX" for arch in sorted(architectures)])
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
    print(f"Compiling for CUDA architectures: {arch_list}")
else:
    os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5+PTX"

from method.utils import import_str, init_seed, put_in_world, get_height, random_point_select_torch
from method.metric import penetrate_1, curvature_1


def get_parser():
    parser = argparse.ArgumentParser(description='motion retargeting')
    parser.add_argument('--config', default='./config/config.yaml', help='path to the configuration file')
    return parser


def test(
    encoder,
    retarget_net,
    dataloader,
    test_info,
    arg,
    full_mesh_info,
    experiment_name,
):

    pbar = tqdm(total=len(dataloader), ncols=140)

    mse_local = []
    mse_local_copy = []
    mse = []
    mse_copy = []
    penetration_mine_rate_gt = []
    penetration_mine_rate_ret = []
    penetration_mine_rate_copy = []
    curvature_source_lst = []
    curvature_gt_lst = []
    curvature_ret_lst = []
    curvature_copy_lst = []

    labels = []

    for batch_idx, (
        indexes,
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

        pbar.set_description("Step %i" % (batch_idx))
        start_time = time.time()

        bs, T = skelA_norm.shape[0], skelA_norm.shape[1]
        seqA_localnorm = seqA_localnorm.float().cuda(arg.device[0])
        skelA_norm = skelA_norm.float().cuda(arg.device[0])
        seqB_localnorm = seqB_localnorm.float().cuda(arg.device[0])
        skelB_norm = skelB_norm.float().cuda(arg.device[0])
        temporal_mask = temporal_mask.float().cuda(arg.device[0])
        heightA = heightA.float().cuda(arg.device[0])
        heightB = heightB.float().cuda(arg.device[0])
        quatA_norm_cp = quatA_norm_cp.float().cuda(arg.device[0])
        quatB_norm = quatB_norm.float().cuda(arg.device[0])
        shapeA = shapeA.float().cuda(arg.device[0])
        shapeB = shapeB.float().cuda(arg.device[0])
        tgt_gt = tgt_gt.cpu().numpy()

        scale_factor = 0.007
        nameA_lst = [name.split('_', 1)[-1] for name in source_name]
        nameB_lst = [name.split('_', 1)[-1] for name in target_name]
        pointsA = [
            random_point_select_torch(torch.Tensor(full_mesh_info['rest_vertices'][name]).cuda(arg.device[0])) - \
            full_mesh_info['scale_info'][name]['centroid'] for name in nameA_lst
        ]
        sampling_pointsA = torch.cat(pointsA, dim=0) * scale_factor
        sampling_pointsA = sampling_pointsA.permute(0, 2, 1)
        pointsB = [
            random_point_select_torch(torch.Tensor(full_mesh_info['rest_vertices'][name]).cuda(arg.device[0])) - \
            full_mesh_info['scale_info'][name]['centroid'] for name in nameB_lst
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
            test_info,
            full_mesh_info,
            scale_factor,
        )

        bs = localB_rt.shape[0]
        T = localB_rt.shape[1]
        ret_total = torch.cat((localB_rt.reshape([bs, T, -1]), globalB_rt), dim=-1)

        copy_total = torch.cat((localB_base.reshape([bs, T, -1]), globalB_rt), dim=-1)
        ret_total_np = ret_total.cpu().numpy()
        copy_total_np = copy_total.cpu().numpy()

        # denorm
        skelA = skelA_norm.view(bs, T, 22, 3) * test_info['local_std'] + test_info['local_mean']
        skelB = skelB_norm.view(bs, T, 22, 3) * test_info['local_std'] + test_info['local_mean']
        quatA = quatA_norm_cp * test_info['quat_std'] + test_info['quat_mean']
        quatB = quatB_norm * test_info['quat_std'] + test_info['quat_mean']

        for i in range(bs):
            nameA = nameA_lst[i]
            nameB = nameB_lst[i]
            labels.append(source_name[i].split('_')[0] + "_" + target_name[i].split('_')[0])

            gt_global, _ = put_in_world(tgt_gt[i].copy())
            gt_local = tgt_gt[i, :, :-4].reshape(T, -1, 3)
            height = get_height(gt_global[0, 0])

            ret_global, _ = put_in_world(ret_total_np[i].copy())
            ret_global_copy, _ = put_in_world(copy_total_np[i].copy())
            ret_local = ret_total_np[i, :, :-4].reshape(T, -1, 3)
            ret_local_copy = copy_total_np[i, :, :-4].reshape(T, -1, 3)

            # according to some previous works, we use this MSE version, actually this metric is not important;
            mse.append(1. / height * np.linalg.norm(ret_global - gt_global, ord=2, axis=3).mean())
            mse_local.append(1. / height * np.linalg.norm(ret_local - gt_local, ord=2, axis=2).mean())
            mse_copy.append(1. / height * np.linalg.norm(ret_global_copy - gt_global, ord=2, axis=3).mean())
            mse_local_copy.append(1. / height * np.linalg.norm(ret_local_copy - gt_local, ord=2, axis=2).mean())

            vert_num, pene_num = penetrate_1(skelB[i, 0], test_info['parents'], nameB, quatB[i], full_mesh_info)
            penetration_mine_rate_gt.append(pene_num / vert_num)
            vert_num, pene_num = penetrate_1(skelB[i, 0], test_info['parents'], nameB, quatB_rt[i], full_mesh_info)
            penetration_mine_rate_ret.append(pene_num / vert_num)
            vert_num, pene_num = penetrate_1(skelB[i, 0], test_info['parents'], nameB, quatB_base[i], full_mesh_info)
            penetration_mine_rate_copy.append(pene_num / vert_num)

            curv = curvature_1(test_info['parents'], skelA[i], quatA[i])
            curvature_source_lst.append(curv)
            curv = curvature_1(test_info['parents'], skelB[i], quatB[i])
            curvature_gt_lst.append(curv)
            curv = curvature_1(test_info['parents'], skelB[i], quatB_rt[i])
            curvature_ret_lst.append(curv)
            curv = curvature_1(test_info['parents'], skelB[i], quatB_base[i])
            curvature_copy_lst.append(curv)

        pbar.set_postfix(time=time.time() - start_time)
        pbar.update(1)
    pbar.close()

    f = open(os.path.join(arg.work_dir, 'eval.txt'), 'w')
    f.write(experiment_name + '\n')
    f.write("MSE.\n")
    f.write("Ret MSE\t\t\t" + "{0:.6f}".format(np.mean(mse)) + "\n")
    f.write("Ret MSE_local\t\t" + "{0:.6f}".format(np.mean(mse_local)) + "\n")
    f.write("Copy MSE\t\t" + "{0:.6f}".format(np.mean(mse_copy)) + "\n")
    f.write("Copy MSE_local\t\t" + "{0:.6f}".format(np.mean(mse_local_copy)) + "\n\n")
    f.write("Penetration.\n")
    f.write("GT Penetration Rate\t\t" + "{0:.6f}".format(sum(penetration_mine_rate_gt) / len(penetration_mine_rate_gt)) + "\n")
    f.write("Ret Penetration Rate\t\t" + "{0:.6f}".format(sum(penetration_mine_rate_ret) / len(penetration_mine_rate_ret)) + "\n")
    f.write("Copy Penetration Rate\t\t" + "{0:.6f}".format(sum(penetration_mine_rate_copy) / len(penetration_mine_rate_copy)) + "\n")
    f.write("Curvature.\n")
    curv_source = torch.cat(curvature_source_lst, dim=0).mean(dim=0)
    f.write("Source Curvature\n\t" + "{}".format(curv_source) + "\n")
    curv_gt = torch.cat(curvature_gt_lst, dim=0).mean(dim=0)
    f.write("GT Curvature\n\t" + "{}".format(curv_gt) + "\n")
    curv_ret = torch.cat(curvature_ret_lst, dim=0).mean(dim=0)
    f.write("Ret Curvature\n\t" + "{}".format(curv_ret) + "\n")
    curv_copy = torch.cat(curvature_copy_lst, dim=0).mean(dim=0)
    f.write("Copy Curvature\n\t" + "{}".format(curv_copy) + "\n")
    f.close()


def get_full_mesh_info(scale_info_path, test_feeder, mesh_file_dic):

    scale_info = OrderedDict()
    characters = os.listdir(scale_info_path)
    characters.sort()
    for character in characters:
        character_scale_info_path = join(scale_info_path, character, 'scale_info.npy')
        scale_info[character] = np.load(character_scale_info_path, allow_pickle=True).item()
        scale_info[character]['scale'] = torch.tensor(scale_info[character]['scale'], dtype=torch.float32).unsqueeze(0).cuda()
        scale_info[character]['centroid'] = torch.Tensor(scale_info[character]['centroid']).cuda()

    vertices_part_dic = {}
    body_vertices_dic = {}
    head_vertices_dic = {}
    leftarm_vertices_dic = {}
    rightarm_vertices_dic = {}
    leftleg_vertices_dic = {}
    rightleg_vertices_dic = {}
    wo_leftarm_vertices_dic = {}
    wo_rightarm_vertices_dic = {}
    wo_leftleg_vertices_dic = {}
    wo_rightleg_vertices_dic = {}
    hands_vertices_dic = {}
    lefthand_vertices_dic = {}
    righthand_vertices_dic = {}
    rest_vertices_dic = {}
    rest_faces_dic = {}
    rest_vertex_normals_dic = {}
    skinning_weights_dic = {}
    new_weights_dic = {}
    vertex_part_dic = {}

    body_bone_lst = test_feeder.body_bone_lst.tolist()[0:4]
    head_bone_lst = test_feeder.body_bone_lst.tolist()[4:]
    leftarm_bone_lst = test_feeder.leftarm_bone_lst.tolist()
    rightarm_bone_lst = test_feeder.rightarm_bone_lst.tolist()
    leftleg_bone_lst = test_feeder.leftleg_bone_lst.tolist()
    rightleg_bone_lst = test_feeder.rightleg_bone_lst.tolist()
    wo_leftarm_bone_lst = test_feeder.wo_leftarm_bone_lst.tolist()
    wo_rightarm_bone_lst = test_feeder.wo_rightarm_bone_lst.tolist()
    wo_leftleg_bone_lst = test_feeder.wo_leftleg_bone_lst.tolist()
    wo_rightleg_bone_lst = test_feeder.wo_rightleg_bone_lst.tolist()

    for mesh_name in mesh_file_dic.keys():
        fbx_data = mesh_file_dic[mesh_name]
        vertex_part_np = fbx_data['vertex_part']
        vertex_num = vertex_part_np.shape[0]
        body_lst = []
        head_lst = []
        leftarm_lst = []
        rightarm_lst = []
        leftleg_lst = []
        rightleg_lst = []
        wo_leftarm_lst = []
        wo_rightarm_lst = []
        wo_leftleg_lst = []
        wo_rightleg_lst = []
        hands_lst = []
        lefthand_lst = []
        righthand_lst = []
        for i in range(vertex_num):
            if vertex_part_np[i] in body_bone_lst:
                body_lst.append(i)
            if vertex_part_np[i] in head_bone_lst:
                head_lst.append(i)
            if vertex_part_np[i] in leftarm_bone_lst:
                leftarm_lst.append(i)
            if vertex_part_np[i] in rightarm_bone_lst:
                rightarm_lst.append(i)
            if vertex_part_np[i] in leftleg_bone_lst:
                leftleg_lst.append(i)
            if vertex_part_np[i] in rightleg_bone_lst:
                rightleg_lst.append(i)

            if vertex_part_np[i] in wo_leftarm_bone_lst:
                wo_leftarm_lst.append(i)
            if vertex_part_np[i] in wo_rightarm_bone_lst:
                wo_rightarm_lst.append(i)
            if vertex_part_np[i] in wo_leftleg_bone_lst:
                wo_leftleg_lst.append(i)
            if vertex_part_np[i] in wo_rightleg_bone_lst:
                wo_rightleg_lst.append(i)

            if vertex_part_np[i] in leftarm_bone_lst[-1:] or vertex_part_np[i] in rightarm_bone_lst[-1:]:
                hands_lst.append(i)
            if vertex_part_np[i] in leftarm_bone_lst[-1:]:
                lefthand_lst.append(i)
            if vertex_part_np[i] in rightarm_bone_lst[-1:]:
                righthand_lst.append(i)

        if len(lefthand_lst) == 0:
            for i in range(vertex_num):
                if vertex_part_np[i] in leftarm_bone_lst[-2:]:
                    lefthand_lst.append(i)
        if len(righthand_lst) == 0:
            for i in range(vertex_num):
                if vertex_part_np[i] in rightarm_bone_lst[-2:]:
                    righthand_lst.append(i)

        vertices_part_dic[mesh_name] = torch.LongTensor(vertex_part_np).cuda()
        body_vertices_dic[mesh_name] = torch.LongTensor(body_lst).cuda()
        head_vertices_dic[mesh_name] = torch.LongTensor(head_lst).cuda()
        leftarm_vertices_dic[mesh_name] = torch.LongTensor(leftarm_lst).cuda()
        rightarm_vertices_dic[mesh_name] = torch.LongTensor(rightarm_lst).cuda()
        leftleg_vertices_dic[mesh_name] = torch.LongTensor(leftleg_lst).cuda()
        rightleg_vertices_dic[mesh_name] = torch.LongTensor(rightleg_lst).cuda()

        wo_leftarm_vertices_dic[mesh_name] = torch.LongTensor(wo_leftarm_lst).cuda()
        wo_rightarm_vertices_dic[mesh_name] = torch.LongTensor(wo_rightarm_lst).cuda()
        wo_leftleg_vertices_dic[mesh_name] = torch.LongTensor(wo_leftleg_lst).cuda()
        wo_rightleg_vertices_dic[mesh_name] = torch.LongTensor(wo_rightleg_lst).cuda()
        
        hands_vertices_dic[mesh_name] = torch.LongTensor(hands_lst).cuda()
        lefthand_vertices_dic[mesh_name] = torch.LongTensor(lefthand_lst).cuda()
        righthand_vertices_dic[mesh_name] = torch.LongTensor(righthand_lst).cuda()

        rest_vertices_dic[mesh_name] = torch.Tensor(fbx_data['rest_vertices']).cuda()
        rest_faces_dic[mesh_name] = torch.Tensor(fbx_data['rest_faces']).cuda()
        rest_vertex_normals_dic[mesh_name] = torch.Tensor(fbx_data['rest_vertex_normals']).cuda()
        skinning_weights_dic[mesh_name] = torch.Tensor(fbx_data['skinning_weights']).cuda()
        vertex_part_dic[mesh_name] = vertex_part_np
        print('Mesh of {} loaded'.format(mesh_name))

    full_mesh_info = OrderedDict(
        scale_info = scale_info,
        vertices_part = vertices_part_dic,
        body_vertices = body_vertices_dic,
        head_vertices = head_vertices_dic,
        leftarm_vertices = leftarm_vertices_dic,
        rightarm_vertices = rightarm_vertices_dic,
        leftleg_vertices = leftleg_vertices_dic,
        rightleg_vertices = rightleg_vertices_dic,
        wo_leftarm_vertices = wo_leftarm_vertices_dic,
        wo_rightarm_vertices = wo_rightarm_vertices_dic,
        wo_leftleg_vertices = wo_leftleg_vertices_dic,
        wo_rightleg_vertices = wo_rightleg_vertices_dic,
        hands_vertices = hands_vertices_dic,
        lefthand_vertices = lefthand_vertices_dic,
        righthand_vertices = righthand_vertices_dic,
        rest_vertices = rest_vertices_dic,
        rest_faces = rest_faces_dic,
        rest_vertex_normals = rest_vertex_normals_dic,
        skinning_weights = skinning_weights_dic,
        new_weights = new_weights_dic,
        vertex_part = vertex_part_dic,
    )

    return full_mesh_info


def resample_mesh_info(full_mesh_info, mesh_file_dic, arg, sample_replace):
    # re-sample vertexes
    resample_lefthand_vertices_dic = {}
    resample_righthand_vertices_dic = {}
    resample_leftarm_vertices_dic = {}
    resample_rightarm_vertices_dic = {}
    resample_leftleg_vertices_dic = {}
    resample_rightleg_vertices_dic = {}
    resample_wo_leftarm_vertices_dic = {}
    resample_wo_rightarm_vertices_dic = {}
    resample_wo_leftleg_vertices_dic = {}
    resample_wo_rightleg_vertices_dic = {}
    for mesh_name in mesh_file_dic.keys():
        hand_num = arg.hand_vert_num
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['lefthand_vertices'][mesh_name].shape[0])), hand_num, replace=sample_replace)).cuda()
        resample_lefthand_vertices_dic[mesh_name] = full_mesh_info['lefthand_vertices'][mesh_name][mask]
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['righthand_vertices'][mesh_name].shape[0])), hand_num, replace=sample_replace)).cuda()
        resample_righthand_vertices_dic[mesh_name] = full_mesh_info['righthand_vertices'][mesh_name][mask]
        limb_num = arg.limb_vert_num
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['leftarm_vertices'][mesh_name].shape[0])), limb_num, replace=sample_replace)).cuda()
        resample_leftarm_vertices_dic[mesh_name] = full_mesh_info['leftarm_vertices'][mesh_name][mask]
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['rightarm_vertices'][mesh_name].shape[0])), limb_num, replace=sample_replace)).cuda()
        resample_rightarm_vertices_dic[mesh_name] = full_mesh_info['rightarm_vertices'][mesh_name][mask]
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['leftleg_vertices'][mesh_name].shape[0])), limb_num, replace=sample_replace)).cuda()
        resample_leftleg_vertices_dic[mesh_name] = full_mesh_info['leftleg_vertices'][mesh_name][mask]
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['rightleg_vertices'][mesh_name].shape[0])), limb_num, replace=sample_replace)).cuda()
        resample_rightleg_vertices_dic[mesh_name] = full_mesh_info['rightleg_vertices'][mesh_name][mask]
        wo_limb_num = arg.wo_limb_vert_num
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['wo_leftarm_vertices'][mesh_name].shape[0])), wo_limb_num, replace=sample_replace)).cuda()
        resample_wo_leftarm_vertices_dic[mesh_name] = full_mesh_info['wo_leftarm_vertices'][mesh_name][mask]
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['wo_rightarm_vertices'][mesh_name].shape[0])), wo_limb_num, replace=sample_replace)).cuda()
        resample_wo_rightarm_vertices_dic[mesh_name] = full_mesh_info['wo_rightarm_vertices'][mesh_name][mask]
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['wo_leftleg_vertices'][mesh_name].shape[0])), wo_limb_num, replace=sample_replace)).cuda()
        resample_wo_leftleg_vertices_dic[mesh_name] = full_mesh_info['wo_leftleg_vertices'][mesh_name][mask]
        mask = torch.LongTensor(np.random.choice(np.array(range(full_mesh_info['wo_rightleg_vertices'][mesh_name].shape[0])), wo_limb_num, replace=sample_replace)).cuda()
        resample_wo_rightleg_vertices_dic[mesh_name] = full_mesh_info['wo_rightleg_vertices'][mesh_name][mask]

    sample_mesh_info = OrderedDict(
        resample_lefthand_vertices = resample_lefthand_vertices_dic,
        resample_righthand_vertices = resample_righthand_vertices_dic,
        resample_leftarm_vertices = resample_leftarm_vertices_dic,
        resample_rightarm_vertices = resample_rightarm_vertices_dic,
        resample_leftleg_vertices = resample_leftleg_vertices_dic,
        resample_rightleg_vertices = resample_rightleg_vertices_dic,
        resample_wo_leftarm_vertices = resample_wo_leftarm_vertices_dic,
        resample_wo_rightarm_vertices = resample_wo_rightarm_vertices_dic,
        resample_wo_leftleg_vertices = resample_wo_leftleg_vertices_dic,
        resample_wo_rightleg_vertices = resample_wo_rightleg_vertices_dic,
    )
    return sample_mesh_info


def main(arg):

    torch.autograd.set_detect_anomaly(True)
    assert arg.work_dir is not None
    assert arg.test_weights is not None
    assert exists(join(arg.work_dir, arg.test_weights))

    test_feeder = import_str(arg.test_set)(**arg.test_set_args)
    print('Test Set: {}'.format(len(test_feeder)))
    test_info = {
        "global_mean": torch.from_numpy(test_feeder.global_mean).cuda(arg.device[0]),
        "global_std": torch.from_numpy(test_feeder.global_std).cuda(arg.device[0]),
        "local_mean": torch.from_numpy(test_feeder.local_mean).cuda(arg.device[0]),
        "local_std": torch.from_numpy(test_feeder.local_std).cuda(arg.device[0]),
        "quat_mean": torch.from_numpy(test_feeder.quat_mean).cuda(arg.device[0]),
        "quat_std": torch.from_numpy(test_feeder.quat_std).cuda(arg.device[0]),
        "parents": torch.from_numpy(test_feeder.parents).cuda(arg.device[0]),
        "all_names": test_feeder.to_name,
    }
    test_loader = torch.utils.data.DataLoader(dataset=test_feeder, batch_size=arg.test_batch_size, num_workers=8, shuffle=False)

    encoder = import_str(arg.model_path + '.' + arg.encoding_model)(**arg.encoding_model_args).cuda(arg.device[0])
    encoder = nn.DataParallel(encoder, device_ids=arg.device)
    assert arg.encoding_model_weights is not None
    encoder.load_state_dict(torch.load(arg.encoding_model_weights))
    encoder.eval()

    retarget_net = import_str(arg.model_path + '.' + arg.model_name)(**arg.model_args).cuda(arg.device[0])
    retarget_net = nn.DataParallel(retarget_net, device_ids=arg.device)
    retarget_net.load_state_dict(torch.load(join(arg.work_dir, arg.test_weights)))
    retarget_net.eval()

    mesh_file_dic = {}
    file_names = os.listdir(arg.mesh_path)
    for mesh_name in file_names:
        mesh_file_dic[mesh_name.split('.')[0]] = np.load(join(arg.mesh_path, mesh_name))
    full_mesh_info = get_full_mesh_info(arg.scale_info_path, test_feeder, mesh_file_dic)

    with torch.no_grad():
        test(
            encoder,
            retarget_net,
            test_loader,
            test_info,
            arg,
            full_mesh_info,
            arg.work_dir.split('/')[-2],
        )


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.device[0])
    init_seed(arg.seed)
    main(arg)
