import os
import sys
import numpy as np
import trimesh
from os import listdir, makedirs
from os.path import exists, join
from tqdm import tqdm

sys.path.append(".")


def compute_scale_info(npz_path, save_dir):
    """Load rest mesh from npz produced by 5_extract_shape.py and save
    {'scale': max bbox extent, 'centroid': bbox centroid} as scale_info.npy.

    Mirrors scale_to_unit_cube_npz() from ReST's all_get_scale_info.py.
    """
    mesh_npz = np.load(npz_path)
    mesh_kwargs = {
        'vertices': mesh_npz['rest_vertices'],
        'faces': mesh_npz['rest_faces'],
    }
    # Some older npz files (e.g. Castle_Guard, Man, XBot) lack rest_vertex_normals.
    # The bounding box only depends on vertices/faces, so normals are optional.
    if 'rest_vertex_normals' in mesh_npz.files:
        mesh_kwargs['vertex_normals'] = mesh_npz['rest_vertex_normals']
    mesh = trimesh.Trimesh(**mesh_kwargs)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    save_dict = {
        'scale': np.max(mesh.bounding_box.extents),
        'centroid': mesh.bounding_box.centroid,
    }
    makedirs(save_dir, exist_ok=True)
    np.save(join(save_dir, 'scale_info.npy'), save_dict)


if __name__ == '__main__':
    # Pull characters from both train and test shape directories so the
    # resulting scale_info covers all splits.
    shape_dirs = [
        './datasets/mixamo/shape',
        './datasets/mixamo/test_shape',
    ]
    save_root = './datasets/mixamo/scale_info'

    npz_lookup = {}
    for shape_dir in shape_dirs:
        if not exists(shape_dir):
            continue
        for fname in listdir(shape_dir):
            if not fname.endswith('.npz'):
                continue
            character = fname[:-len('.npz')]
            npz_lookup.setdefault(character, join(shape_dir, fname))

    for character, npz_path in tqdm(sorted(npz_lookup.items())):
        compute_scale_info(npz_path, join(save_root, character))
