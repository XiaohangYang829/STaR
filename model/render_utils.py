import torch
import numpy as np
import torch.nn as nn
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    Materials,
    MeshRenderer,
    SoftPhongShader,
    MeshRasterizer,
    BlendParams,
    PointLights,
    RasterizationSettings,
)
from src.linear_blend_skin import linear_blend_skinning
from pytorch3d.structures import Meshes

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class DiffRender(nn.Module):
    """
    Differential Render 
    """
    def __init__(self, R, T, zfar=400, sigma=1e-4, image_size=1080, device=torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.lights = PointLights(device=device, location=[[0.0, 300.0, 300.0]])
        self.renderers = []
        self.cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                           T=T[None, i, ...], zfar=zfar) for i in range(R.shape[0])]
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=20,
            max_faces_per_bin=50000,
        )
        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras[0], 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=self.cameras[0],
                lights=self.lights,
            )
        )

    def forward_rgb(self, mesh, cameras):
        images_rgb = []
        materials = Materials(
            device=self.device,
            shininess=5.0
        )
        for camera in cameras:
            images_rgb.append(self.renderer_textured(mesh, cameras=camera, lights=self.lights, materials=materials)[..., :3])
        images_rgb = torch.stack(images_rgb, dim=1).squeeze()
        return images_rgb

    def forward(self, mesh, camera_id="all"):
        if camera_id == "all":
            cameras = self.cameras
        else:
            cameras = [self.cameras[i] for i in camera_id]        
        images_rgb = self.forward_rgb(mesh, cameras)

        return images_rgb


def render(
        renderer,
        skelB,
        parents,
        nameB,
        quatB_rt,
        geometric_dict,
        center=None,
):
    rest_skelB = skelB.reshape(-1, 3)
    meshB = geometric_dict['rest_vertices_dict'][nameB]
    skinB_weights = geometric_dict['skinning_weights_dict'][nameB]
    vertices_normals = geometric_dict['rest_vertex_normals_dict'][nameB]
    vertices_lbs, normals_lbs = linear_blend_skinning(parents, quatB_rt, rest_skelB, meshB, skinB_weights, vertices_normals)

    if center is None:
        vertex_min, _ = torch.min(vertices_lbs[0], dim=0)
        vertex_max, _ = torch.max(vertices_lbs[0], dim=0)
        center = (vertex_min + vertex_max) / 2.0
        center[2] = 0.0
    video = []
    for t in range(vertices_lbs.shape[0]):
        vertices_lbs[t] = vertices_lbs[t] - center
        mesh = Meshes(vertices_lbs[t, None], geometric_dict['rest_faces_dict'][nameB][None], geometric_dict['texture_dict'][nameB])
        image_rgb = renderer(mesh)
        video.append(image_rgb[None])

    if len(video) > 0:
        video = torch.cat(video, dim=0).cpu().numpy()
    else:
        video = None

    return video, center
