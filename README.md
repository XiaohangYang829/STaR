# \[ICCV 2025\] STaR: Seamless Spatial-Temporal Aware Motion Retargeting with Penetration and Consistency Constraints

This is the code for the ICCV 2025 paper ["STaR: Seamless Spatial-Temporal Aware Motion Retargeting with Penetration and Consistency Constraints"](https://arxiv.org/abs/2504.06504), by Xiaohang Yang, et al.

Neural motion retargeting model, STaR, is designed for a balance between: (1) motion semantics preservation, (2) physical plausibility, and (3) temporal consistency.

 > 🚀 **Update:** The full code will be released soon—stay tuned! 👀

- [x] test code, with network weight and a processed test set;
- [ ] inference code, with a PyTorch3D-based rendering function;
- [ ] data preparation;
- [ ] release a skel2mesh motion retargeting sub-model based on STaR, which can be used for retargeting motion generation results to new characters;

![Architecture Details](assets/detail_dark.png)

## Environment Setup

### 1. Creete Conda environment
We tested with Python 3.9, CUDA 11.8.
```
conda create python=3.9 -n star
conda activate star
```

### 2. Install main dependencies

* **Install [PyTorch](https://pytorch.org/get-started/previous-versions/). We have tested on PyTorch 2.3.1**.

```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

* **Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d) (for visualisation only).**

  Build from source code or refer to [this link](https://github.com/facebookresearch/pytorch3d/discussions/1752) for pre-built wheels that matches your environments.

### 3. Install other packages

```
pip install numpy==1.23.3 tqdm wandb scipy matplotlib==3.4.2 ninja
cd ./submodules
pip install ./pointnet2_ops_lib/.
```

## Test

Download the processed test set from [here](https://drive.google.com/drive/folders/1DFx-JhjPzE4njujtRWxGRSfl2m0NFU8f?usp=drive_link). Put all content under ```./datasets/mixamo/```

```
python test.py --config ./config/test_config.yaml
```

## Data Preparation (TODO)

(We use scripts to download, match motion instances and create datesets. This part is a little bit complicated, we will release this part as soon as possible, together with a pre-processed version.)

## Inference and Visualisation (TODO)

```
python inference.py --config ./config/config.yaml
```

## Train

```
python train.py --config ./config/config.yaml
```

## Acknowledgments
The code is partially built based on [R2ET](https://github.com/Kebii/R2ET).

The dataset is taken from [Adobe Mixamo](https://www.mixamo.com/).

The Chamfer Distance is adapted from [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch).

The shape encoder and PointNet2_lib is adapted from [PCT](https://github.com/Strawberry-Eat-Mango/PCT_Pytorch).

Thanks to all of them.

## Citation
If you find this work helpful, please consider citing it as follows:
```   
@InProceedings{Yang_2025_STaR,
    author    = {Yang, Xiaohang and Wang, Qing and Yang, Jiahao and Slabaugh, Gregory and Yuan, Shanxin},
    title     = {STaR: Seamless Spatial-Temporal Aware Motion Retargeting with Penetration and Consistency Constraints},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {Oct},
    year      = {2025}
}

```
