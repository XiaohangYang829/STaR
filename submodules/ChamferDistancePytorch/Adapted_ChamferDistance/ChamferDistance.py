from torch import nn
from torch.autograd import Function
import torch
import importlib
import os


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


chamfer_found = importlib.find_loader("chamfer") is not None
if not chamfer_found:
    print("Jitting Chamfer")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    build_path = cur_path.replace('Adapted_ChamferDistance', 'tmp')
    os.makedirs(build_path, exist_ok=True)

    from torch.utils.cpp_extension import load
    chamfer = load(name="chamfer", sources=[
        "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
        "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer.cu"]),
    ], build_directory=build_path)
    print("Loaded JIT CUDA chamfer distance")
else:
    import chamfer
    print("Loaded compiled CUDA chamfer distance")


class chamfer_Function(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, dim = xyz1.size()
        assert dim==3, "Wrong last dimension for the chamfer distance!"
        _, m, dim = xyz2.size()
        assert dim==3, "Wrong last dimension for the chamfer distance!"
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)

        dist1 = dist1.to(device)
        idx1 = idx1.to(device)
        torch.cuda.set_device(device)

        chamfer.forward(xyz1, xyz2, dist1, idx1)
        return dist1, idx1


class chamfer_distance(nn.Module):
    def __init__(self):
        super(chamfer_distance, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_Function.apply(input1, input2)
