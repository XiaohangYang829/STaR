from torch import nn
from torch.autograd import Function
import torch
import importlib
import os

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

chamfer_found = importlib.find_loader("chamfer_3D_Mine") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer 3D")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    build_path = cur_path.replace('chamfer3D', 'tmp')
    os.makedirs(build_path, exist_ok=True)

    from torch.utils.cpp_extension import load
    chamfer_3D_Mine = load(name="chamfer_3D",
          sources=[
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
              ], build_directory=build_path)
    print("Loaded JIT 3D CUDA chamfer distance")

else:
    import chamfer_3D_Mine
    print("Loaded compiled 3D CUDA chamfer distance")


# # Chamfer's distance module @thibaultgroueix
# # GPU tensors only
# class chamfer_3DFunction(Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         batchsize, n, dim = xyz1.size()
#         assert dim==3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
#         _, m, dim = xyz2.size()
#         assert dim==3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
#         device = xyz1.device

#         device = xyz1.device

#         dist1 = torch.zeros(batchsize, n)
#         dist2 = torch.zeros(batchsize, m)

#         idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
#         idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

#         dist1 = dist1.to(device)
#         dist2 = dist2.to(device)
#         idx1 = idx1.to(device)
#         idx2 = idx2.to(device)
#         torch.cuda.set_device(device)

#         chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
#         ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
#         return dist1, dist2, idx1, idx2

#     @staticmethod
#     def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
#         graddist1 = graddist1.contiguous()
#         graddist2 = graddist2.contiguous()
#         device = graddist1.device

#         gradxyz1 = torch.zeros(xyz1.size())
#         gradxyz2 = torch.zeros(xyz2.size())

#         gradxyz1 = gradxyz1.to(device)
#         gradxyz2 = gradxyz2.to(device)
#         chamfer_3D.backward(
#             xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
#         )
#         return gradxyz1, gradxyz2


class chamfer_3DFunction_Mine(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, dim = xyz1.size()
        assert dim==3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
        _, m, dim = xyz2.size()
        assert dim==3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)

        dist1 = dist1.to(device)
        idx1 = idx1.to(device)
        torch.cuda.set_device(device)

        chamfer_3D_Mine.forward(xyz1, xyz2, dist1, idx1)
        # ctx.save_for_backward(xyz1, xyz2, idx1)
        return dist1, idx1

    ''' I dont need backward for SDF loss '''
    # @staticmethod
    # def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
    #     xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
    #     graddist1 = graddist1.contiguous()
    #     graddist2 = graddist2.contiguous()
    #     device = graddist1.device

    #     gradxyz1 = torch.zeros(xyz1.size())
    #     gradxyz2 = torch.zeros(xyz2.size())

    #     gradxyz1 = gradxyz1.to(device)
    #     gradxyz2 = gradxyz2.to(device)
    #     chamfer_3D_Mine.backward(
    #         xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
    #     )
    #     return gradxyz1, gradxyz2


# class chamfer_3DFunction_ReturnDist(Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         batchsize, n, dim = xyz1.size()
#         assert dim==3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
#         _, m, dim = xyz2.size()
#         assert dim==3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
#         device = xyz1.device

#         dist1 = torch.zeros(batchsize, n)
#         idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)

#         dist1 = dist1.to(device)
#         idx1 = idx1.to(device)
#         torch.cuda.set_device(device)

#         chamfer_3D_Mine.forward(xyz1, xyz2, dist1, idx1)
#         ctx.save_for_backward(xyz1, xyz2, idx1)
#         return dist1, idx1

#     @staticmethod
#     def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
#         graddist1 = graddist1.contiguous()
#         graddist2 = graddist2.contiguous()
#         device = graddist1.device

#         gradxyz1 = torch.zeros(xyz1.size())
#         gradxyz2 = torch.zeros(xyz2.size())

#         gradxyz1 = gradxyz1.to(device)
#         gradxyz2 = gradxyz2.to(device)
#         chamfer_3D_Mine.backward(
#             xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
#         )
#         return gradxyz1, gradxyz2


# class chamfer_3DDist(nn.Module):
#     def __init__(self):
#         super(chamfer_3DDist, self).__init__()

#     def forward(self, input1, input2):
#         input1 = input1.contiguous()
#         input2 = input2.contiguous()
#         return chamfer_3DFunction.apply(input1, input2)


class chamfer_3DDist_Mine(nn.Module):
    def __init__(self):
        super(chamfer_3DDist_Mine, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction_Mine.apply(input1, input2)


# class chamfer_3DDist_ReturnDist(nn.Module):
#     def __init__(self):
#         super(chamfer_3DDist_Mine, self).__init__()

#     def forward(self, input1, input2):
#         input1 = input1.contiguous()
#         input2 = input2.contiguous()
#         return chamfer_3DFunction_ReturnDist.apply(input1, input2)


