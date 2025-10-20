#include <torch/torch.h>
#include <vector>


int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor idx1);


int chamfer_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor idx1) {
    return chamfer_cuda_forward(xyz1, xyz2, dist1, idx1);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &chamfer_forward, "chamfer forward (CUDA)");
}