from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='chamfer',
    ext_modules=[
        CUDAExtension('chamfer', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
