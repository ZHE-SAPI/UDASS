from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='resample2d_cuda',
    ext_modules=[
        CUDAExtension(
            name='resample2d_package.resample2d_cuda',
            sources=[
                'resample2d_cuda.cc',
                'resample2d_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
