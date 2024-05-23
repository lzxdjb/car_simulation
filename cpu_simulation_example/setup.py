from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cpu',
    ext_modules=[
        CUDAExtension(
            name='cpu',
            # sources=['cpu.cpp'],
            sources=['cpu.cpp', '/media/lzx/lzx/lzx/car_simulation/src/tinympc/admm.cpp'],
            include_dirs=['/media/lzx/lzx/lzx/car_simulation/include/Eigen',
            '/media/lzx/lzx/lzx/car_simulation/examples',
            '/media/lzx/lzx/lzx/car_simulation/src'],  
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

