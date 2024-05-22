from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='graph_max',
    ext_modules=[
        CUDAExtension(
            name='graph_max',
            sources=['graph_max.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

