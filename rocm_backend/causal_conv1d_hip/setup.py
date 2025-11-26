"""
Setup script for Causal Conv1D HIP Extension
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import os

# Get ROCm path
rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')

# Determine GPU architecture
gpu_arch = os.environ.get('GPU_ARCH', 'gfx942')

# PyTorch include directories
torch_include = torch.utils.cpp_extension.include_paths()

# Extension definition
ext_modules = [
    CppExtension(
        name='causal_conv1d_hip_ext',
        sources=[
            'causal_conv1d_hip.cpp',
            'causal_conv1d_hip_launcher.hip',
        ],
        include_dirs=torch_include + [
            f'{rocm_path}/include',
            f'{rocm_path}/include/hipcub',
        ],
        library_dirs=[
            f'{rocm_path}/lib',
        ],
        libraries=[
            'amdhip64',
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-std=c++17',
                '-fPIC',
            ],
            'nvcc': [  # In ROCm, this maps to hipcc
                f'--offload-arch={gpu_arch}',
                '-O3',
                '-std=c++17',
            ],
        },
        language='c++',
    ),
]

setup(
    name='causal_conv1d_hip',
    version='0.1.0',
    author='Tri Dao (CUDA), Adapted for HIP',
    description='Causal Conv1D with HIP/ROCm backend',
    long_description=open('../../../README.md').read() if os.path.exists('../../../README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False),
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.12',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

