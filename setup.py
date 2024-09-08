from setuptools import setup, find_packages
import sys

# Check if the user wants CUDA support
cuda_support = '--cuda' in sys.argv
if cuda_support:
    sys.argv.remove('--cuda')

# Base dependencies
install_requires = [
    "numpy",
    "pandas",
    "scikit-learn",
    "transformers",
    "tqdm",
]

# PyTorch dependencies
if cuda_support:
    pytorch_deps = [
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "torchaudio>=2.3.1",
    ]
else:
    pytorch_deps = [
        "torch>=2.3.1+cpu",
        "torchvision>=0.18.1+cpu",
        "torchaudio>=2.3.1+cpu",
    ]

install_requires.extend(pytorch_deps)

setup(
    name="literary-style-analysis",
    version="0.1",
    packages=find_packages(),
    description="Literary Style Analysis using Triplet Networks",
    author="Alexandre FLEUTELOT",
    author_email="fleutelot.alexandre@gmail.com",
    install_requires=install_requires,
    dependency_links=[
        "https://download.pytorch.org/whl/cu118" if cuda_support else "https://download.pytorch.org/whl/cpu"
    ],
)