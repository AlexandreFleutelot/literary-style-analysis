from setuptools import setup, find_packages

setup(
    name="literary-style-analysis",
    version="0.1",
    packages=find_packages(),
    description="Literary Style Analysis using Triplet Networks",
    author="Alexandre FLEUTELOT",
    author_email="fleutelot.alexandre@gmail.com",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "transformers",
        "tqdm",
    ],
    extras_require={
        "cuda": [
            "torch>=2.3.1",
            "torchvision>=0.18.1",
            "torchaudio>=2.3.1",
        ],
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cu118"
    ],
)