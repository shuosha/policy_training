from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

# Read README for long_description
long_description = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="openpi",
    version="0.1.0",
    description="Physical Intelligence open source repo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    license="MIT",  # uses LICENSE file via license_files below
    license_files=["LICENSE"],
    url="https://github.com/Physical-Intelligence/openpi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "augmax>=0.3.4",
        "dm-tree>=0.1.8",
        "einops>=0.8.0",
        "equinox>=0.11.8",
        "flatbuffers>=24.3.25",
        "flax==0.10.2",
        "fsspec[gcs]>=2024.6.0",
        "gym-aloha>=0.1.1",
        "imageio>=2.36.1",
        "jax[cuda12]==0.5.3",
        "jaxtyping==0.2.36",
        "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        # "openpi-client @ file:///" + str((HERE / "packages" / "openpi-client").resolve()),
        "ml_collections==1.0.0",
        "numpy>=1.22.4,<2.0.0",
        "numpydantic>=1.6.6",
        "opencv-python>=4.10.0.84",
        "orbax-checkpoint==0.11.13",
        "pillow>=11.0.0",
        "sentencepiece>=0.2.0",
        "torch>=2.7.0",
        "tqdm-loggable>=0.2",
        "typing-extensions>=4.12.2",
        "tyro>=0.9.5",
        "wandb>=0.19.1",
        "filelock>=3.16.1",
        "beartype==0.19.0",
        "treescope>=0.1.7",
        "transformers==4.48.1",
        "rich>=14.0.0",
        "polars>=1.30.0",
        # uv override-dependencies mapped as pins here:
        "ml-dtypes==0.4.1",
        "tensorstore==0.1.74",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.4",
            "ruff>=0.8.6",
            "pre-commit>=4.0.1",
            "ipykernel>=6.29.5",
            "ipywidgets>=8.1.5",
            "matplotlib>=3.10.0",
            "pynvml>=12.0.0",
        ],
        "rlds": [
            "dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a",
            "tensorflow-cpu==2.15.0",
            "tensorflow-datasets==4.9.9",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)