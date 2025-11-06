from setuptools import setup, find_packages

setup(
    name="openpi-client",
    version="0.1.0",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "dm-tree>=0.1.8",
        "msgpack>=1.0.5",
        "numpy>=1.22.4,<2.0.0",
        "pillow>=9.0.0",
        "tree>=0.2.4",
        "websockets>=11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.4",
        ]
    },
)