from setuptools import setup

setup(
    name="my_package",
    version="0.1",
    author="Your Name",
    url="https://github.com/yourname/yourpackage",
    description="Your package description.",
    packages=["your_package"],
    python_requires=">=3.7",
    install_requires=[
        "Pillow>=7.1",
        "matplotlib",
        "pycocotools>=2.0.2",
        "termcolor>=1.1",
        "yacs>=0.1.8",
        "tabulate",
        "cloudpickle",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore>=0.1.5,<0.1.6",
        "iopath>=0.1.7,<0.1.10",
        "dataclasses; python_version<'3.7'",
        "omegaconf>=2.1,<2.4",
        "hydra-core>=1.1",
        "black",
        "packaging",
    ],
)
