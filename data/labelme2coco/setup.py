import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "labelme2coco", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setuptools.setup(
    name="labelme2coco",
    version=get_version(),
    author="Fatih Cagatay Akyon",
    author_email="",
    description="Convert labelme annotations into coco format in one step",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fcakyon/labelme2coco",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=get_requirements(),
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "labelme2coco=labelme2coco.cli:app",
        ],
    },
)
