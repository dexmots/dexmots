"""Installation script for the 'diff_manip' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "protobuf>=3.20",
    "filelock>=3.12",
    "absl-py>=2.0.0",
]


# Installation operation
setup(
    name="dmanip",
    author="Krishnan Srinivasan",
    version="0.1.0",
    description="Diffsim manipulation tasks",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(".", exclude=["*.tests", "*.tests.*", "tests.*", "tests", "external"]),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    zip_safe=False,
)

# EOF
