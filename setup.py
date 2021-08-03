#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='diffhod',
      description="Differentiable Halo Occupation Distribution",
      python_requires='>3.6',
      description='differentiable HOD',
      author="DiffHOD developers",
      packages=find_packages(),
      install_requires=['tensorflow-probability', 'edward2'],
      tests_require=['halotools'],
      use_scm_version=True,
      setup_requires=["setuptools_scm"],
      classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
      ])
