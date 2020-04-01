#!/usr/bin/env python
from setuptools import setup, find_packages

__version__ = '0.1'

setup(name = 'diffhod',
      version = __version__,
      python_requires='>3.5.2',
      description = 'differentiable HOD',
      packages=find_packages(),
      install_requires = ['tensorflow-probability==0.9'],
      tests_require = ['halotools']
        )
