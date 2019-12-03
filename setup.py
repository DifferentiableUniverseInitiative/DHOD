#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

setup(name = 'diffhod',
      version = __version__,
      python_requires='>3.5.2',
      description = 'differentiable HOD',
      provides = ['diffhod'],
      packages = ['diffhod']
      )
