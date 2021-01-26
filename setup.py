#!/usr/bin/env python
from __future__ import print_function
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='micro21cm',
      version='0.1',
      description='Model-Independent Constraints on Reionization from Observations of the 21-cm background',
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url='https://github.com/mirochaj/micro21cm',
      packages=['micro21cm'],
     )
