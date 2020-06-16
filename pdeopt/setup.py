# ~~~
# This file is part of the paper:
#
#  "A NON-CONFORMING DUAL APPROACH FOR ADAPTIVE TRUST-REGION REDUCED BASIS
#           APPROXIMATION OF PDE-CONSTRAINED OPTIMIZATION"
#
#   https://github.com/TiKeil/NCD-corrected-TR-RB-approach-for-pde-opt
#
# Copyright 2019-2020 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Tim Keil (2020)
# ~~~

from setuptools import setup

setup(name='pdeopt',
      version='2020.1',
      description='Pymor support for PDE-constrained optimization',
      author='Tim Keil, Luca Mechelli',
      author_email='tim.keil@wwu.de',
      license='MIT',
      packages=['pdeopt'])
