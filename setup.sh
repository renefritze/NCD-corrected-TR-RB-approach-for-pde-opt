#!/bin/bash
#
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
#   Felix Schindler (2020)
#   Tim Keil        (2020)
# ~~~

set -e

# initialize the virtualenv
export BASEDIR="${PWD}"
virtualenv --python=python3 venv
source venv/bin/activate

# install python dependencies into the virtualenv
cd "${BASEDIR}"
pip install --upgrade pip
pip install $(grep Cython requirements.txt)
pip install -r requirements.txt

# install local pymor and pdeopt version
cd "${BASEDIR}"
cd pymor && pip install -e .
cd "${BASEDIR}"
cd pdeopt && pip install -e .

cd "${BASEDIR}"
echo
echo "All done! From now on run"
echo "  source venv/bin/activate"
echo "to activate the virtualenv!"
