```
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
# ~~~
```

In this repository, we provide jupyter-notebooks and the entire code for the numerical experiments in Section 4 of the paper 
"A NON-CONFORMING DUAL APPROACH FOR ADAPTIVE TRUST-REGION REDUCED BASIS APPROXIMATION OF PDE-CONSTRAINED OPTIMIZATION"
by Tim Keil, Luca Mechelli, Mario Ohlberger, Felix Schindler and Stefan Volkwein. 

For just taking a look at the provided (precompiled) jupyter-notebooks, you do not need to install the software.
Just go to `notebooks/Paper_1_simulations`. If you want to have a closer look at the implementation or compile the results by
yourself, we provide simple setup instructions for configuring your own Python environment in a few steps.
We note that our setup instructions are written for Linux or Mac OS only and we do not provide setup instructions for Windows.
We also emphasize that our experiments have been computed on a fresh Ubuntu 18 system with Python version 3.7.5. with 12 GB RAM. 

# Organization of the repository

Our implementation is based on pyMOR (https://github.com/pymor/pymor).
Further extensions that we used for this paper can be found in the directory `pdeopt/`. 
For our three optimization experiments we considered ten different starting parameters with different seeds in the random generator. 
The complete results from these starting values are stored in each respective directory under

* **Section 4.2:** `notebooks/Paper_1_simulations/Model_Problem_1_FIN_6_Parameters/`
* **Section 4.3.2:** `notebooks/Paper_1_simulations/Model_Problem_2_EXC_10_Parameters_5e-4/`
* **Section 4.3.2:** `notebooks/Paper_1_simulations/Model_Problem_2_EXC_10_Parameters_1e-6/`

where the last two solve the same problem with same starting parameters, but with different stopping tolerance tau_FOC.
We also provide an extensive view of the results of the estimator study under 

* **Section 4.3.1:** `notebooks/Paper_1_simulations/Model_Problem_2_Estimator_study/`

Note that we have partially used a machine with 256 GB RAM for handling the basis sizes n=40,48,56.

# How to find figures and tables from the paper

We provide instructions on how to find all figures and tables from the paper. 

**Figure 1**: You can see the fin geometry and mesh in the directory `fin_data/`

**Figure 2**: This result is based on starting value seed 744 (Starter744) in 
`notebooks/Paper_1_simulations/Model_Problem_1_FIN_6_Parameters/All_BFGS_methods_in_FIN6-TEST11.ipynb`. 
Go to the bottom of the notebook to see the Figure.
If you followed the setup instructions you can also construct this figure by running the file `figure_2.py`
in `notebooks/Paper_1_simulations/Model_Problem_1_FIN_6_Parameters/results/` 

**Figure 3**: The data of the blueprint is in `EXC_data/`. 
The used file for Figure 3 is `full_diffusion_with_big_numbers_with_D.png`

**Figure 4**: You can view Figure 4 (plus the decay for the sensitivity estimators) in 
`notebooks/Paper_1_simulations/Model_Problem_2_Estimator_study/Extended_results_of_the_estimator_study(Figure_4).ipynb`.
This notebook gathers all data from all notebooks for all basis sizes.

**Figure 5**: This result is based on starting value seed 10 (Starter10) which you can view in
`notebooks/Paper_1_simulations/Model_Problem_2_EXC_10_Parameters_5e-4/All_BFGS_methods_in_EXC10-TEST10.ipynb` for (A) and
`notebooks/Paper_1_simulations/Model_Problem_2_EXC_10_Parameters_1e-6/All_BFGS_methods_in_EXC10-Test10.ipynb` for (B).
Also, you can run the corresponding file in the respective `results/` directory.

**Table 1**: In the same `results/` directory of Figure 2, you can run `table_1.py` to get (a sligthly changed version of) Table 1.

**Table 2A**: You can get (a sligthly changed version of) Table 3 by running table_2A.py in `notebooks/Paper_1_simulations/Model_Problem_2_EXC_10_Parameters_5e-4/results/`

**Table 2B**: You can get (a sligthly changed version of) Table 4 by running table_2B.py in `notebooks/Paper_1_simulations/Model_Problem_2_EXC_10_Parameters_1e-6/results/`

# Setup

On a Linux or Mac OS system with Python and git installed, clone
the repo in your favorite directory

```
git clone https://github.com/TiKeil/NCD-corrected-TR-RB-approach-for-pde-opt
```

Initialize all submodules via

```
cd NCD-corrected-TR-RB-approach-for-pde-opt
git submodule update --init --recursive
```

Now, run the provided setup file via 

```
./setup.sh
```

# Running the jupyter-notebooks

If you want to interactively view or compile the notebooks, just activate and start jupyter-notebook 

```
source venv/bin/activate
jupyter-notebook --notebook-dir=notebooks
```

We recommend to use notebook extensions for a better overview in the notebooks.
After starting the jupyter-notebook server go to Nbextensions, deactivate the first box and activate at least `codefolding` and `Collapsible Headings`. 
