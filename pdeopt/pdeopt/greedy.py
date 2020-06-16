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
#   Tim Keil (2019 - 2020)
# ~~~

import numpy as np

from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy
from pymor.algorithms.adaptivegreedy import adaptive_weak_greedy

def pdeopt_greedy(fom, reductor, training_set, use_estimator=True, error_norm=None,
              J_atol=None, DJ_atol=None, rtol=None, max_extensions=None, extension_params={}, pool=None):
    if 'estimator_depth' not in extension_params:
        # start with optimizing functional
        surrogate = QuadraticPdeoptSurrogate(fom, reductor, 'functional')
    else:
        surrogate = QuadraticPdeoptSurrogate(fom, reductor, extension_params['estimator_depth'])

    print('Global Greedy for J target')
    result = weak_greedy(surrogate, training_set, atol=J_atol, rtol=rtol, max_extensions=max_extensions, pool=pool)
    print(' ... finished after {} extensions'.format(result['extensions']))
    if max_extensions is not None and result['extensions'] == max_extensions:
        result_DJ = {'rom': surrogate.rom, 'max_errs': [result['max_errs'][-1]], 'max_err_mus': [], 'extensions': 0,'time': 0}
        return result, result_DJ

    if 'estimator_depth' not in extension_params and DJ_atol is not None:
        print('Continuing with DJ target')
        surrogate = QuadraticPdeoptSurrogate(fom, reductor, 'gradient')
        result_DJ = weak_greedy(surrogate, training_set, atol=DJ_atol, rtol=rtol, max_extensions=max_extensions, pool=pool)
    else:
        result_DJ = {'max_errs': [result['max_errs'][-1]], 'max_err_mus': [], 'extensions': 0,'time': 0}

    result_DJ['rom'] = surrogate.rom

    return result, result_DJ

    return result

def pdeopt_adaptive_greedy(fom, reductor, parameter_space, validation_mus=0,
              J_atol=None, DJ_atol=None, rtol=None, max_extensions=None, extension_params={}, pool=None):
    if 'estimator_depth' not in extension_params:
        surrogate = QuadraticPdeoptSurrogate(fom, reductor, 'functional')
    else:
        surrogate = QuadraticPdeoptSurrogate(fom, reductor, extension_params['estimator_depth'])
    
    print('Global Greedy for J target')
    result = adaptive_weak_greedy(surrogate, parameter_space, validation_mus=validation_mus, target_error=J_atol, max_extensions=max_extensions, pool=pool)
    print(' ... finished after {} extensions'.format(result['extensions']))
    if max_extensions is not None:
        if result['extensions'] == max_extensions:
            result_DJ = {'rom': surrogate.rom, 'max_errs': [result['max_errs'][-1]], 'max_err_mus': [], 'extensions': 0,'time': 0, 'training_set_sizes': []}
            return result, result_DJ

    if 'estimator_depth' not in extension_params and DJ_atol is not None:
        print('Continuing with DJ target')
        surrogate = QuadraticPdeoptSurrogate(fom, reductor, 'gradient')
        result_DJ = adaptive_weak_greedy(surrogate, parameter_space, validation_mus=validation_mus, target_error=DJ_atol, max_extensions=max_extensions-result['extensions'], pool=pool)
    else:
        result_DJ = {'max_errs': [result['max_errs'][-1]], 'max_err_mus': [], 'extensions': 0,'time': 0, 'training_set_sizes': []}

    result_DJ['rom'] = surrogate.rom

    return result, result_DJ

class QuadraticPdeoptSurrogate(WeakGreedySurrogate):
    def __init__(self, fom, reductor, estimator_depth='functional'):
        self.__auto_init(locals())
        self.rom = None

    def evaluate(self, mus, return_all_values=False):
        if self.rom is None:
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()

        def estimate(mu):
            u_rom = self.rom.solve(mu)
            p_rom = self.rom.solve_dual(mu)
            Delta_J = self.rom.estimate_output_functional_hat(u_rom, p_rom, mu)
            J_abs = np.abs(self.rom.output_functional_hat(mu))
            if self.estimator_depth == 'functional':
                return Delta_J/J_abs
            if self.estimator_depth == 'gradient':
                Delta_J_Prime = []
                for (component, item) in self.rom.parameter_space.parameter_type.items():
                    index_dict, new_item = self.rom._collect_indices(item)
                    for (l, index) in index_dict.items():
                        Delta_J_Prime.append(self.rom.estimate_output_functional_hat_d_mu(component, index, u_rom, p_rom, mu))
            J_grad_norm = np.linalg.norm(self.rom.output_functional_hat_gradient(mu))
            if self.estimator_depth == 'gradient':
                return np.linalg.norm(Delta_J_Prime)/J_grad_norm
            else:
                return 0

        result = [estimate(mu) for mu in mus]

        if return_all_values:
            return np.hstack(result)
        else:
            errs, max_err_mus = result, mus
            max_err_ind = np.argmax(errs)
            if isinstance(errs[max_err_ind], float):
                return errs[max_err_ind], max_err_mus[max_err_ind]
            else:
                return errs[max_err_ind][0], max_err_mus[max_err_ind]

    def extend(self, mu):
        with self.logger.block('Extending basis with solution snapshot ...'):
            self.reductor.extend_bases(mu, printing=True)
        with self.logger.block('Reducing ...'):
            self.rom = self.reductor.reduce()
