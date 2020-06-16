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

from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.functionals import GenericParameterFunctional, ConstantParameterFunctional


def build_output_coefficient(parameter_type, parameter_weights=None, mu_d_=None, parameter_scales=None,
                             state_functional=None, constant_part=None):
    def _collect_indices(item):
        item_dict = {}
        if sum(item) < 2: # case () and (1,)
            item_dict[1] = ()
            new_item = ()
        else:
            assert len(item) == 1 # case (l,) with l > 1 
            for l in range(item[0]):
                item_dict[l] = (l,)
            new_item = item
        return item_dict, new_item

    if parameter_weights is None:
        parameter_weights= {'walls': 1, 'heaters': 1, 'doors': 1, 'windows': 1}
    if parameter_scales is None:
        parameter_scales= {'walls': 1, 'heaters': 1, 'doors': 1, 'windows': 1}
    mu_d={}
    if mu_d_ is None:
        mu_d_ = {}
    for key, item in parameter_type.items():
        if not isinstance(parameter_weights[key], list):
            if len(item) == 0:  # for the case when item = () 
                parameter_weights[key] = [parameter_weights[key]]
            else:
                parameter_weights[key] = [parameter_weights[key] for i in range(item[0])]
        if key not in mu_d_:
            mu_d_[key] = 0
        if not isinstance(mu_d_[key], list):
            if len(item) == 0:  # for the case when item = () 
                mu_d[key] = [mu_d_[key]]
            else:
                if isinstance(mu_d_[key],type(np.array([]))):
                    mu_d[key] = [mu_d_[key][i] for i in range(item[0])]
                else:
                    mu_d[key] = [mu_d_[key] for i in range(item[0])]
        else:
            mu_d[key] = mu_d_[key]

    parameter_functionals = []

    if constant_part is not None:
        # sigma_d * constant_part
        parameter_functionals.append(state_functional * constant_part)
        # + 1
        parameter_functionals.append(ConstantParameterFunctional(1))

    def make_zero_expressions(parameter_type):
        zero_derivative_expression = {}
        for key, item in parameter_type.items():
            zero_expressions = np.array([], dtype='<U60')
            item_dict, new_item = _collect_indices(item)
            for (l, index) in item_dict.items():
                zero_expressions = np.append(zero_expressions, ['0'])
            if len(zero_expressions) == 1:
                zero_expressions = np.array(zero_expressions[0], dtype='<U60')
            else:
                zero_expressions = np.array(zero_expressions, dtype='<U60')
            zero_derivative_expression[key] = zero_expressions
        return zero_derivative_expression

    #prepare dict
    def make_dict_zero_expressions(parameter_type):
        zero_dict = {}
        for key, item in parameter_type.items():
            index_dict, new_item = _collect_indices(item)
            zero_ = np.empty(new_item, dtype=dict)
            zero_dict[key] = zero_
            for (l, index) in index_dict.items():
                zero_dict[key][index] = make_zero_expressions(parameter_type)
        return zero_dict

    for key, item in parameter_type.items():
        if len(item) == 0:  # for the case when item = () 
            weight = parameter_weights[key][0]
            derivative_expression = make_zero_expressions(parameter_type)
            derivative_expression[key] = '{}*{}**2*'.format(weight,parameter_scales[key]) \
                                          +'({}[{}]'.format(key,()) \
                                          +'-{}'.format(mu_d[key][0]) +')'
            second_derivative_expressions = make_dict_zero_expressions(parameter_type)
            second_derivative_expressions[key][()][key]= '{}*{}**2'.format(weight,parameter_scales[key])
            parameter_functionals.append(ExpressionParameterFunctional('{}*{}**2*0.5*({}[{}]'.format(
                                                                        weight,parameter_scales[key],key,()) \
                                                                        +'-{}'.format(mu_d[key][0])+')**2',
                                                                        parameter_type,
                                                                        derivative_expressions=derivative_expression,
                                                                        second_derivative_expressions=second_derivative_expressions))
        else:
            for i in range(item[0]):
                weight = parameter_weights[key][i]
                derivative_expression = make_zero_expressions(parameter_type)
                second_derivative_expressions = make_dict_zero_expressions(parameter_type)
                if item[0] == 1:
                    derivative_expression[key] = \
                        '{}*{}**2*({}[{}]-'.format(weight,parameter_scales[key],key,i) + '{}'.format(mu_d[key][i])+')'
                    second_derivative_expressions[key][()][key]= '{}*{}**2'.format(weight,parameter_scales[key])
                else:
                    derivative_expression[key][i] = \
                            '{}*{}**2*({}[{}]-'.format(weight,parameter_scales[key],key,i) + '{}'.format(mu_d[key][i])+')'
                    second_derivative_expressions[key][i][key][i]= '{}*{}**2'.format(weight,parameter_scales[key])
                parameter_functionals.append(ExpressionParameterFunctional('{}*{}**2*0.5*({}[{}]'.format( \
                                                                            weight,parameter_scales[key],key,i) \
                                                                            +'-{}'.format(mu_d[key][i])+')**2',
                                                                            parameter_type,
                                                                            derivative_expressions=derivative_expression,
                                                                            second_derivative_expressions=second_derivative_expressions))
    def mapping(mu):
        ret = 0
        for f in parameter_functionals:
            ret += f.evaluate(mu)
        return ret

    def make_mapping(key, i):
        def sum_derivatives(mu):
            ret = 0
            if i == -1:
                index = ()
            else:
                index = (i,)
            for f in parameter_functionals:
                ret += f.d_mu(key, index).evaluate(mu)
            return ret
        return sum_derivatives

    def make_second_mapping(key, i):
        def sum_second_derivatives(mu):
            ret = 0
            if i == -1:
                index = ()
            else:
                index = (i,)
            for f in parameter_functionals:
                ret += f.d_mu(key, index).d_mu(key, index).evaluate(mu)
            return ret
        return sum_second_derivatives

    def make_zero_mappings(parameter_type):
        zero_derivative_mappings = {}
        for key, item in parameter_type.items():
            zero_mappings = np.array([], dtype=object)
            zero_mapping = lambda mu: 0.
            if len(item) == 0:  # for the case when item = () 
                zero_mappings = np.append(zero_mappings, [zero_mapping])
                zero_mappings = np.array(zero_mappings[0])
            else:
                for i in range(item[0]):
                    zero_mappings = np.append(zero_mappings, [zero_mapping])
                if item[0] == 1:
                    zero_mappings = np.array(zero_mappings[0])
            zero_derivative_mappings[key] = zero_mappings
        return zero_derivative_mappings

    #prepare dict
    def make_dict_zero_mapping(parameter_type):
        zero_dict = {}
        for key, item in parameter_type.items():
            index_dict, new_item = _collect_indices(item)
            zero_ = np.empty(new_item, dtype=dict)
            zero_dict[key] = zero_
            for (l, index) in index_dict.items():
                zero_dict[key][index] = make_zero_mappings(parameter_type)
        return zero_dict

    derivative_mappings = make_zero_mappings(parameter_type)
    for key, item in parameter_type.items():
        if len(item) == 0:  # for the case when item = () 
            derivative_mappings[key] = np.array(make_mapping(key,-1))
        else:
            for i in range(item[0]):
                if item[0] == 1:
                    derivative_mappings[key] = np.array(make_mapping(key,-1))
                else:
                    derivative_mappings[key][i] = make_mapping(key,i)

    second_derivative_mappings = make_dict_zero_mapping(parameter_type)
    for key, item in parameter_type.items():
        if len(item) == 0:  # for the case when item = () 
            second_derivative_mappings[key][()][key] = np.array(make_second_mapping(key,-1))
        else:
            for i in range(item[0]):
                if item[0] == 1:
                    second_derivative_mappings[key][()][key] = np.array(make_second_mapping(key,-1))
                else:
                    second_derivative_mappings[key][i][key][i] = make_second_mapping(key,i)

    output_coefficient = GenericParameterFunctional(mapping, parameter_type,
                                                    derivative_mappings=derivative_mappings,
                                                    second_derivative_mappings=second_derivative_mappings)
    return output_coefficient
