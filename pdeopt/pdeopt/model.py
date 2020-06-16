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
#   Luca Mechelli (2019)
#   Tim Keil      (2019 - 2020)
# ~~~

import numpy as np
import scipy
from numbers import Number

from pymor.models.basic import StationaryModel
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.operators.constructions import ZeroOperator
from pymor.operators.interface import Operator


class QuadraticPdeoptStationaryModel(StationaryModel):

    def __init__(self, primal_model, output_functional_dict, opt_product=None, estimators=None, dual_model=None,
                 primal_sensitivity_model=None, dual_sensitivity_model=None, name=None,
                 fin_model=False, use_corrected_functional=True, use_corrected_gradient=False,
                 adjoint_approach=False, separated_bases=True, true_lagrange=True,
                 second_primal_sens_model=None, second_dual_sens_model=None):
        super().__init__(primal_model.operator, primal_model.rhs, primal_model.output_functional, primal_model.products,
                         primal_model.parameter_space, primal_model.estimator,
                         primal_model.visualizer,  name)
        self.__auto_init(locals())
        if self.opt_product is None:
            self.opt_product = primal_model.h1_product
        if self.estimators is None:
            self.estimators = {'primal': None, 'dual': None, 'output_functional_hat': None,
                               'output_functional_hat_d_mus': None, 'u_d_mu': None, 'p_d_mu': None}
        self.hessian_parts = {}
        self.local_index_to_global_index = {}
        k = 0
        for (key, item) in self.parameter_space.parameter_type.items():
            index_dict, new_item = self._collect_indices(item)
            array_ = np.empty(new_item, dtype=object)
            for (l, index) in index_dict.items():
                array_[index] = k
                k += 1
            self.local_index_to_global_index[key] = array_
        self.number_of_parameters = k

    def solve(self, mu):
        return super().solve(mu)

    def solve_dual(self, mu, U=None):
        if U is None:
            U = self.solve(mu)
        if self.dual_model is not None:
            U = self.solve(mu)
            mu = self._add_primal_to_parameter(mu, U)
            return self.dual_model.solve(mu)
        else:
            dual_fom = self._build_dual_model(U, mu)
            return dual_fom.solve(mu)

    def solve_for_u_d_mu(self, component, index, mu, U=None, second_rom=False):
        if U is None:
            U = self.solve(mu)
        if self.primal_sensitivity_model is not None and self.separated_bases and self.true_lagrange and not second_rom:
            if isinstance(self.primal_sensitivity_model, dict):
                mu = self._add_primal_to_parameter(mu, U)
                U_d_mu_old = self.primal_sensitivity_model[component][index].solve(mu)
                return U_d_mu_old
        residual_dmu_lhs = VectorOperator(self.primal_model.operator.d_mu(component, index).apply(U, mu=mu))
        residual_dmu_rhs = self.primal_model.rhs.d_mu(component, index)
        rhs_operator = residual_dmu_rhs-residual_dmu_lhs
        u_d_mu = self.primal_model.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)
        return u_d_mu

    def solve_for_u_d_eta(self, mu, eta, U=None):
        # differentiate in arbitrary direction
        if U is None:
            U = self.solve(mu)
        if self.dual_model is not None and self.separated_bases and self.true_lagrange:
            if isinstance(self.primal_sensitivity_model, dict):
                mu = self._add_primal_to_parameter(mu, U)
                result = []
                k = 0
                for (key, item) in self.parameter_space.parameter_type.items():
                    index_dict, _ = self._collect_indices(item)
                    for (l, index) in index_dict.items():
                        res = self.solve_for_u_d_mu(key,index,mu, U)
                        int_res = res.to_numpy() * eta[k]
                        result.append(res.space.from_numpy(int_res))
                        k += 1
                return result
        gradient_operator = ZeroOperator(self.primal_model.rhs.range, self.primal_model.rhs.source)
        gradient_rhs = ZeroOperator(self.primal_model.rhs.range, self.primal_model.rhs.source)
        k = 0
        for (key, item) in self.parameter_space.parameter_type.items():
            index_dict, _ = self._collect_indices(item)
            for (l, index) in index_dict.items():
                gradient_operator= gradient_operator + eta[k] * VectorOperator(self.primal_model.operator.d_mu(key, index).apply(U, mu=mu))
                gradient_rhs = gradient_rhs + eta[k] * self.primal_model.rhs.d_mu(key, index)
                k +=1
        rhs_operator = gradient_rhs - gradient_operator
        u_d_mu = self.primal_model.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)
        return u_d_mu

    def solve_for_p_d_mu(self, component, index, mu, U=None, P=None, u_d_mu=None, second_rom=False):
        if U is None:
            U = self.solve(mu)
        if P is None:
            P = self.solve_dual(mu, U)
        if u_d_mu is None:
            u_d_mu = self.solve_for_u_d_mu(component, index, mu=mu, U=U)
        if self.dual_sensitivity_model is not None and self.separated_bases and self.true_lagrange and not second_rom:
            if isinstance(self.dual_sensitivity_model, dict):
                mu = self._add_primal_to_parameter(mu, U)
                mu = self._add_dual_to_parameter(mu, P)
                mu = self._add_primal_sensitivity_to_parameter(mu, u_d_mu)
                return self.dual_sensitivity_model[component][index].solve(mu)
        if self.dual_model is not None:
            mu = self._add_primal_to_parameter(mu, U)
            residual_dmu_lhs = VectorOperator(self.dual_model.operator.d_mu(component, index).apply(P, mu=mu))
            residual_dmu_rhs = self.dual_model.rhs.d_mu(component, index)
            k_term = VectorOperator(self.output_functional_dict['dual_projected_d_u_bilinear_part'].apply_adjoint(u_d_mu, mu))
        else:
            dual_fom = self._build_dual_model(U, mu)
            residual_dmu_lhs = VectorOperator(dual_fom.operator.d_mu(component, index).apply(P, mu=mu))
            residual_dmu_rhs = dual_fom.rhs.d_mu(component, index)
            k_term = VectorOperator(self.output_functional_dict['d_u_bilinear_part'].apply(u_d_mu, mu))
        rhs_operator = residual_dmu_rhs-residual_dmu_lhs+k_term   
        if self.dual_model is not None:
            p_d_mu = self.dual_model.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)
        else:
            p_d_mu = self.primal_model.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)
        return p_d_mu

    def solve_for_p_d_eta(self, mu, eta, U=None, P=None, u_d_eta=None):
        if U is None:
            U = self.solve(mu)
        if P is None:
            P = self.solve_dual(mu, U)
        mu = self._add_primal_to_parameter(mu, U)
        if self.dual_model is not None and self.separated_bases and self.true_lagrange:
            if isinstance(self.dual_sensitivity_model, dict):
                result = []
                k = 0
                for (key, item) in self.parameter_space.parameter_type.items():
                    index_dict, _ = self._collect_indices(item)
                    for (l, index) in index_dict.items():
                        res = self.solve_for_p_d_mu(key,index,mu, U, P)
                        int_res = res.to_numpy() * eta[k]
                        result.append(res.space.from_numpy(int_res))
                        k += 1
                return result
        if self.dual_model is None:
            dual_model = self._build_dual_model(U, mu)
        else:
            dual_model = self.dual_model
        if u_d_eta is None:
            u_d_eta = self.solve_for_u_d_eta(mu=mu, eta=eta, U=U)
        gradient_operator = ZeroOperator(dual_model.rhs.range, dual_model.rhs.source)
        gradient_rhs = ZeroOperator(dual_model.rhs.range, dual_model.rhs.source)
        k = 0
        for (key, item) in self.parameter_space.parameter_type.items():
            index_dict, _ = self._collect_indices(item)
            for (l, index) in index_dict.items():
                gradient_operator= gradient_operator + eta[k] * VectorOperator(dual_model.operator.d_mu(key, index).apply(P, mu=mu))
                gradient_rhs = gradient_rhs + eta[k] * dual_model.rhs.d_mu(key, index)
                k +=1
        if self.dual_model is not None:
            k_term = VectorOperator(self.output_functional_dict['dual_projected_d_u_bilinear_part'].apply_adjoint(u_d_eta, mu))
        else:
            k_term = VectorOperator(self.output_functional_dict['d_u_bilinear_part'].apply(u_d_eta, mu))
        rhs_operator = gradient_rhs-gradient_operator+k_term
        p_d_eta = dual_model.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu)
        return p_d_eta
               
    def solve_auxiliary_dual_problem(self, mu, U=None):
        assert self.dual_model is not None, 'this is only a ROM method'
        if U is None:
            U = self.solve(mu)
        mu = self._add_primal_to_parameter(mu, U)
        rhs_operator_1 = VectorOperator(self.output_functional_dict['primal_dual_projected_op'].apply_adjoint(U, mu=mu))
        rhs_operator_2 = self.output_functional_dict['dual_projected_rhs']
        rhs_operator = rhs_operator_1 - rhs_operator_2
        Z = self.dual_model.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu) 
        return Z
    
    def solve_auxiliary_primal_problem(self, mu, Z=None, U=None, P=None):
        assert self.dual_model is not None, 'this is only a ROM method'
        if U is None:
            U = self.solve(mu)
        if P is None:
            P = self.solve_dual(mu, U)
        if Z is None:
            Z = self.solve_auxiliary_dual_problem(mu, U)
        mu = self._add_primal_to_parameter(mu, U)
        mu = self._add_dual_to_parameter(mu, P)

        rhs_operator_1 = VectorOperator(self.output_functional_dict['primal_dual_projected_op'].apply(P, mu=mu))
        rhs_operator_2 = self.output_functional_dict['primal_projected_dual_rhs']
        rhs_operator_3 = VectorOperator(self.output_functional_dict['dual_projected_d_u_bilinear_part'].apply(Z, mu=mu))
        rhs_operator = rhs_operator_2 - rhs_operator_1 - rhs_operator_3
        W = self.primal_model.operator.apply_inverse(rhs_operator.as_range_array(mu), mu=mu) 
        return W

    def output_functional_hat(self, mu, U=None, P=None):
        mu = self.parse_parameter(mu)
        if U is None:
            U = self.solve(mu=mu)
        constant_part = self.output_functional_dict['output_coefficient']
        linear_part = self.output_functional_dict['linear_part'].apply_adjoint(U, mu).to_numpy()[0][0]
        bilinear_part = self.output_functional_dict['bilinear_part'].apply2(U, U, mu)[0][0]
        correction_term = 0
        if self.use_corrected_functional and self.dual_model is not None:
            if P is None:
               P = self.solve_dual(mu=mu, U=U)
            residual_lhs = self.output_functional_dict['primal_dual_projected_op'].apply2(U, P, mu=mu)[0][0]
            residual_rhs = self.output_functional_dict['dual_projected_rhs'].apply_adjoint(P, mu=mu).to_numpy()[0][0]
            correction_term = residual_rhs - residual_lhs
        return constant_part(mu) + linear_part + bilinear_part + correction_term

    def corrected_output_functional_hat(self, mu, u=None, p=None):
        mu = self.parse_parameter(mu)
        if u is None:
            u = self.solve(mu=mu)
        if p is None:
            p = self.solve_dual(mu=mu, U=u)
        constant_part = self.output_functional_dict['output_coefficient']
        linear_part = self.output_functional_dict['linear_part'].apply_adjoint(u, mu).to_numpy()[0][0]
        bilinear_part = self.output_functional_dict['bilinear_part'].apply2(u, u, mu)[0][0]
        if self.dual_model is not None:
            residual_lhs = self.output_functional_dict['primal_dual_projected_op'].apply2(u, p, mu=mu)[0][0]
            residual_rhs = self.output_functional_dict['dual_projected_rhs'].apply_adjoint(p, mu=mu).to_numpy()[0][0]
        else:
            residual_lhs = self.primal_model.operator.apply2(u, p, mu=mu)[0][0]
            residual_rhs = self.primal_model.rhs.apply_adjoint(p, mu=mu).to_numpy()[0][0]
        correction_term = residual_rhs - residual_lhs
        # print(correction_term)
        return constant_part(mu) + linear_part + bilinear_part + correction_term

    def output_functional_hat_d_mu(self, component, index, mu, U=None, P=None, Z=None, W=None):
        if self.dual_model is not None:
            if self.use_corrected_gradient:
                if self.adjoint_approach:
                    print('Warning: You are using an inefficient way of computing the adjoint approach. Turn off currected gradient!')
                return self.corrected_output_functional_hat_d_mu(component, index, mu, U, P)
            elif self.adjoint_approach:
                return self.adjoint_corrected_output_functional_hat_d_mu(component, index, mu, U, P, Z, W)
        return self.uncorrected_output_functional_hat_d_mu(component, index, mu, U, P)

    def uncorrected_output_functional_hat_d_mu(self, component, index, mu, U=None, P=None):
        mu = self.parse_parameter(mu)
        if U is None:
            U = self.solve(mu=mu)
        if P is None:
            P = self.solve_dual(mu=mu, U=U)
        output_coefficient = self.output_functional_dict['output_coefficient']
        J_dmu = output_coefficient.d_mu(component, index).evaluate(mu)
        if self.dual_model is not None:  # This is a cheat for detecting if it's a rom
            projected_op = self.output_functional_dict['primal_dual_projected_op']
            projected_rhs = self.output_functional_dict['dual_projected_rhs']
            residual_dmu_lhs = projected_op.d_mu(component, index).apply2(U, P, mu=mu)
            residual_dmu_rhs = projected_rhs.d_mu(component, index).apply_adjoint(P, mu=mu).to_numpy()[0][0]
            projected_k = self.output_functional_dict['bilinear_part']
            projected_j = self.output_functional_dict['linear_part']
            if isinstance(projected_k, LincombOperator) and self.fin_model is False:
                k_part = projected_k.d_mu(component, index).apply2(U, U, mu=mu)
                j_part = projected_j.d_mu(component, index).apply_adjoint(U, mu=mu).to_numpy()[0][0]
            else:
                k_part = 0
                j_part = 0
        else:
            bilinear_part = self.output_functional_dict['bilinear_part']
            linear_part = self.output_functional_dict['linear_part']
            if isinstance(bilinear_part, LincombOperator) and self.fin_model is False:
                k_part = bilinear_part.d_mu(component, index).apply2(U, U, mu=mu)
                j_part = linear_part.d_mu(component, index).apply_adjoint(U, mu=mu).to_numpy()[0][0]
            else:
                k_part = 0
                j_part = 0
            residual_dmu_lhs = self.primal_model.operator.d_mu(component, index).apply2(U, P, mu=mu)
            residual_dmu_rhs = self.primal_model.rhs.d_mu(component, index).apply_adjoint(P, mu=mu).to_numpy()[0][0]
        return (J_dmu + j_part + k_part - residual_dmu_lhs + residual_dmu_rhs)[0][0]
    
    def corrected_output_functional_hat_d_mu(self, component, index, mu, U=None, P=None):
        assert self.dual_model is not None, 'not a FOM method'
        mu = self.parse_parameter(mu)
        if U is None:
            U = self.solve(mu=mu)
        if P is None:
            P = self.solve_dual(mu=mu, U=U)
        output_coefficient = self.output_functional_dict['output_coefficient']
        J_dmu = output_coefficient.d_mu(component, index).evaluate(mu)
        res_pr_du_sens = 0
        res_du_pr_sens = 0
        projected_op = self.output_functional_dict['primal_dual_projected_op']
        projected_rhs = self.output_functional_dict['dual_projected_rhs']
        residual_dmu_lhs = projected_op.d_mu(component, index).apply2(U, P, mu=mu)
        residual_dmu_rhs = projected_rhs.d_mu(component, index).apply_adjoint(P, mu=mu).to_numpy()[0][0]
        projected_k = self.output_functional_dict['bilinear_part']
        projected_j = self.output_functional_dict['linear_part']
        if isinstance(projected_k, LincombOperator) and self.fin_model is False:
            k_part = projected_k.d_mu(component, index).apply2(U, U, mu=mu)
            j_part = projected_j.d_mu(component, index).apply_adjoint(U, mu=mu).to_numpy()[0][0]
        else:
            k_part = 0
            j_part = 0
            
        U_sens = self.solve_for_u_d_mu(component, index, mu, U=U)
        P_sens = self.solve_for_p_d_mu(component, index, mu, U=U, P=P, u_d_mu=U_sens)
        if self.separated_bases:
            assert 'dual_sens_projected_rhs' in self.output_functional_dict
            sens_projected_rhs = self.output_functional_dict['dual_sens_projected_rhs'][component][index]
            sens_projected_op = self.output_functional_dict['dual_sens_projected_op'][component][index] 
            sens_projected_dual_rhs_1 = self.output_functional_dict['primal_sens_projected_dual_rhs_1'][component][index] 
            sens_projected_dual_rhs_2 = self.output_functional_dict['primal_sens_projected_dual_rhs_2'][component][index] 
            sens_projected_dual_op = self.output_functional_dict['primal_sens_projected_dual_op'][component][index]
            res_pr_du_sens = sens_projected_rhs.apply_adjoint(P_sens, mu=mu).to_numpy()[0][0] \
                    - sens_projected_op.apply2(U, P_sens, mu=mu)
            res_1 = sens_projected_dual_rhs_1.apply_adjoint(U_sens, mu=mu).to_numpy()[0][0] 
            res_2 = sens_projected_dual_rhs_2.apply2(U_sens, U, mu=mu) 
            res_3 = sens_projected_dual_op.apply2(U_sens, P, mu=mu)
        else:
            # P_sens and U_sens are in V_pr and V_du # this approach is equivalent to the adjoint approach but less efficient
            res_pr_du_sens_1 = projected_rhs.apply_adjoint(P_sens, mu=mu).to_numpy()[0][0] 
            res_pr_du_sens_2 = projected_op.apply2(U, P_sens, mu=mu)
            res_pr_du_sens = res_pr_du_sens_1 - res_pr_du_sens_2
            res_1 = projected_j.apply_adjoint(U_sens, mu=mu).to_numpy()[0][0] 
            res_2 = 2*projected_k.apply2(U_sens, U, mu=mu) 
            res_3 = projected_op.apply2(U_sens, P, mu=mu)
        res_du_pr_sens = res_1 + res_2 - res_3
        return (J_dmu + j_part + k_part - residual_dmu_lhs + residual_dmu_rhs + res_pr_du_sens + res_du_pr_sens)[0][0]

    def adjoint_corrected_output_functional_hat_d_mu(self, component, index, mu, U=None, P=None, Z=None, W=None):
        assert self.dual_model is not None, 'not a FOM method'
        mu = self.parse_parameter(mu)
        if U is None:
            U = self.solve(mu=mu)
        if P is None:
            P = self.solve_dual(mu=mu, U=U)
        if Z is None:
            Z = self.solve_auxiliary_dual_problem(mu, U=U)
        if W is None:
            W = self.solve_auxiliary_primal_problem(mu, Z=Z, U=U, P=P)
        
        output_coefficient = self.output_functional_dict['output_coefficient']
        J_dmu = output_coefficient.d_mu(component, index).evaluate(mu)
        projected_op = self.output_functional_dict['primal_dual_projected_op']
        projected_rhs = self.output_functional_dict['dual_projected_rhs']
        residual_dmu_lhs = projected_op.d_mu(component, index).apply2(U, P, mu=mu)
        residual_dmu_rhs = projected_rhs.d_mu(component, index).apply_adjoint(P, mu=mu).to_numpy()[0][0]
        projected_k = self.output_functional_dict['bilinear_part']
        projected_j = self.output_functional_dict['linear_part']
        if isinstance(projected_k, LincombOperator) and self.fin_model is False:
            k_part = projected_k.d_mu(component, index).apply2(U, U, mu=mu)
            j_part = projected_j.d_mu(component, index).apply_adjoint(U, mu=mu).to_numpy()[0][0]
        else:
            k_part = 0
            j_part = 0
        # auxilialy problems
        mu = self._add_primal_to_parameter(mu, U)
        mu = self._add_dual_to_parameter(mu, P)
        
        w_term_1 = self.primal_model.operator.d_mu(component,index).apply2(U, W, mu=mu)
        w_term_2 = self.primal_model.rhs.d_mu(component,index).apply_adjoint(W, mu=mu).to_numpy()[0][0]
        w_term = w_term_2 - w_term_1
        z_term_1 = self.dual_model.operator.d_mu(component,index).apply2(Z, P, mu=mu)
        z_term_2 = self.dual_model.rhs.d_mu(component,index).apply_adjoint(Z, mu=mu).to_numpy()[0][0]
        z_term = z_term_1 - z_term_2 

        return (J_dmu + j_part + k_part - residual_dmu_lhs + residual_dmu_rhs + w_term + z_term)[0][0]
    
    def output_functional_hat_gradient(self, mu, adjoint_approach=None, U=None, P=None):
        if adjoint_approach is None:
            if self.dual_model is not None:
                adjoint_approach = self.adjoint_approach
        gradient = []
        mu = self.parse_parameter(mu)
        if U is None:
            U = self.solve(mu=mu)
        if P is None:
            P = self.solve_dual(mu=mu, U=U)
        if adjoint_approach:
            Z = self.solve_auxiliary_dual_problem(mu, U=U)
            W = self.solve_auxiliary_primal_problem(mu, Z=Z, U=U, P=P)
        for (key, item) in self.parameter_space.parameter_type.items():
            index_dict, _ = self._collect_indices(item)
            for (l, index) in index_dict.items():
                if adjoint_approach:
                    gradient.append(self.adjoint_corrected_output_functional_hat_d_mu(key, index, mu,
                        U, P, Z, W))
                else:
                    gradient.append(self.output_functional_hat_d_mu(key, index, mu, U, P))
        gradient = np.array(gradient)
        return gradient

    def output_functional_hat_gradient_adjoint(self, mu):
        return self.output_functional_hat_gradient(mu, adjoint_approach=True)

    def extract_hessian_parts(self, mu, extract_sensitivities=True):
        mu_tuple = tuple(self.parse_parameter_inverse(mu))
        if mu_tuple not in self.hessian_parts:
            # self.hessian_parts = {}
            output_coefficient = self.output_functional_dict['output_coefficient']
            d_u_bilinear_part = self.output_functional_dict['d_u_bilinear_part']
            d_u_linear_part = self.output_functional_dict['d_u_linear_part']
            bilinear_part = self.output_functional_dict['bilinear_part']
            linear_part = self.output_functional_dict['linear_part']
            parts_dict = {}
            U = self.solve(mu=mu)
            P = self.solve_dual(mu=mu, U=U)
            U_d_mu_dict = {}
            P_d_mu_dict = {}
            k = 0
            if extract_sensitivities:
                for (key, item) in self.parameter_space.parameter_type.items():
                    index_dict, new_item = self._collect_indices(item)
                    U_d_mu = np.empty(new_item, dtype=object)
                    P_d_mu = np.empty(new_item, dtype=object)
                    for (l, index) in index_dict.items():
                        U_d_mu_ = self.solve_for_u_d_mu(key, index, mu, U)
                        P_d_mu_ = self.solve_for_p_d_mu(key, index, mu, U, P, U_d_mu_)
                        U_d_mu[index] = U_d_mu_
                        P_d_mu[index] = P_d_mu_
                    U_d_mu_dict[key] = U_d_mu
                    P_d_mu_dict[key] = P_d_mu
                gradient_operator_1, gradient_operator_2, second_gradient_operator = [], [], []
                gradient_rhs, second_gradient_rhs, J_vector, j_gradient, k_gradient = [], [], [], [], []
                for (key, item) in self.parameter_space.parameter_type.items():
                    index_dict, _ = self._collect_indices(item)
                    for (l, index) in index_dict.items():
                        J_vector.append(output_coefficient.d_mu(key, index).d_mu(key, index).evaluate(mu))
                        if isinstance(bilinear_part, LincombOperator) and self.fin_model is False:
                            J_vector[-1] += bilinear_part.d_mu(key, index).d_mu(key, index).apply2(U, U, mu=mu)[0][0]
                            J_vector[-1] += linear_part.d_mu(key, index).d_mu(key, index).apply_adjoint(U,
                                                                                                    mu=mu).to_numpy()[0][0]
                        go_1, go_2, rhs, s_op, s_rhs, gj, gk  = [], [], [], 0, 0, [], []
                        k_ = 0
                        proj_ops = self.projected_hessian[key][index] # be careful ! this is key and not key_
                        for (key_, item_) in self.parameter_space.parameter_type.items():
                            index_dict_, _ = self._collect_indices(item_)
                            for (l_, index_) in index_dict_.items():
                                go_1.append(proj_ops['PS_D_op'][key_][index_].apply2(U_d_mu_dict[key_][index_],P,mu=mu)[0][0])
                                go_2.append(proj_ops['P_DS_op'][key_][index_].apply2(U, P_d_mu_dict[key_][index_],mu=mu)[0][0])
                                rhs.append(proj_ops['DS_rhs'][key_][index_].apply_adjoint(P_d_mu_dict[key_][index_], mu=mu).to_numpy()[0][0])
                                if isinstance(d_u_linear_part, LincombOperator) and self.fin_model is False:
                                    gj.append(proj_ops['PS_j'][key_][index_].apply_adjoint(U_d_mu_dict[key_][index_], mu=mu).to_numpy()[0][0])
                                else:
                                    gj.append(0)
                                if isinstance(d_u_bilinear_part, LincombOperator) and self.fin_model is False:
                                    gk.append(proj_ops['PS_P_k'][key_][index_].apply2(U_d_mu_dict[key_][index_], U, mu=mu)[0][0])
                                else:
                                    gk.append(0)
                                s_rhs += proj_ops['P_rhs'][key_][index_].apply_adjoint(P,mu=mu).to_numpy()[0][0]
                                s_op += proj_ops['P_D_op'][key_][index_].apply2(U, P,mu=mu)[0][0]
                                k_ += 1
                        gradient_operator_1.append(go_1)
                        gradient_operator_2.append(go_2)
                        gradient_rhs.append(rhs)
                        second_gradient_operator.append(s_op)
                        second_gradient_rhs.append(s_rhs)
                        j_gradient.append(gj)
                        k_gradient.append(gk)
                        k +=1
                gradient_vector = []
                for l in range(k):
                    gradient_vector.append([gradient_operator_1[l]] + [gradient_operator_2[l]] + [gradient_rhs[l]] + [j_gradient[l]] + [k_gradient[l]])
                    gradient_vector[-1] = np.einsum('jk -> k', gradient_vector[-1])
                second_gradient_vector = [second_gradient_operator] + [second_gradient_rhs] + [J_vector]
                second_gradient_vector = np.einsum('jk -> k', second_gradient_vector)
                parts_dict['gradient_vector'] = gradient_vector
                parts_dict['second_gradient_vector'] = second_gradient_vector
            else:
                parts_dict['U'], parts_dict['P'] = U, P
            self.hessian_parts[mu_tuple] = parts_dict
        else:
            # print('I do not need to solve any equation')
            parts_dict = self.hessian_parts[mu_tuple]
        if extract_sensitivities:
            gradient_vector = parts_dict['gradient_vector']
            second_gradient_vector = parts_dict['second_gradient_vector']
        else:
            U, P = parts_dict['U'], parts_dict['P']
        if extract_sensitivities:
            return gradient_vector, second_gradient_vector
        else:
            return U, P

    def output_functional_hessian_operator(self, mu, eta, printing=False):
        mu = self.parse_parameter(mu)
        output_coefficient = self.output_functional_dict['output_coefficient']
        d_u_bilinear_part = self.output_functional_dict['d_u_bilinear_part']
        d_u_linear_part = self.output_functional_dict['d_u_linear_part']
        bilinear_part = self.output_functional_dict['bilinear_part']
        linear_part = self.output_functional_dict['linear_part']
        gradient_operator_1, gradient_operator_2, second_gradient_operator = [], [], []
        gradient_rhs, second_gradient_rhs, J_vector, j_gradient, k_gradient = [], [], [], [], []
        k = 0
        if printing:
            print('this is my current mu {}'.format(mu))
            print('this is my current eta {}'.format(eta))
        if self.dual_model is not None:
            assert 0, 'You can not compute a ROM version of the Hessian' 
        else:
            U, P = self.extract_hessian_parts(mu, extract_sensitivities=False) 
            U_d_eta = self.solve_for_u_d_eta(mu, eta, U)
            P_d_eta = self.solve_for_p_d_eta(mu, eta, U, P, U_d_eta)
            for (key, item) in self.parameter_space.parameter_type.items():
                index_dict, _ = self._collect_indices(item)
                for (l, index) in index_dict.items():
                    if isinstance(d_u_linear_part, LincombOperator) and self.fin_model is False:
                        j_gradient.append(d_u_linear_part.d_mu(key, index).apply_adjoint(U_d_eta, mu=mu).to_numpy()[0][0])
                    else:
                        j_gradient.append(0)
                    if isinstance(d_u_bilinear_part, LincombOperator) and self.fin_model is False:
                        k_gradient.append(d_u_bilinear_part.d_mu(key, index).apply2(U_d_eta, U, mu=mu)[0][0])
                    else:
                        k_gradient.append(0)

                    gradient_rhs.append(self.primal_model.rhs.d_mu(key, index).apply_adjoint(P_d_eta, mu=mu).to_numpy()[0][0])
                    gradient_operator_1.append(self.primal_model.operator.d_mu(key, index).apply2(U_d_eta, P, mu=mu)[0][0])
                    gradient_operator_2.append(self.primal_model.operator.d_mu(key, index).apply2(U, P_d_eta, mu=mu)[0][0])
                    
                    J_vector.append(output_coefficient.d_mu(key, index).d_mu(key, index).evaluate(mu) * eta[k])
                    if isinstance(bilinear_part, LincombOperator) and self.fin_model is False:
                        J_vector[-1] += bilinear_part.d_mu(key, index).d_mu(key, index).apply2(U, U, mu=mu)[0][0] * eta[k]
                        J_vector[-1] += linear_part.d_mu(key, index).d_mu(key, index).apply_adjoint(U,
                                                                                                mu=mu).to_numpy()[0][0] * eta[k]
                    second_rhs_vector = []
                    second_operator_vector = []
                    for (key_, item_) in self.parameter_space.parameter_type.items():
                        index_dict_, _ = self._collect_indices(item_)
                        for (li, index_) in index_dict_.items():
                            second_rhs_vector.append(self.primal_model.rhs.d_mu(key_, index_).d_mu(key, index).apply_adjoint(P, mu=mu).to_numpy()[0][0])
                            second_operator_vector.append(self.primal_model.operator.d_mu(key_, index_).d_mu(key, index).apply2(U, P, mu=mu)[0][0])
                    second_gradient_rhs.append(np.dot(second_rhs_vector, eta))
                    second_gradient_operator.append(np.dot(second_operator_vector, eta))
                    k +=1
            gradient_operator_1 = np.array(gradient_operator_1)
            gradient_operator_2 = np.array(gradient_operator_2)
            gradient_rhs = np.array(gradient_rhs)
            J_vector = np.array(J_vector)
            second_gradient_operator = np.array(second_gradient_operator)
            second_gradient_rhs = np.array(second_gradient_rhs)
            j_gradient = np.array(j_gradient)
            k_gradient = np.array(k_gradient)
            if len(j_gradient) == 0:
                hessian_application = gradient_rhs - gradient_operator_1 - gradient_operator_2 + J_vector
            else:
                hessian_application = j_gradient + k_gradient + gradient_rhs - gradient_operator_1 - gradient_operator_2 + J_vector + \
                            second_gradient_rhs - second_gradient_operator
        return hessian_application

    def estimate(self, U, mu):
        estimator = self.estimators['primal']
        if self.estimator is not None:
            return estimator.estimate(U, mu=mu)
        else:
            raise NotImplementedError('Model has no primal estimator.')

    def estimate_dual(self, U, P, mu):
        estimator = self.estimators['dual']
        if estimator is not None:
            mu = self._add_primal_to_parameter(mu, U)
            return estimator.estimate(U, P, mu=mu)
        else:
            raise NotImplementedError('Model has no estimator for the dual problem.')

    def estimate_output_functional_hat(self, U, P, mu, residual_based=True, both_estimators=False):
        estimator = self.estimators['output_functional_hat']
        if estimator is not None:
            mu = self._add_primal_to_parameter(mu, U)
            return estimator.estimate(U, P, mu=mu, residual_based=residual_based, both_estimators=both_estimators)
        else:
            raise NotImplementedError('Model has no estimator for the output functional hat.')

    def estimate_output_functional_hat_d_mu(self, component, index, U, P, mu, 
            U_d_mu=None, P_d_mu=None, both_estimators=False, non_assembled=False):
        estimator = self.estimators['output_functional_hat_d_mus']
        if estimator is not None:
            if self._check_input(component, index):
                mu = self._add_primal_to_parameter(mu, U)
                if self.use_corrected_gradient:
                    if U_d_mu is None:
                        U_d_mu = self.solve_for_u_d_mu(component=component, index=index, mu=mu, U=U)
                    if P_d_mu is None:
                        P_d_mu = self.solve_for_p_d_mu(component=component, index=index, mu=mu, U=U, u_d_mu=U_d_mu, P=P)
                if isinstance(estimator, dict):
                    corrected = True if self.use_corrected_gradient or (self.adjoint_approach and self.true_lagrange) else False
                    return estimator[component][index].estimate(U, P, mu=mu, corrected=corrected, \
                            U_d_mu=U_d_mu, P_d_mu=P_d_mu, both_estimators=both_estimators, non_assembled=non_assembled)
                else:
                    eta = self.extract_eta_from_component(component, index)
                    if self.second_primal_sens_model is not None:
                        print('solving additional sensitivity problem for the estimator')
                        U_d_mu = self.solve_for_u_d_mu(component=component, index=index, mu=mu, U=U, second_rom=True)
                        P_d_mu = self.solve_for_p_d_mu(component=component, index=index, mu=mu, U=U, u_d_mu=U_d_mu, P=P, second_rom=True)
                    elif self.primal_sensitivity_model is not None and self.second_primal_sens_model is None:
                        U_d_mu = self.solve_for_u_d_mu(component=component, index=index, mu=mu, U=U)
                        P_d_mu = self.solve_for_p_d_mu(component=component, index=index, mu=mu, U=U, u_d_mu=U_d_mu, P=P)
                    return estimator.estimate(U, P, mu=mu, eta=eta, U_d_eta=U_d_mu, P_d_eta=P_d_mu, non_assembled=non_assembled)
            else:
                return 0
        else:
            raise NotImplementedError('Model has no estimator for d_mu of the output functional hat. \n If you need it, set prepare_for_gradient_estimate=True in the reductor')

    def estimate_output_functional_hat_gradient_norm(self, mu, U=None, P=None, non_assembled=False):
        gradient = []
        if U is None:
            U = self.solve(mu)
        if P is None:
            P = self.solve_dual(mu, U)
        for (key, item) in self.parameter_space.parameter_type.items():
            index_dict, _ = self._collect_indices(item)
            for (l, index) in index_dict.items():
                if self.use_corrected_gradient:
                    U_d_mu = self.solve_for_u_d_mu(component=key, index=index, mu=mu, U=U)
                    P_d_mu = self.solve_for_p_d_mu(component=key, index=index, mu=mu, U=U, u_d_mu=U_d_mu, P=P)
                    gradient.append(self.estimate_output_functional_hat_d_mu(key, index, U=U, P=P, mu=mu, \
                        U_d_mu=U_d_mu, P_d_mu=P_d_mu, non_assembled=non_assembled))
                else:
                    gradient.append(self.estimate_output_functional_hat_d_mu(key, index, U, P, mu, non_assembled=non_assembled))
        gradient = np.array(gradient)
        return np.linalg.norm(gradient)

    def estimate_u_d_mu(self, component, index, U, u_d_mu, mu):
        estimator = self.estimators['u_d_mu']
        if estimator is not None:
            if self._check_input(component, index):
                mu = self._add_primal_to_parameter(mu, U)
                return estimator[component][index].estimate(U, u_d_mu, mu=mu)
            else:
                return 0
        else:
            raise NotImplementedError('Model has no estimator for u_d_mu. \n If you need it, set prepare_for_gradient_estimate=True in the reductor')
    
    def estimate_u_d_eta(self, U, u_d_eta, mu, eta):
        estimator = self.estimators['u_d_mu']
        if estimator is not None:
            mu = self._add_primal_to_parameter(mu, U)
            mu = self._add_eta_to_parameter(mu, eta)
            return estimator.estimate(U, u_d_eta, mu=mu, eta=eta)
        else:
            raise NotImplementedError('Model has no estimator for u_d_eta. \n If you need it, set prepare_for_gradient_estimate=True in the reductor')

    def estimate_p_d_mu(self, component, index, U, P, u_d_mu, p_d_mu, mu):
        estimator = self.estimators['p_d_mu']
        if estimator is not None:
            if self._check_input(component, index):
                mu = self._add_primal_to_parameter(mu, U)
                mu = self._add_dual_to_parameter(mu, P)
                mu = self._add_primal_sensitivity_to_parameter(mu, u_d_mu)
                return estimator[component][index].estimate(U, P, u_d_mu, p_d_mu, mu=mu)
            else:
                return 0
        else:
            raise NotImplementedError('Model has no estimator for p_d_mu. \n If you need it, set prepare_for_hessian_estimate=True in the reductor')
    
    def estimate_p_d_eta(self, U, P, u_d_eta, p_d_eta, mu, eta):
        estimator = self.estimators['p_d_mu']
        if estimator is not None:
            mu = self._add_primal_to_parameter(mu, U)
            mu = self._add_dual_to_parameter(mu, P)
            mu = self._add_primal_sensitivity_to_parameter(mu, u_d_eta)
            mu = self._add_eta_to_parameter(mu, eta)
            return estimator.estimate(U, P, u_d_eta, p_d_eta, mu=mu, eta=eta)
        else:
            raise NotImplementedError('Model has no estimator for p_d_eta. \n If you need it, set prepare_for_hessian_estimate=True in the reductor')
    

    def compute_coercivity(self, operator, mu, product):
        A = operator.assemble(mu).matrix
        K = product.matrix
        # source: A Tutorial on RB-Methods
        # see documentation of shift invert mode for smallest eigenvalue
        return scipy.sparse.linalg.eigsh(A, k=1, M=K, sigma=0, which='LM', return_eigenvectors=False)

    def compute_continuity_bilinear(self, operator, product, mu=None):
        # exclude zero case:
        if isinstance(operator, LincombOperator):
            if np.sum(operator.evaluate_coefficients(mu)) == 0:
                return 0
        elif not isinstance(operator, Operator):
            return 1
        A = operator.assemble(mu).matrix
        K = product.assemble().matrix
        return scipy.sparse.linalg.eigsh(A, k=1, M=K, which='LM', return_eigenvectors=False)[0]

    def compute_continuity_linear(self, operator, product, mu=None):
        riesz_rep = product.apply_inverse(operator.as_vector(mu))
        output = np.sqrt(product.apply2(riesz_rep, riesz_rep))[0][0]
        return output

    def _build_dual_model(self, U, mu=None):
        if isinstance(self.output_functional_dict['d_u_linear_part'], LincombOperator) and self.fin_model is False:
            operators = list(self.output_functional_dict['d_u_linear_part'].operators)
            coefficients = list(self.output_functional_dict['d_u_linear_part'].coefficients)
        else:
            operators = [self.output_functional_dict['d_u_linear_part']]
            coefficients = [1]
        if isinstance(self.output_functional_dict['d_u_bilinear_part'], LincombOperator) and self.fin_model is False:
            operators.extend([VectorOperator(op.apply(U)) for op in
                          self.output_functional_dict['d_u_bilinear_part'].operators])
            coefficients.extend(self.output_functional_dict['d_u_bilinear_part'].coefficients)
        else:
            operators.append(VectorOperator(self.output_functional_dict['d_u_bilinear_part'].apply(U, mu)))
            coefficients.append(1)
        dual_rhs_operator = LincombOperator(operators, coefficients)
        return self.primal_model.with_(rhs=dual_rhs_operator, parameter_space=None)

    def _add_primal_to_parameter(self, mu, U):
        assert mu is not None
        mu_with_primal = mu.copy()
        mu_with_primal['basis_coefficients'] = U.to_numpy()[0]
        return mu_with_primal

    def _add_dual_to_parameter(self, mu, P):
        assert mu is not None
        mu_with_dual = mu.copy()
        mu_with_dual['basis_coefficients_dual'] = P.to_numpy()[0]
        return mu_with_dual

    def _add_primal_sensitivity_to_parameter(self, mu, U):
        assert mu is not None
        mu_with_sens = mu.copy()
        mu_with_sens['basis_coefficients_primal_sens'] = U.to_numpy()[0]
        return mu_with_sens
    
    def _add_eta_to_parameter(self, mu, eta):
        assert mu is not None
        mu_with_eta = mu.copy()
        mu_with_eta['eta'] = eta
        return mu_with_eta

    def _check_input(self, component, index):
        # check whether component is in parameter_type
        if component not in self.parameter_type:
            return False
        # check whether index has the correct shape
        if isinstance(index, Number):
            index = (index,)
        index = tuple(index)
        for idx in index:
            assert isinstance(idx, Number)
        shape = self.parameter_type[component]
        for i,idx in enumerate(index):
            assert idx < shape[i], 'wrong input `index` given'
        return True

    def _collect_indices(self, item):
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
    
    def extract_eta_from_component(self, component, index):
        eta = np.zeros(self.number_of_parameters)
        eta[self.local_index_to_global_index[component][index]] = 1
        return eta

    def parse_parameter_inverse(self, mu):
        #convert a parameter into a numpy_array
        mu_k = []
        for (key, item) in self.parameter_space.parameter_type.items():
            if len(item) == 0:
                mu_k.append(mu[key][()])
            else:
                for i in range(item[0]):
                    mu_k.append(mu[key][i])
        mu_array = np.array(mu_k)
        return mu_array

    def pre_parse_parameter(self, mu_list):
        # convert a list into a list that can be put into parse_parameter
        mu_k = []
        k = 0
        for (key, item) in self.parameter_space.parameter_type.items():
            if len(item) == 0:
                mu_k.append(mu_list[k])
                k+=1
            else:
                if item[0] == 1:
                    mu_k.append(mu_list[k])
                    k+=1
                else:
                    mu_k_add = []
                    for i in range(item[0]):
                        mu_k_add.append(mu_list[k])
                        k += 1
                    mu_k.append(mu_k_add)
        return mu_k


def build_initial_basis(opt_fom, mus=None, build_sensitivities=False):
    primal_basis = opt_fom.solution_space.empty()
    dual_basis = opt_fom.solution_space.empty()
    if build_sensitivities:
        primal_sens_basis = {}
        dual_sens_basis = {}
        for (key, item) in opt_fom.parameter_space.parameter_type.items():
            index_dict, new_item = opt_fom._collect_indices(item)
            sens_pr = np.empty(new_item, dtype=object)
            sens_du = np.empty(new_item, dtype=object)
            for (l, index) in index_dict.items():
                sens_pr_ = opt_fom.solution_space.empty()
                sens_du_ = opt_fom.solution_space.empty()
                sens_pr[index] = sens_pr_
                sens_du[index] = sens_du_
            primal_sens_basis[key] = sens_pr
            dual_sens_basis[key] = sens_du
    for (i,mu) in enumerate(mus):
        dont_debug = 1 # set 0 for debugging 
        primal_basis.append(opt_fom.solve(mu))
        if i != 1 or dont_debug: #< -- for debuginng
            dual_basis.append(opt_fom.solve_dual(mu))
        dual_basis = gram_schmidt(dual_basis, product=opt_fom.opt_product)
        primal_basis = gram_schmidt(primal_basis, product=opt_fom.opt_product)
        if build_sensitivities:
            for (key, item) in opt_fom.parameter_space.parameter_type.items():
                index_dict, _ = opt_fom._collect_indices(item)
                for (l, index) in index_dict.items():
                    if key == 'biot' or dont_debug: #< -- for debuginng
                        if i != 2 and i!=3 or dont_debug: #< -- for debuginng
                            primal_sens_basis[key][index].append(opt_fom.solve_for_u_d_mu(key, index, mu))
                    else: #< -- for debuginng
                        if i!=3 or dont_debug: #< -- for debuginng
                            primal_sens_basis[key][index].append(opt_fom.solve_for_u_d_mu(key, index, mu))
                    dual_sens_basis[key][index].append(opt_fom.solve_for_p_d_mu(key, index, mu))
                    primal_sens_basis[key][index] = gram_schmidt(primal_sens_basis[key][index], product=opt_fom.opt_product)
                    dual_sens_basis[key][index] = gram_schmidt(dual_sens_basis[key][index], product=opt_fom.opt_product)
    if build_sensitivities:
        return primal_basis, dual_basis, primal_sens_basis, dual_sens_basis
    else:
        return primal_basis, dual_basis
