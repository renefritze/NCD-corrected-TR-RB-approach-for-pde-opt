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
#   Luca Mechelli (2019 - 2020)
#   Tim Keil      (2019 - 2020)
# ~~~

import numpy as np

from pymor.core.base import ImmutableObject
from pymor.algorithms.projection import project
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.constructions import VectorOperator, LincombOperator, ZeroOperator
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.basic import StationaryRBReductor
from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.parameters.functionals import BaseMaxThetaParameterFunctional
from pymor.parameters.functionals import MaxThetaParameterFunctional


class QuadraticPdeoptStationaryCoerciveReductor(CoerciveRBReductor):
    def __init__(self, fom, RBPrimal=None, RBDual=None, RBPrimalSensitivity=None,
                 RBDualSensitivity=None, opt_product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None, unique_basis=False, mu_bar=None,
                 prepare_for_gradient_estimate=False, prepare_for_sensitivity_estimate=False, adjoint_estimate=False,
                 true_lagrange=True, global_greedy=False):

        self.fin_model = fom.fin_model

        self.__auto_init(locals())
        if self.opt_product is None:
            self.opt_product = fom.opt_product
        super().__init__(fom, RBPrimal, product=opt_product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol, coercivity_estimator=coercivity_estimator)

        self.non_assembled_primal_reductor = None
        self.non_assembled_primal_rom = None
        self.non_assembled_dual_rom = None

        self.adjoint_approach = self.fom.adjoint_approach
        self.separated_bases = False if (self.unique_basis or RBPrimalSensitivity is None) else True
        if self.fom.adjoint_approach and true_lagrange is True and self.fom.use_corrected_gradient:
            assert RBPrimalSensitivity is not None, 'I need the FOM sensitivities for the true Lagrangian case in the adjoint approach'
        if self.fom.adjoint_approach and self.separated_bases and true_lagrange is False:
            assert 0, 'I do NOT use the FOM sensitivities that you gave to me as long the true Lagrangian case in the adjoint approach is switched off'

        if RBPrimalSensitivity is not None and self.unique_basis is False:
            self.bases = {'RB' : RBPrimal, 'DU': RBDual, 'PrimalSens': RBPrimalSensitivity, 'DualSens': RBDualSensitivity}
            print('Starting with separated sensitivity bases. ', end='')
            print('Primal and dual have length {} and {}'.format(
                len(RBPrimal), len(RBDual))) if RBPrimal is not None and RBDual is not None else print('The Primal and/or the  dual bases are empty')
        elif RBPrimalSensitivity is not None and unique_basis is True:
            self._build_unique_basis()
            self.bases = {'RB' : self.RBPrimal}
            print('Starting with only one basis with length {}'.format(len(self.RBPrimal)))
        elif RBPrimalSensitivity is None and unique_basis is True:
            self._build_unique_basis()
            self.bases = {'RB' : self.RBPrimal}
            print('Starting with only one basis with length {}'.format(len(self.RBPrimal)))
        else:
            self.bases = {'RB' : RBPrimal, 'DU': RBDual}
            print('Starting with two bases. ', end='')
            print('Primal and dual have length {} and {}'.format(
                len(RBPrimal), len(RBDual))) if RBPrimal is not None and RBDual is not None else print('The Primal and/or the dual bases are empty')
        
        # primal model
        self.primal_fom = self.fom.primal_model
        self.primal_rom, self.primal_reductor = self._build_primal_rom()
        self.primal = self.primal_reductor
        
        # dual model
        if self.RBPrimal is not None:
            self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
            self.dual = self.dual_reductor
        
        self.use_corrected_gradient = self.fom.use_corrected_gradient
        
        if self.use_corrected_gradient is True and self.RBPrimalSensitivity is None:
            self.RBPrimalSensitivity = self.RBPrimal
            self.RBDualSensitivity = self.RBDual
        
        if self.separated_bases and self.fom.use_corrected_gradient:
            self.primal_sensitivity_fom, self.primal_sensitivity_rom, self.primal_sensitivity_reductor = self._build_primal_sensitivity_models()
            self.primal_sensitivity = self.primal_sensitivity_reductor
        elif self.separated_bases is False and (self.prepare_for_sensitivity_estimate or self.fom.use_corrected_gradient or (not self.adjoint_estimate and self.fom.adjoint_approach)):
            if self.prepare_for_gradient_estimate or self.prepare_for_sensitivity_estimate:
                self.primal_sensitivity_fom, self.primal_sensitivity_rom, self.primal_sensitivity_reductor = self._build_primal_sensitivity_model_for_all_directions()
                self.primal_sensitivity = self.primal_reductor
            else:
                self.primal_sensitivity_rom = None
        else:
            self.primal_sensitivity_rom = None
        
        if self.separated_bases and self.fom.use_corrected_gradient:
            self.dual_sensitivity_fom, self.dual_sensitivity_rom, self.dual_sensitivity_reductor = self._build_dual_sensitivity_models()
            self.dual_sensitivity = self.dual_sensitivity_reductor
        elif self.separated_bases is False and (self.prepare_for_sensitivity_estimate or self.fom.use_corrected_gradient or (not self.adjoint_estimate and self.fom.adjoint_approach)):
            if self.prepare_for_gradient_estimate or self.prepare_for_sensitivity_estimate:
                self.dual_sensitivity_fom, self.dual_sensitivity_rom, self.dual_sensitivity_reductor = self._build_dual_sensitivity_model_for_all_directions()
                self.dual_sensitivity = self.dual_reductor
            else:
                self.dual_sensitivity_rom = None
        else:
            self.dual_sensitivity_rom = None

        # pre compute constants for estimators
        k_form = self.fom.output_functional_dict['bilinear_part']
        if isinstance(k_form, LincombOperator) and self.fin_model is False:
            alpha_mu_bar = self.fom.compute_continuity_bilinear(k_form, self.fom.opt_product, mu_bar)
            self.cont_k = MaxThetaParameterFunctional(k_form.coefficients, mu_bar, gamma_mu_bar=alpha_mu_bar)
        else:
            self.cont_k = lambda mu: self.fom.compute_continuity_bilinear(k_form, self.fom.opt_product)

        j_form = self.fom.output_functional_dict['linear_part']
        if isinstance(j_form, LincombOperator) and self.fin_model is False:
            conts_j = []
            for op in j_form.operators:
                conts_j.append(self.fom.compute_continuity_linear(op, self.fom.opt_product))
            self.cont_j = lambda mu: np.dot(conts_j,np.abs(j_form.evaluate_coefficients(mu)))
        else:
            self.cont_j = lambda mu: self.fom.compute_continuity_linear(j_form, self.fom.opt_product)

        if self.prepare_for_gradient_estimate or self.prepare_for_sensitivity_estimate:
            self.cont_j_dmu = self._construct_zero_dict(self.primal_fom.parameter_type)
            self.cont_k_dmu = self.cont_j_dmu.copy()
            for (key, item) in self.primal_fom.parameter_type.items():
                index_dict, _ = self.fom._collect_indices(item)
                for (l, index) in index_dict.items():
                    if isinstance(k_form, LincombOperator) and self.fin_model is False:
                        k_dmu = k_form.d_mu(key, index)
                        alpha_mu_bar = self.fom.compute_continuity_bilinear(k_dmu, self.fom.opt_product, mu_bar)
                        if alpha_mu_bar == 0:
                            self.cont_k_dmu[key][index] = lambda mu: 0
                        else:
                            self.cont_k_dmu[key][index] = MaxThetaParameterFunctional(k_dmu.coefficients, mu_bar, gamma_mu_bar=alpha_mu_bar)
                    else:
                        self.cont_k_dmu[key][index] = lambda mu: 0
                    if isinstance(j_form, LincombOperator) and self.fin_model is False:
                        j_dmu = j_form.d_mu(key, index)
                        conts_j = []
                        for op in j_dmu.operators:
                            conts_j.append(self.fom.compute_continuity_linear(op, self.fom.opt_product))
                        self.cont_j_dmu[key][index] = lambda mu: np.dot(conts_j,np.abs(j_dmu.evaluate_coefficients(mu)))
                    else:
                        self.cont_j_dmu[key][index] = lambda mu: 0

        if self.coercivity_estimator is None:
            print('WARNING: coercivity_estimator is None ... setting it to constant 1.')
            self.coercivity_estimator = lambda mu: 1.

        # precompute ||d_mui l_h || 
        if self.prepare_for_gradient_estimate or self.prepare_for_sensitivity_estimate:
            self.cont_l_dmu = self._construct_zero_dict(self.primal_fom.parameter_type)
            for (key, item) in self.primal_fom.parameter_type.items():
                index_dict, _ = self.fom._collect_indices(item)
                for (l, index) in index_dict.items():
                    conts_l = []
                    l_dmu = self.fom.primal_model.rhs.d_mu(key, index)
                    for op in l_dmu.operators:
                        conts_l.append(self.fom.compute_continuity_linear(op, self.fom.opt_product))
                        self.cont_l_dmu[key][index] = lambda mu: np.dot(conts_l,np.abs(l_dmu.evaluate_coefficients(mu)))

        # precomput parts of || d_mui a_h ||
        if self.prepare_for_gradient_estimate or self.prepare_for_sensitivity_estimate:
            self.cont_a_dmu_functional = self._construct_zero_dict(self.primal_fom.parameter_type)
            for (key, item) in self.primal_fom.parameter_type.items():
                index_dict, _ = self.fom._collect_indices(item)
                for (l, index) in index_dict.items():
                    self.cont_a_dmu_functional[key][index] = BaseMaxThetaParameterFunctional( \
                                                                 self.fom.primal_model.operator.d_mu(key, index).coefficients,
                                                                 self.fom.primal_model.operator.coefficients, mu_bar)
        
        self.cont_a = MaxThetaParameterFunctional(self.primal_fom.operator.coefficients, mu_bar)
        
        self.second_primal_sensitivity_rom = None
        self.second_dual_sensitivity_rom = None        

    def reduce(self):
        assert self.RBPrimal is not None, 'I can not reduce without a RB basis'
        return super().reduce()

    def build_rom(self, projected_operators, estimator):
        projected_product = self.project_product()
        return self.fom.with_(primal_model=self.primal_rom, dual_model=self.dual_rom,
                              primal_sensitivity_model=self.primal_sensitivity_rom,
                              dual_sensitivity_model=self.dual_sensitivity_rom,
                              opt_product=projected_product,
                              estimators=estimator, output_functional_dict=self.projected_output,
                              use_corrected_gradient=self.use_corrected_gradient,
                              separated_bases=self.separated_bases,
                              true_lagrange=self.true_lagrange,
                              second_primal_sens_model=self.second_primal_sensitivity_rom,
                              second_dual_sens_model=self.second_dual_sensitivity_rom)

    def extend_bases(self, mu, printing=True):
        if self.unique_basis:
            u, p = self.extend_unique_basis(mu)
            return u, p
        
        u = self.fom.solve(mu)
        p = self.fom.solve_dual(mu,U=u)
        try:
            self.primal_reductor.extend_basis(u)
            self.non_assembled_primal_reductor.extend_basis(u)
        except:
            pass
        self.primal_rom = self.primal_reductor.reduce()
        if self.non_assembled_primal_rom is not None:
            self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()
        self.bases['RB'] = self.primal_reductor.bases['RB']
        self.RBPrimal = self.bases['RB']
        self.RBDual.append(p)
        self.RBDual = gram_schmidt(self.RBDual, product=self.opt_product)
        an, bn = len(self.RBPrimal), len(self.RBDual)
        self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
        self.dual = self.dual_reductor
        self.bases['DU'] = self.dual_reductor.bases['RB']
        cn = []
        dn = []
        if self.separated_bases and self.fom.use_corrected_gradient:
            for (key, item) in self.fom.parameter_space.parameter_type.items():
                index_dict, _ = self.fom._collect_indices(item)
                for (l, index) in index_dict.items():
                    u_d_mu = self.fom.solve_for_u_d_mu(key, index, mu, u)
                    p_d_mu = self.fom.solve_for_p_d_mu(key, index, mu, u, p, u_d_mu)
                    self.RBPrimalSensitivity[key][index].append(u_d_mu)
                    self.RBPrimalSensitivity[key][index] = gram_schmidt(self.RBPrimalSensitivity[key][index], product=self.opt_product)
                    self.RBDualSensitivity[key][index].append(p_d_mu)
                    self.RBDualSensitivity[key][index] = gram_schmidt(self.RBDualSensitivity[key][index], product=self.opt_product)
                    cn.append(len(self.RBPrimalSensitivity[key][index]))
                    dn.append(len(self.RBDualSensitivity[key][index]))
                    self.bases['PrimalSens'][key][index] = self.RBPrimalSensitivity[key][index]
                    self.bases['DualSens'][key][index] = self.RBDualSensitivity[key][index]
            self.primal_sensitivity_fom, self.primal_sensitivity_rom, self.primal_sensitivity_reductor = self._build_primal_sensitivity_models()
            self.primal_sensitivity = self.primal_sensitivity_reductor
            self.dual_sensitivity_fom, self.dual_sensitivity_rom, self.dual_sensitivity_reductor = self._build_dual_sensitivity_models()
            self.dual_sensitivity = self.dual_sensitivity_reductor
        elif self.separated_bases is False and (self.prepare_for_sensitivity_estimate or self.fom.use_corrected_gradient or (not self.adjoint_estimate and self.fom.adjoint_approach)):
            if self.prepare_for_gradient_estimate or self.prepare_for_sensitivity_estimate:
                self.primal_sensitivity_fom, self.primal_sensitivity_rom, self.primal_sensitivity_reductor = self._build_primal_sensitivity_model_for_all_directions()
                self.primal_sensitivity = self.primal_reductor
                self.dual_sensitivity_fom, self.dual_sensitivity_rom, self.dual_sensitivity_reductor = self._build_dual_sensitivity_model_for_all_directions()
                self.dual_sensitivity = self.dual_reductor

        if printing:
            print('Enrichment completed... length of Bases are {}, {}, {} and {}'.format(an,bn,cn,dn))
        return u, p


    def extend_unique_basis(self,mu):
        assert self.unique_basis
        U = self.fom.solve(mu=mu)
        P = self.fom.solve_dual(mu=mu, U=U)
        try:
            self.primal_reductor.extend_basis(U)
            self.non_assembled_primal_reductor.extend_basis(U)
        except:
            pass
        try:
            self.primal_reductor.extend_basis(P)
            self.non_assembled_primal_reductor.extend_basis(P)
        except:
            pass
        if self.RBPrimalSensitivity is not None:
            for (key, item) in self.fom.parameter_space.parameter_type.items():
                index_dict, _ = self.fom._collect_indices(item)
                for (l, index) in index_dict.items():
                    u_d_mu = self.fom.solve_for_u_d_mu(key, index, mu=mu, U=U)
                    p_d_mu = self.fom.solve_for_p_d_mu(key, index, mu=mu, U=U, P=P, u_d_mu=u_d_mu)
                    try:
                        self.primal_reductor.extend_basis(u_d_mu)
                        self.non_assembled_primal_reductor.extend_basis(u_d_mu)
                    except:
                        pass
                    try:
                        self.primal_reductor.extend_basis(p_d_mu)
                        self.non_assembled_primal_reductor.extend_basis(p_d_mu)
                    except:
                        pass
            self.primal_rom = self.primal_reductor.reduce()
            if self.non_assembled_primal_rom is not None:
                self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()
            self.bases['RB'] = self.primal_reductor.bases['RB']
            self.RBPrimal = self.bases['RB']

            self.RBDual = self.RBPrimal
            self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
            self.dual = self.primal_reductor

            self.RBPrimalSensitivity = self.RBPrimal
            self.RBDualSensitivity = self.RBDual

            if self.prepare_for_gradient_estimate or self.prepare_for_sensitivity_estimate:
                self.primal_sensitivity_fom, self.primal_sensitivity_rom, self.primal_sensitivity_reductor = self._build_primal_sensitivity_model_for_all_directions()
                self.primal_sensitivity = self.primal_reductor
                self.dual_sensitivity_fom, self.dual_sensitivity_rom, self.dual_sensitivity_reductor = self._build_dual_sensitivity_model_for_all_directions()
                self.dual_sensitivity = self.dual_reductor
                
            an = len(self.RBPrimal)
            print('Length of Basis is {}'.format(an))
        else:
            self.primal_rom = self.primal_reductor.reduce()
            if self.non_assembled_primal_rom is not None:
                self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()
            self.bases['RB'] = self.primal_reductor.bases['RB']
            self.RBPrimal = self.bases['RB']

            self.RBDual = self.RBPrimal
            self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
            self.dual = self.primal_reductor
            
            if self.prepare_for_gradient_estimate or self.prepare_for_sensitivity_estimate:
                self.primal_sensitivity_fom, self.primal_sensitivity_rom, self.primal_sensitivity_reductor = self._build_primal_sensitivity_model_for_all_directions()
                self.primal_sensitivity = self.primal_reductor
                self.dual_sensitivity_fom, self.dual_sensitivity_rom, self.dual_sensitivity_reductor = self._build_dual_sensitivity_model_for_all_directions()
                self.dual_sensitivity = self.dual_reductor
            
            an = len(self.RBPrimal)
            print('Length of Basis is {}'.format(an))
        return U, P
    
    def _build_unique_basis(self):
        if self.RBPrimalSensitivity is not None:
            for component, item in self.fom.parameter_space.parameter_type.items():
                component_bases_pr = self.RBPrimalSensitivity[component]
                component_bases_du = self.RBDualSensitivity[component]
                index_dict, _ = self.fom._collect_indices(item)
                if sum(item) < 2:
                    self.RBPrimal.append(component_bases_pr[()])
                    self.RBDual.append(component_bases_du[()])
                else:
                    for basis in component_bases_pr:
                        self.RBPrimal.append(basis)
                    for basis in component_bases_du:
                        self.RBDual.append(basis)
            self.RBPrimal.append(self.RBDual)
            self.RBPrimal = gram_schmidt(self.RBPrimal, product=self.opt_product)
            self.RBDual = self.RBPrimal
            self.RBPrimalSensitivity = self.RBPrimal
            self.RBDualSensitivity = self.RBDual
        else:
            self.RBPrimal.append(self.RBDual)
            self.RBPrimal = gram_schmidt(self.RBPrimal, product=self.opt_product)
            self.RBDual = self.RBPrimal

    def _build_primal_rom(self):
        primal_reductor = SimpleCoerciveRBReductor(self.fom.primal_model, RB=self.RBPrimal, product=self.opt_product,
                                            coercivity_estimator=self.coercivity_estimator)
        self.non_assembled_primal_reductor = NonAssembledCoerciveRBReductor(self.fom.primal_model, RB=self.RBPrimal, product=self.opt_product,
                                                        coercivity_estimator=self.coercivity_estimator)
        self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()

        primal_rom = primal_reductor.reduce()
        return primal_rom, primal_reductor

    def _build_dual_models(self):
        assert self.primal_rom is not None
        assert self.RBPrimal is not None
        RBbasis = self.RBPrimal
        rhs_operators = list(self.fom.output_functional_dict['d_u_linear_part'].operators)
        rhs_coefficients = list(self.fom.output_functional_dict['d_u_linear_part'].coefficients)

        bilinear_part = self.fom.output_functional_dict['d_u_bilinear_part']

        for i in range(len(RBbasis)):
            u = RBbasis[i]
            if isinstance(bilinear_part, LincombOperator) and self.fin_model is False:
                for j, op in enumerate(bilinear_part.operators):
                    rhs_operators.append(VectorOperator(op.apply(u)))
                    rhs_coefficients.append(ExpressionParameterFunctional('basis_coefficients[{}]'.format(i),
                                                                  {'basis_coefficients': (len(RBbasis),)})
                                        * bilinear_part.coefficients[j])
            else:
                rhs_operators.append(VectorOperator(bilinear_part.apply(u, None)))
                rhs_coefficients.append(1. * ExpressionParameterFunctional('basis_coefficients[{}]'.format(i),
                                                                  {'basis_coefficients': (len(RBbasis),)}))

        dual_rhs_operator = LincombOperator(rhs_operators,rhs_coefficients)

        dual_intermediate_fom = self.fom.primal_model.with_(rhs = dual_rhs_operator,
                                   parameter_space=None)

        dual_reductor = SimpleCoerciveRBReductor(dual_intermediate_fom, RB=self.RBDual,
                                           product=self.opt_product,
                                           coercivity_estimator=self.coercivity_estimator)
        non_assembled_reductor = NonAssembledCoerciveRBReductor(dual_intermediate_fom, RB=self.RBDual,
                                            product=self.opt_product, coercivity_estimator=self.coercivity_estimator)
        self.non_assembled_dual_rom = non_assembled_reductor.reduce()

        dual_rom = dual_reductor.reduce()
        return dual_intermediate_fom, dual_rom, dual_reductor
    
    def _build_primal_sensitivity_models(self):
        print('building MULTIPLE sensitivity models for {} directions...'.format(self.fom.number_of_parameters))
        assert self.primal_rom is not None
        assert self.RBPrimal is not None

        RBbasis = self.RBPrimal
        primal_sensitivity_fom_dict = {}
        primal_sensitivity_rom_dict = {}
        primal_sensitivity_reductor_dict = {}
        for (key,item) in self.primal_fom.parameter_space.parameter_type.items():
            index_dict, new_item = self.fom._collect_indices(item)
            array_fom = np.empty(new_item, dtype=object)
            array_reductor = np.empty(new_item, dtype=object)
            array_rom = np.empty(new_item, dtype=object)
            for (l, index) in index_dict.items():
                if self.unique_basis:
                    SensitivityBasis = self.RBPrimal
                else:
                    SensitivityBasis = self.RBPrimalSensitivity[key][index]
                op_d_mu = self.primal_fom.rhs.d_mu(key, index)
                rhs_operators = op_d_mu.operators
                rhs_coefficients = op_d_mu.coefficients
                for i in range(len(RBbasis)):
                    u = RBbasis[i]
                    op = self.primal_fom.operator.d_mu(key,index)
                    operator = op.with_(operators=[VectorOperator(o.apply(u)) for o in op.operators])
                    rhs_operators += operator.operators
                    for l in range(len(operator.operators)):
                        rhs_coefficients += (operator.coefficients[l] * ExpressionParameterFunctional('-basis_coefficients[{}]'.format(i),
                                                                              {'basis_coefficients': (len(RBbasis),)}),)

                sensitivity_rhs_operator = LincombOperator(rhs_operators,rhs_coefficients)
                primal_sensitivity_fom = self.primal_fom.with_(rhs = sensitivity_rhs_operator,
                                                                        parameter_space=None)

                if self.global_greedy is True:
                    primal_sensitivity_reductor = SimpleCoerciveRBReductor(primal_sensitivity_fom, RB=SensitivityBasis,
                                                                         product=self.opt_product)
                else:
                    primal_sensitivity_reductor = NonAssembledCoerciveRBReductor(primal_sensitivity_fom, RB=SensitivityBasis,
                                                                         product=self.opt_product)

                primal_sensitivity_rom = primal_sensitivity_reductor.reduce()

                array_fom[index] = primal_sensitivity_fom
                array_reductor[index] = primal_sensitivity_reductor
                array_rom[index] = primal_sensitivity_rom
            primal_sensitivity_fom_dict[key] = array_fom
            primal_sensitivity_reductor_dict[key] = array_reductor
            primal_sensitivity_rom_dict[key] = array_rom

        return primal_sensitivity_fom_dict, primal_sensitivity_rom_dict, primal_sensitivity_reductor_dict
    
    def _build_primal_sensitivity_model_for_all_directions(self):
        print('building a SINGLE sensitivtiy model for any direction...')
        assert self.primal_rom is not None
        assert self.RBPrimal is not None
        # assert self.fom.separated_bases is False

        RBbasis = self.RBPrimal
        SensitivityBasis = self.RBPrimal
        rhs_operators = ()
        rhs_coefficients = ()
        k = 0
        for (key,item) in self.primal_fom.parameter_space.parameter_type.items():
            index_dict, new_item = self.fom._collect_indices(item)
            for (l, index) in index_dict.items():
                op_d_mu = self.primal_fom.rhs.d_mu(key, index)
                rhs_operators += op_d_mu.operators
                factor = ProjectionParameterFunctional('eta', (self.fom.number_of_parameters,), (k,))
                rhs_coefficients += tuple([factor * op_ for op_ in op_d_mu.coefficients])
                for i in range(len(RBbasis)):
                    u = RBbasis[i]
                    op = self.primal_fom.operator.d_mu(key,index)
                    operator = op.with_(operators=[VectorOperator(o.apply(u)) for o in op.operators])
                    rhs_operators += operator.operators
                    for l in range(len(operator.operators)):
                        rhs_coefficients += (operator.coefficients[l] * ExpressionParameterFunctional('-basis_coefficients[{}]'.format(i),
                                                        {'basis_coefficients': (len(RBbasis),)}) * factor,)
                k += 1

        sensitivity_rhs_operator = LincombOperator(rhs_operators,rhs_coefficients)
        primal_sensitivity_fom = self.primal_fom.with_(rhs = sensitivity_rhs_operator,
                                                                parameter_space=None)

        if self.global_greedy is True:
            primal_sensitivity_reductor = SimpleCoerciveRBReductor(primal_sensitivity_fom, RB=SensitivityBasis,
                                                                 product=self.opt_product)
        else:
            primal_sensitivity_reductor = NonAssembledCoerciveRBReductor(primal_sensitivity_fom, RB=SensitivityBasis,
                                                                 product=self.opt_product)

        primal_sensitivity_rom = primal_sensitivity_reductor.reduce()

        return primal_sensitivity_fom, primal_sensitivity_rom, primal_sensitivity_reductor

    def _build_dual_sensitivity_models(self):
        # print('build_dual_sens_models')
        assert self.primal_rom is not None
        assert self.RBPrimal is not None
        assert self.RBDual is not None
        assert self.RBPrimalSensitivity is not None

        RBDual = self.RBDual
        RBPrimal = self.RBPrimal
        RBSens = self.RBPrimalSensitivity
        dual_sensitivity_fom_dict = {}
        dual_sensitivity_rom_dict = {}
        dual_sensitivity_reductor_dict = {}
        d_u_bilinear_part = self.fom.output_functional_dict['d_u_bilinear_part']
        d_u_linear_part = self.fom.output_functional_dict['d_u_linear_part']
        for (key,item) in self.primal_fom.parameter_space.parameter_type.items():
            index_dict, new_item = self.fom._collect_indices(item)
            array_fom = np.empty(new_item, dtype=object)
            array_reductor = np.empty(new_item, dtype=object)
            array_rom = np.empty(new_item, dtype=object)
            for (l, index) in index_dict.items():
                if self.unique_basis:
                    SensitivityBasis = self.RBDual
                else:
                    SensitivityBasis = self.RBDualSensitivity[key][index]
                rhs_operators = (ZeroOperator(d_u_linear_part.range, d_u_linear_part.source),)
                rhs_coefficients = (0,)
                # dual residual d_mu part
                for i in range(len(RBDual)):
                    p = RBDual[i]
                    op = self.primal_fom.operator.d_mu(key,index)
                    operator = op.with_(operators=[VectorOperator(o.apply(p)) for o in op.operators])
                    rhs_operators += operator.operators
                    for l in range(len(operator.operators)):
                        rhs_coefficients += (operator.coefficients[l] * ExpressionParameterFunctional('-basis_coefficients_dual[{}]'.format(i),
                                                                          {'basis_coefficients_dual': (len(RBDual),)}),)
                if isinstance(d_u_bilinear_part, LincombOperator) and self.fin_model is False:
                    for i in range(len(RBPrimal)):
                        u = RBPrimal[i]
                        k_op = d_u_bilinear_part.d_mu(key,index)
                        k_operator = k_op.with_(operators=[VectorOperator(o.apply(u)) for o in k_op.operators])
                        rhs_operators += k_operator.operators
                        for l in range(len(k_operator.operators)):
                            rhs_coefficients += (k_operator.coefficients[l] *
                                                 ExpressionParameterFunctional('basis_coefficients[{}]'.format(i),
                                                                              {'basis_coefficients':
                                                                               (len(RBPrimal),)}),)
                if isinstance(d_u_linear_part, LincombOperator) and self.fin_model is False:
                    j_op = d_u_linear_part.d_mu(key,index)
                    rhs_operators += j_op.operators
                    for l in range(len(j_op.operators)):
                       rhs_coefficients += (j_op.coefficients[l],)

                # 2k(q, d_mu_u) part
                if self.unique_basis:
                    for i in range(len(RBSens)):
                        u_d = RBSens[i]
                        if isinstance(d_u_bilinear_part, LincombOperator) and self.fin_model is False:
                            for j, op in enumerate(d_u_bilinear_part.operators):
                                rhs_operators += (VectorOperator(op.apply(u_d)),)
                                rhs_coefficients += (ExpressionParameterFunctional('basis_coefficients_primal_sens[{}]'.format(i),
                                                                  {'basis_coefficients_primal_sens': (len(RBSens),)})
                                                    * d_u_bilinear_part.coefficients[j],)
                        else:
                            rhs_operators += (VectorOperator(d_u_bilinear_part.apply(u_d), None),)
                            rhs_coefficients += (ExpressionParameterFunctional('basis_coefficients_primal_sens[{}]'.format(i),
                                                                              {'basis_coefficients_primal_sens':
                                                                               (len(RBSens),)}),)
                else:
                    for i in range(len(RBSens[key][index])):
                        u_d = RBSens[key][index][i]
                        if isinstance(d_u_bilinear_part, LincombOperator) and self.fin_model is False:
                            for j, op in enumerate(d_u_bilinear_part.operators):
                                rhs_operators += (VectorOperator(op.apply(u_d)),)
                                rhs_coefficients += (ExpressionParameterFunctional('basis_coefficients_primal_sens[{}]'.format(i),
                                                                  {'basis_coefficients_primal_sens':
                                                                   (len(RBSens[key][index]),)}) * d_u_bilinear_part.coefficients[j],)
                        else:
                            rhs_operators += (VectorOperator(d_u_bilinear_part.apply(u_d, None)),)
                            rhs_coefficients += (ExpressionParameterFunctional('basis_coefficients_primal_sens[{}]'.format(i),
                                                                              {'basis_coefficients_primal_sens':
                                                                               (len(RBSens[key][index]),)}),)

                sensitivity_rhs_operator = LincombOperator(rhs_operators,rhs_coefficients)

                dual_sensitivity_fom = self.primal_fom.with_(rhs = sensitivity_rhs_operator,
                                                                        parameter_space=None)
                if self.global_greedy is True:
                    dual_sensitivity_reductor = SimpleCoerciveRBReductor(dual_sensitivity_fom, RB=SensitivityBasis,
                                                                     product=self.opt_product)
                else:
                    dual_sensitivity_reductor = NonAssembledCoerciveRBReductor(dual_sensitivity_fom, RB=SensitivityBasis,
                                                                     product=self.opt_product)

                dual_sensitivity_rom = dual_sensitivity_reductor.reduce()

                array_fom[index] = dual_sensitivity_fom
                array_reductor[index] = dual_sensitivity_reductor
                array_rom[index] = dual_sensitivity_rom
            dual_sensitivity_fom_dict[key] = array_fom
            dual_sensitivity_reductor_dict[key] = array_reductor
            dual_sensitivity_rom_dict[key] = array_rom

        return dual_sensitivity_fom_dict, dual_sensitivity_rom_dict, dual_sensitivity_reductor_dict
    
    def _build_dual_sensitivity_model_for_all_directions(self):
        assert self.primal_rom is not None
        assert self.RBPrimal is not None
        assert self.RBDual is not None

        RBDual = self.RBDual
        RBPrimal = self.RBPrimal
        RBSens = self.RBPrimal
        SensitivityBasis = self.RBDual
        
        d_u_bilinear_part = self.fom.output_functional_dict['d_u_bilinear_part']
        d_u_linear_part = self.fom.output_functional_dict['d_u_linear_part']
        
        rhs_operators = (ZeroOperator(d_u_linear_part.range,d_u_linear_part.source),)
        rhs_coefficients = (0.,)
        k = 0
        
        for (key,item) in self.primal_fom.parameter_space.parameter_type.items():
            index_dict, new_item = self.fom._collect_indices(item)
            for (l, index) in index_dict.items():
                factor = ProjectionParameterFunctional('eta', (self.fom.number_of_parameters,), (k,))
                # dual residual d_mu part
                for i in range(len(RBDual)):
                    p = RBDual[i]
                    op = self.primal_fom.operator.d_mu(key,index)
                    operator = op.with_(operators=[VectorOperator(o.apply(p)) for o in op.operators])
                    rhs_operators += operator.operators
                    for l in range(len(operator.operators)):
                        rhs_coefficients += (operator.coefficients[l] * ExpressionParameterFunctional('-basis_coefficients_dual[{}]'.format(i),
                                                    {'basis_coefficients_dual': (len(RBDual),)}) * factor,)
                if isinstance(d_u_bilinear_part, LincombOperator) and self.fin_model is False:
                    for i in range(len(RBPrimal)):
                        u = RBPrimal[i]
                        k_op = d_u_bilinear_part.d_mu(key,index)
                        k_operator = k_op.with_(operators=[VectorOperator(o.apply(u)) for o in k_op.operators])
                        rhs_operators += k_operator.operators
                        for l in range(len(k_operator.operators)):
                            rhs_coefficients += (k_operator.coefficients[l] *
                                                 ExpressionParameterFunctional('basis_coefficients[{}]'.format(i),
                                                                              {'basis_coefficients':
                                                                               (len(RBPrimal),)}) * factor,)
                if isinstance(d_u_linear_part, LincombOperator) and self.fin_model is False:
                    j_op = d_u_linear_part.d_mu(key,index)
                    rhs_operators += j_op.operators
                    for l in range(len(j_op.operators)):
                       rhs_coefficients += (j_op.coefficients[l] * factor,)
                k += 1

        # 2k(q, d_mu_u) part
        for i in range(len(RBSens)):
            u_d = RBSens[i]
            if isinstance(d_u_bilinear_part, LincombOperator) and self.fin_model is False:
                for j, op in enumerate(d_u_bilinear_part.operators):
                    rhs_operators += (VectorOperator(op.apply(u_d)),)
                    rhs_coefficients += (ExpressionParameterFunctional('basis_coefficients_primal_sens[{}]'.format(i),
                                                      {'basis_coefficients_primal_sens': (len(RBSens),)})
                                        * d_u_bilinear_part.coefficients[j],)
            else:
                rhs_operators += (VectorOperator(d_u_bilinear_part.apply(u_d), None),)
                rhs_coefficients += (ExpressionParameterFunctional('basis_coefficients_primal_sens[{}]'.format(i),
                                                                  {'basis_coefficients_primal_sens':
                                                                   (len(RBSens),)}),)
        sensitivity_rhs_operator = LincombOperator(rhs_operators,rhs_coefficients)

        dual_sensitivity_fom = self.primal_fom.with_(rhs = sensitivity_rhs_operator,
                                                                parameter_space=None)
        if self.global_greedy is True:
            dual_sensitivity_reductor = SimpleCoerciveRBReductor(dual_sensitivity_fom, RB=SensitivityBasis,
                                                             product=self.opt_product)
        else:
            dual_sensitivity_reductor = NonAssembledCoerciveRBReductor(dual_sensitivity_fom, RB=SensitivityBasis,
                                                             product=self.opt_product)

        dual_sensitivity_rom = dual_sensitivity_reductor.reduce()

        return dual_sensitivity_fom, dual_sensitivity_rom, dual_sensitivity_reductor

    def _construct_zero_dict(self, parameter_type):
        #prepare dict
        zero_dict = {}
        for key, item in parameter_type.items():
            _, new_item = self.fom._collect_indices(item)
            zero_ = np.empty(new_item, dtype=object)
            zero_dict[key] = zero_
        return zero_dict

    #prepare dict
    def _construct_zero_dict_dict(self, parameter_type):
        zero_dict = {}
        for key, item in parameter_type.items():
            index_dict, new_item = self.fom._collect_indices(item)
            zero_ = np.empty(new_item, dtype=dict)
            zero_dict[key] = zero_
            for (l, index) in index_dict.items():
                zero_dict[key][index] = self._construct_zero_dict(parameter_type)
        return zero_dict

    def assemble_estimator(self):
        # I need the output in advance
        self.projected_output = self.project_output()
        
        # print_pieces 
        print_pieces = 0

        estimators = {}

        # primal
        class PrimalCoerciveRBEstimator(ImmutableObject):
            def __init__(self, primal_rom, non_assembled_rom=None):
                self.__auto_init(locals())
            def estimate(self, U, mu, non_assembled=False):
                if non_assembled and self.non_assembled_rom is not None:
                    return self.non_assembled_rom.estimate(U, mu)
                else:
                    return self.primal_rom.estimate(U, mu)

        estimators['primal'] = PrimalCoerciveRBEstimator(self.primal_rom, self.non_assembled_primal_rom)

        ##########################################

        # dual
        class DualCoerciveRBEstimator(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_k, primal_estimator, dual_rom, non_assembled_rom=None):
                self.__auto_init(locals())

            def estimate(self, U, P, mu, non_assembled=False):
                primal_estimate = self.primal_estimator.estimate(U, mu, non_assembled=non_assembled)[0]
                if non_assembled and self.non_assembled_rom is not None:
                    dual_intermediate_estimate = self.non_assembled_rom.estimate(P, mu)[0]
                else:
                    dual_intermediate_estimate = self.dual_rom.estimate(P, mu)
                if print_pieces or 0:
                    print(self.cont_k(mu), self.coercivity_estimator(mu), primal_estimate, dual_intermediate_estimate)
                return 2* self.cont_k(mu) /self.coercivity_estimator(mu) * primal_estimate + dual_intermediate_estimate

        estimators['dual'] = DualCoerciveRBEstimator(self.coercivity_estimator, self.cont_k, estimators['primal'], self.dual_rom, self.non_assembled_dual_rom)
        ##########################################

        # output hat
        class output_hat_RBEstimator(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_k, cont_j, primal_estimator, dual_estimator,
                         projected_output, dual_rom, P_product, U_product, corrected_output):
                self.__auto_init(locals())

            def estimate(self, U, P, mu, residual_based=True, both_estimators=False):
                if residual_based:
                    primal_estimate = self.primal_estimator.estimate(U, mu)[0]
                    dual_estimate = self.dual_estimator.estimate(U, P, mu)

                    residual_lhs = self.projected_output['primal_dual_projected_op'].apply2(U, P, mu=mu)[0][0]
                    residual_rhs = self.projected_output['dual_projected_rhs'].apply_adjoint(P, mu=mu).to_numpy()[0][0]

                    if print_pieces or 0:
                        print(self.coercivity_estimator(mu), primal_estimate, dual_estimate, primal_estimate**2,
                              self.cont_k(mu), primal_estimate, self.coercivity_estimator(mu))
                    if both_estimators:
                        est1 = self.coercivity_estimator(mu) * primal_estimate * dual_estimate + \
                           primal_estimate**2 * self.cont_k(mu)
                        est2 = est1 + np.abs(residual_rhs - residual_lhs)
                        if self.corrected_output:
                            return [est1, est2]
                        else:
                            return [est2, est1]
                
                    if self.corrected_output:
                        est1 = self.coercivity_estimator(mu) * primal_estimate * dual_estimate + \
                           primal_estimate**2 * self.cont_k(mu)
                        return est1
                    else:
                        est2 = self.coercivity_estimator(mu) * primal_estimate * dual_estimate + \
                           primal_estimate**2 * self.cont_k(mu) + \
                           + np.abs(residual_rhs - residual_lhs)
                        return est2
                else:
                    primal_estimate = self.primal_estimator.estimate(U, mu)[0]

                    norm_U = np.sqrt(self.U_product.apply2(U, U))[0][0]

                    if print_pieces or 0:
                        print(primal_estimate, self.cont_j(mu), self.cont_k(mu), norm_U, primal_estimate)
                    return primal_estimate * ( self.cont_j(mu) + self.cont_k(mu) * \
                                              (2 * norm_U + primal_estimate))


        estimators['output_functional_hat'] = output_hat_RBEstimator(self.coercivity_estimator,
                                                                     self.cont_k, self.cont_j,
                                                                     estimators['primal'], estimators['dual'],
                                                                     self.projected_output, self.dual_rom, self.dual_rom.opt_product,
                                                                     self.primal_rom.opt_product,
                                                                     self.fom.use_corrected_functional)


        ##########################################
        estimators['u_d_mu'] = None
        estimators['p_d_mu'] = None
        estimators['output_functional_hat_d_mus'] = None
        
        # sensitivity_u
        class u_d_mu_RBEstimator(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_a_dmu_functional, primal_estimator, primal_rom, primal_product,
                         primal_sensitivity_product, product, sens_rom, SensBasis, component, index, opt_fom):
                self.__auto_init(locals())

            def estimate(self, U, U_d_mu, mu):
                mu = self.opt_fom._add_primal_to_parameter(mu, U)
                output = self.sens_rom.estimate(U_d_mu, mu)
                primal_estimate = self.primal_estimator.estimate(U, mu)[0]

                a_dmu_norm_estimate = self.cont_a_dmu_functional(mu)

                if 0:
                    print(1./self.coercivity_estimator(mu), a_dmu_norm_estimate, primal_estimate, output)
                return 1./self.coercivity_estimator(mu) * (a_dmu_norm_estimate * primal_estimate + output)
        
        class u_d_mu_RBEstimator_for_all_etas(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_a_dmu_functional, primal_estimator, primal_rom, primal_product,
                         primal_sensitivity_product, product, sens_rom, SensBasis, opt_fom):
                self.__auto_init(locals())

            def estimate(self, U, U_d_mu, mu, eta):
                mu = self.opt_fom._add_primal_to_parameter(mu, U)
                mu = self.opt_fom._add_eta_to_parameter(mu, eta)
                output = self.sens_rom.estimate(U_d_mu, mu)
                primal_estimate = self.primal_estimator.estimate(U, mu)[0]

                a_dmu_norm_estimate = 0 
                k = 0
                for (key, item) in self.opt_fom.parameter_type.items():
                    index_dict, _ = self.opt_fom._collect_indices(item)
                    for (l, index) in index_dict.items():
                        a_dmu_norm_estimate += eta[k] * self.cont_a_dmu_functional[key][index](mu)
                        k+=1
                if 0:
                    print(1./self.coercivity_estimator(mu), a_dmu_norm_estimate, primal_estimate, output)
                return 1./self.coercivity_estimator(mu) * (a_dmu_norm_estimate * primal_estimate + output)

        if self.prepare_for_gradient_estimate:
            if self.separated_bases and self.fom.use_corrected_gradient:
                if self.fom.use_corrected_gradient is True or not self.fom.adjoint_approach:
                    u_d_mu_dict = self._construct_zero_dict(self.primal_fom.parameter_type)
                    for (key, item) in self.primal_fom.parameter_type.items():
                        index_dict, _ = self.fom._collect_indices(item)
                        for (l, index) in index_dict.items():
                            if self.unique_basis:
                                RBSens = self.RBPrimalSensitivity
                            else:
                                RBSens = self.RBPrimalSensitivity[key][index]
                            u_d_mu_dict[key][index] = u_d_mu_RBEstimator(self.coercivity_estimator, self.cont_a_dmu_functional[key][index], estimators['primal'],
                                                                           self.primal_rom, self.primal_rom.opt_product,
                                                                           self.primal_sensitivity_rom[key][index].opt_product,
                                                                           self.opt_product, self.primal_sensitivity_rom[key][index],
                                                                           RBSens, key, index, self.fom)
                    estimators['u_d_mu'] = u_d_mu_dict
                elif self.fom.use_corrected_gradient is False and self.fom.adjoint_approach and self.adjoint_estimate is False:
                    _ , self.second_primal_sensitivity_rom, _ = self._build_primal_sensitivity_model_for_all_directions()
                    self.primal_sensitivity = self.primal_reductor
                    RBSens = self.RBPrimal
                    estimators['u_d_mu'] = u_d_mu_RBEstimator_for_all_etas(self.coercivity_estimator, self.cont_a_dmu_functional, estimators['primal'],
                                                            self.primal_rom, self.primal_rom.opt_product,
                                                            self.second_primal_sensitivity_rom.opt_product,
                                                            self.opt_product, self.second_primal_sensitivity_rom,
                                                            RBSens, self.fom)


            elif self.separated_bases is False and self.prepare_for_sensitivity_estimate:
                RBSens = self.RBPrimalSensitivity
                estimators['u_d_mu'] = u_d_mu_RBEstimator_for_all_etas(self.coercivity_estimator, self.cont_a_dmu_functional, estimators['primal'],
                                                        self.primal_rom, self.primal_rom.opt_product,
                                                        self.primal_sensitivity_rom.opt_product,
                                                        self.opt_product, self.primal_sensitivity_rom,
                                                        RBSens, self.fom)
        ###############################################

        # sensitivity_p
        class p_d_mu_RBEstimator(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_k, cont_k_dmu, cont_a_dmu, primal_estimator, dual_estimator, primal_sens_estimator,
                         primal_rom, sens_rom, product, SensBasis, component, index, opt_fom):
                self.__auto_init(locals())
            
            def estimate(self, U, P, U_d_mu, P_d_mu, mu):
                mu = self.opt_fom._add_primal_to_parameter(mu, U)
                mu = self.opt_fom._add_dual_to_parameter(mu, P)
                mu = self.opt_fom._add_primal_sensitivity_to_parameter(mu, U_d_mu)
                output = self.sens_rom.estimate(P_d_mu, mu)

                primal_estimate = self.primal_estimator.estimate(U, mu)[0]
                dual_estimate = self.dual_estimator.estimate(U, P, mu)
                primal_sens_estimate = self.primal_sens_estimator.estimate(U, U_d_mu, mu)

                if 0:
                    print(1./self.coercivity_estimator(mu), self.cont_k_dmu(mu), primal_estimate)
                    print(dual_estimate, self.cont_k(mu), primal_sens_estimate, output)
                return 1./self.coercivity_estimator(mu) * (2 * self.cont_k_dmu(mu) * primal_estimate + 
                                                        self.cont_a_dmu(mu) * dual_estimate + 
                                                        2 * self.cont_k(mu) * primal_sens_estimate + output)

        
        class p_d_mu_RBEstimator_for_all_etas(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_k, cont_k_dmu, cont_a_dmu, primal_estimator, dual_estimator, primal_sens_estimator,
                         primal_rom, sens_rom, product, SensBasis, opt_fom):
                self.__auto_init(locals())

            def estimate(self, U, P, U_d_eta, P_d_eta, mu, eta):
                mu = self.opt_fom._add_primal_to_parameter(mu, U)
                mu = self.opt_fom._add_dual_to_parameter(mu, P)
                mu = self.opt_fom._add_primal_sensitivity_to_parameter(mu, U_d_eta)
                mu = self.opt_fom._add_eta_to_parameter(mu, eta)
                output = self.sens_rom.estimate(P_d_eta, mu)

                primal_estimate = self.primal_estimator.estimate(U, mu)[0]
                dual_estimate = self.dual_estimator.estimate(U, P, mu)
                primal_sens_estimate = self.primal_sens_estimator.estimate(U, U_d_eta, mu, eta)

                k = 0
                cont_a_dmu, cont_k_dmu = 0, 0
                for (key, item) in self.opt_fom.parameter_type.items():
                    index_dict, _ = self.opt_fom._collect_indices(item)
                    for (l, index) in index_dict.items():
                        cont_a_dmu += eta[k] * self.cont_a_dmu[key][index](mu)
                        cont_k_dmu += eta[k] * self.cont_k_dmu[key][index](mu)
                        k+=1
                if 0:
                    print(1./self.coercivity_estimator(mu), cont_k_dmu, primal_estimate)
                    print(cont_a_dmu, dual_estimate, self.cont_k(mu), primal_sens_estimate, output)
                return 1./self.coercivity_estimator(mu) * (2 * cont_k_dmu * primal_estimate + \
                                                        cont_a_dmu * dual_estimate + \
                                                        2 * self.cont_k(mu) * primal_sens_estimate + output)


        if self.prepare_for_sensitivity_estimate:
            if self.separated_bases and self.fom.use_corrected_gradient:
                if (self.fom.use_corrected_gradient is True or not self.fom.adjoint_approach):  
                    p_d_mu_dict = self._construct_zero_dict(self.primal_fom.parameter_type)
                    for (key, item) in self.fom.parameter_type.items():
                        index_dict, _ = self.fom._collect_indices(item)
                        for (l, index) in index_dict.items():
                            if self.unique_basis:
                                RBSens = self.RBDualSensitivity
                            else:
                                RBSens = self.RBDualSensitivity[key][index]
                            p_d_mu_dict[key][index] = p_d_mu_RBEstimator(self.coercivity_estimator, self.cont_k,
                                                                           self.cont_k_dmu[key][index], self.cont_a_dmu_functional[key][index], estimators['primal'],
                                                                           estimators['dual'], u_d_mu_dict[key][index],
                                                                           self.primal_rom, self.dual_sensitivity_rom[key][index], self.opt_product,
                                                                           RBSens, key, index, self.fom)
                    estimators['p_d_mu'] = p_d_mu_dict
                elif self.fom.use_corrected_gradient is False and self.fom.adjoint_approach and self.adjoint_estimate is False:
                    print('NOTE: I am using an ADDITIONAL sensitivity model for estimation')
                    _ , self.second_dual_sensitivity_rom, _ = self._build_dual_sensitivity_model_for_all_directions()
                    self.dual_sensitivity = self.dual_reductor
                    RBSens = self.RBDual
                    estimators['p_d_mu'] = p_d_mu_RBEstimator_for_all_etas(self.coercivity_estimator, self.cont_k,
                                                                       self.cont_k_dmu, self.cont_a_dmu_functional, estimators['primal'],
                                                                       estimators['dual'], estimators['u_d_mu'],
                                                                       self.primal_rom, self.second_dual_sensitivity_rom, self.opt_product,
                                                                       RBSens, self.fom)

            elif self.separated_bases is False:
                RBSens = self.RBPrimalSensitivity
                estimators['p_d_mu'] = p_d_mu_RBEstimator_for_all_etas(self.coercivity_estimator, self.cont_k,
                                                                       self.cont_k_dmu, self.cont_a_dmu_functional, estimators['primal'],
                                                                       estimators['dual'], estimators['u_d_mu'],
                                                                       self.primal_rom, self.dual_sensitivity_rom, self.opt_product,
                                                                       RBSens, self.fom)
        ##########################################

        # Functional hat d_mu 
        class output_hat_d_mu_RBEstimator(ImmutableObject):
            def __init__(self, primal_estimator, dual_estimator, cont_k_dmu, cont_j_dmu, cont_l_dmu, cont_a_dmu_functional,
                         primal_rom, primal_product, dual_product, component, index, 
                         cont_a=None, primal_sens_estimator=None, sens_rom=None, coercivity_estimator=None, opt_fom=None):
                self.__auto_init(locals())

            def estimate(self, U, P, mu, corrected=False, U_d_mu=None, P_d_mu=None, both_estimators=False, non_assembled=False):
                primal_estimate = self.primal_estimator.estimate(U, mu, non_assembled=non_assembled)[0]
                dual_estimate = self.dual_estimator.estimate(U, P, mu, non_assembled=non_assembled)

                norm_P = np.sqrt(self.dual_product.apply2(P,P))[0][0]
                norm_U = np.sqrt(self.primal_product.apply2(U,U))[0][0]

                a_dmu_norm_estimate = self.cont_a_dmu_functional(mu)

                l_dmu_norm_estimate = self.cont_l_dmu(mu)
                if print_pieces or 0:
                    print(primal_estimate, norm_U, self.cont_k_dmu(mu))
                    print(primal_estimate, self.cont_j_dmu(mu), a_dmu_norm_estimate, norm_P)
                    print(dual_estimate, l_dmu_norm_estimate, a_dmu_norm_estimate, norm_U)
                    print(primal_estimate, dual_estimate, a_dmu_norm_estimate)
                    print(primal_estimate**2, self.cont_k_dmu(mu))
                
                if corrected:
                    # print('... NOTE: I need a FOM solve for this estimator')
                    assert U_d_mu is not None
                    mu = self.opt_fom._add_primal_to_parameter(mu, U)
                    mu = self.opt_fom._add_dual_to_parameter(mu, P)
                    mu = self.opt_fom._add_primal_sensitivity_to_parameter(mu, U_d_mu)
                    output = self.sens_rom.estimate(P_d_mu, mu)
                    primal_sens = self.primal_sens_estimator.estimate(U, U_d_mu, mu)
                    if print_pieces or 0:
                        print(1./self.coercivity_estimator(mu), output, primal_estimate)
                        print(self.cont_k_dmu(mu), primal_estimate**2) 
                        print(self.cont_a(mu) , primal_sens , dual_estimate)

                    est1= self.cont_k_dmu(mu) * primal_estimate**2 + \
                        self.cont_a(mu) * primal_sens * dual_estimate + \
                        1./self.coercivity_estimator(mu) * output * primal_estimate
                    if not both_estimators:
                        return est1

                est2 = 2 * primal_estimate * norm_U * self.cont_k_dmu(mu) + \
                       primal_estimate *  (self.cont_j_dmu(mu) + a_dmu_norm_estimate * norm_P) + \
                       dual_estimate * (l_dmu_norm_estimate + a_dmu_norm_estimate * norm_U) + \
                       primal_estimate * dual_estimate * a_dmu_norm_estimate + \
                       primal_estimate**2 * self.cont_k_dmu(mu)
                if not both_estimators:
                    return est2
                else:
                    if corrected:
                        return [est1, est2]
                    else:
                        return [est2, est1]
        
        class output_hat_d_mu_RBEstimator_for_all_etas(ImmutableObject):
            def __init__(self, primal_estimator, dual_estimator, cont_k, cont_k_dmu, cont_j_dmu, cont_l_dmu, cont_a_dmu,
                         primal_rom, primal_product, dual_product, adjoint_estimate, 
                         cont_a=None, primal_sens_estimator=None, sens_rom=None, coercivity_estimator=None, opt_fom=None):
                self.__auto_init(locals())

            def estimate(self, U, P, mu, eta, U_d_eta=None, P_d_eta=None, non_assembled=False):
                primal_estimate = self.primal_estimator.estimate(U, mu, non_assembled=non_assembled)[0]
                dual_estimate = self.dual_estimator.estimate(U, P, mu, non_assembled=non_assembled)
                
                if not self.adjoint_estimate:
                    # print('NOTE: I need a FOM solve for this directional estimator')
                    assert U_d_eta is not None
                    mu = self.opt_fom._add_primal_to_parameter(mu, U)
                    mu = self.opt_fom._add_dual_to_parameter(mu, P)
                    mu = self.opt_fom._add_primal_sensitivity_to_parameter(mu, U_d_eta)
                    mu = self.opt_fom._add_eta_to_parameter(mu, eta)
                    output = self.sens_rom.estimate(P_d_eta, mu)
                    primal_sens = self.primal_sens_estimator.estimate(U, U_d_eta, mu, eta)
        
                    cont_k_dmu, k = 0, 0
                    for (key, item) in self.opt_fom.parameter_type.items():
                        index_dict, _ = self.opt_fom._collect_indices(item)
                        for (l, index) in index_dict.items():
                            cont_k_dmu += eta[k] * self.cont_k_dmu[key][index](mu)
                            k+=1

                    est1= cont_k_dmu * primal_estimate**2 + \
                        self.cont_a(mu) * primal_sens * dual_estimate + \
                        output * primal_estimate
                    return est1
                else:
                    norm_P = np.sqrt(self.dual_product.apply2(P,P))[0][0]
                    norm_U = np.sqrt(self.primal_product.apply2(U,U))[0][0]

                    cont_k_dmu, cont_a_dmu, cont_j_dmu, cont_l_dmu, k = 0, 0, 0, 0, 0
                    for (key, item) in self.opt_fom.parameter_type.items():
                        index_dict, _ = self.opt_fom._collect_indices(item)
                        for (l, index) in index_dict.items():
                            cont_k_dmu += eta[k] * self.cont_k_dmu[key][index](mu)
                            cont_a_dmu += eta[k] * self.cont_a_dmu[key][index](mu)
                            cont_j_dmu += eta[k] * self.cont_j_dmu[key][index](mu)
                            cont_l_dmu += eta[k] * self.cont_l_dmu[key][index](mu)
                            k+=1

                    est2 = 2 * primal_estimate * norm_U * cont_k_dmu + \
                       primal_estimate *  (cont_j_dmu + cont_a_dmu * norm_P) + \
                       dual_estimate * (cont_l_dmu + cont_a_dmu * norm_U) + \
                       primal_estimate * dual_estimate * cont_a_dmu + \
                       primal_estimate**2 * cont_k_dmu + \
                       cont_a_dmu * norm_U * (dual_estimate + 2 * self.cont_k(mu) * primal_estimate) + \
                       primal_estimate * (cont_j_dmu + 2 * cont_k_dmu * norm_U + cont_a_dmu * norm_P)
                    return est2

        if self.prepare_for_gradient_estimate:
            if self.separated_bases and (not self.fom.adjoint_approach or self.use_corrected_gradient):
                d_mu_dict = self._construct_zero_dict(self.primal_fom.parameter_type)
                print('GRAD J ESTIMATOR: standard (d_mu-tilde sensitivity correction optional)')
                for (key, item) in self.primal_fom.parameter_type.items():
                    index_dict, _ = self.fom._collect_indices(item)
                    for (l, index) in index_dict.items():
                        if self.primal_sensitivity_rom is not None and self.use_corrected_gradient:
                            d_mu_dict[key][index] = output_hat_d_mu_RBEstimator(estimators['primal'], estimators['dual'],
                                                                       self.cont_k_dmu[key][index], self.cont_j_dmu[key][index],
                                                                       self.cont_l_dmu[key][index], self.cont_a_dmu_functional[key][index],
                                                                       self.primal_rom,
                                                                       self.primal_rom.opt_product, self.dual_rom.opt_product,
                                                                       key, index, self.cont_a, estimators['u_d_mu'][key][index],
                                                                       self.dual_sensitivity_rom[key][index], self.coercivity_estimator, self.fom)
                        else:
                            assert 0, 'This should not happen'
                estimators['output_functional_hat_d_mus'] = d_mu_dict
            else:
                if self.adjoint_estimate:
                    print('GRAD J ESTIMATOR: adjoint approach - adjoint estimate (no sensitivities)')
                    J_d_mu = output_hat_d_mu_RBEstimator_for_all_etas(estimators['primal'], estimators['dual'],
                                                                   self.cont_k, self.cont_k_dmu, self.cont_j_dmu,
                                                                   self.cont_l_dmu, self.cont_a_dmu_functional,
                                                                   self.primal_rom, self.primal_rom.opt_product, 
                                                                   self.dual_rom.opt_product, self.adjoint_estimate, opt_fom=self.fom)
                else:
                    if self.primal_sensitivity_rom is not None: 
                        if self.adjoint_approach and not self.use_corrected_gradient and self.separated_bases:
                            dual_sensitivity_rom = self.second_dual_sensitivity_rom
                        else:
                            dual_sensitivity_rom = self.dual_sensitivity_rom
                        if (not self.adjoint_approach and not self.use_corrected_gradient):
                            print('GRAD J ESTIMATOR: non corrected estimator for all directions')
                            J_d_mu = self._construct_zero_dict(self.primal_fom.parameter_type)
                            for (key, item) in self.primal_fom.parameter_type.items():
                                index_dict, _ = self.fom._collect_indices(item)
                                for (l, index) in index_dict.items():
                                    J_d_mu[key][index] = output_hat_d_mu_RBEstimator(estimators['primal'], estimators['dual'],
                                                                                   self.cont_k_dmu[key][index], self.cont_j_dmu[key][index],
                                                                                   self.cont_l_dmu[key][index], self.cont_a_dmu_functional[key][index],
                                                                                   self.primal_rom,
                                                                                   self.primal_rom.opt_product, self.dual_rom.opt_product,
                                                                                   key, index)
                        else:
                            print('GRAD J ESTIMATOR: adjoint approach - sensitivity estimate (d_mu) - Also valid for unique')
                            J_d_mu = output_hat_d_mu_RBEstimator_for_all_etas(estimators['primal'], estimators['dual'], self.cont_k,
                                                                   self.cont_k_dmu, self.cont_j_dmu,
                                                                   self.cont_l_dmu, self.cont_a_dmu_functional,
                                                                   self.primal_rom,
                                                                   self.primal_rom.opt_product, self.dual_rom.opt_product, self.adjoint_estimate,
                                                                   self.cont_a, estimators['u_d_mu'],
                                                                   dual_sensitivity_rom, self.coercivity_estimator, opt_fom=self.fom)
                    else:
                        print('GRAD J ESTIMATOR: non corrected estimator')
                        J_d_mu = self._construct_zero_dict(self.primal_fom.parameter_type)
                        for (key, item) in self.primal_fom.parameter_type.items():
                            index_dict, _ = self.fom._collect_indices(item)
                            for (l, index) in index_dict.items():
                                J_d_mu[key][index] = output_hat_d_mu_RBEstimator(estimators['primal'], estimators['dual'],
                                                                               self.cont_k_dmu[key][index], self.cont_j_dmu[key][index],
                                                                               self.cont_l_dmu[key][index], self.cont_a_dmu_functional[key][index],
                                                                               self.primal_rom,
                                                                               self.primal_rom.opt_product, self.dual_rom.opt_product,
                                                                               key, index)

                estimators['output_functional_hat_d_mus'] = J_d_mu

        return estimators

    def project_output(self):
        output_functional = self.fom.output_functional_dict
        li_part = output_functional['linear_part']
        bi_part = output_functional['bilinear_part']
        d_u_li_part = output_functional['d_u_linear_part']
        d_u_bi_part = output_functional['d_u_bilinear_part']
        if self.use_corrected_gradient and self.separated_bases:
            def projected_boundary_functional(key, index):
                if self.unique_basis:
                    B_left = project(self.primal_fom.rhs.operators[1], self.RBPrimalSensitivity, None)
                else:
                    B_left = project(self.primal_fom.rhs.operators[1], self.RBPrimalSensitivity[key][index], None)
                B_right = project(self.primal_fom.rhs.operators[1], self.RBPrimal, None)
                range_ = B_left.range
                class Boundary_squared_projected(ImmutableObject):
                    def __init__(self, source, range, B_left, B_right):
                        self.__auto_init(locals())
                        self.linear = False
                    def apply2(self, U, V, mu=None):   # haack ! 
                        return self.B_left.apply_adjoint(U).to_numpy() * self.B_right.apply_adjoint(V).to_numpy()
                return Boundary_squared_projected(range_, range_, B_left, B_right)
            sens_projected_rhs = self._construct_zero_dict(self.primal_fom.parameter_type)
            sens_projected_dual_rhs_1 = self._construct_zero_dict(self.primal_fom.parameter_type)
            sens_projected_dual_rhs_2 = self._construct_zero_dict(self.primal_fom.parameter_type)
            sens_projected_op = self._construct_zero_dict(self.primal_fom.parameter_type)
            sens_projected_dual_op = self._construct_zero_dict(self.primal_fom.parameter_type)
            for (key, item) in self.primal_fom.parameter_type.items():
                index_dict, _ = self.fom._collect_indices(item)
                for (l, index) in index_dict.items():
                    if not isinstance(self.RBPrimalSensitivity, dict):
                        RBDualSens = self.RBDualSensitivity
                        RBPrimalSens = self.RBPrimalSensitivity
                    else:
                        RBDualSens = self.RBDualSensitivity[key][index]
                        RBPrimalSens = self.RBPrimalSensitivity[key][index]
                    sens_projected_rhs[key][index] = project(self.fom.primal_model.rhs, RBDualSens, None)
                    sens_projected_op[key][index] = project(self.fom.primal_model.operator, self.RBPrimal, RBDualSens)
                    sens_projected_dual_rhs_1[key][index] = project(d_u_li_part, RBPrimalSens, None) 
                    if self.fin_model is True:
                        sens_projected_dual_rhs_2[key][index] = projected_boundary_functional(key, index)
                    else:
                        sens_projected_dual_rhs_2[key][index] = project(d_u_bi_part, RBPrimalSens, self.RBPrimal)
                    sens_projected_dual_op[key][index] = project(self.fom.primal_model.operator, RBPrimalSens, self.RBDual)

        if self.fin_model is False:
            RB = self.RBPrimal
            if self.use_corrected_gradient and self.separated_bases: 
                projected_functionals = {
                    'output_coefficient' : output_functional['output_coefficient'],
                    'linear_part' : project(li_part, RB, None),
                    'bilinear_part' : project(bi_part, RB, RB),
                    'd_u_linear_part' : project(d_u_li_part, RB, None),
                    'd_u_bilinear_part' : project(d_u_bi_part, RB, RB),
                    'dual_projected_d_u_bilinear_part' : project(d_u_bi_part, RB, self.RBDual),
                    'primal_dual_projected_op': project(self.fom.primal_model.operator, RB, self.RBDual),
                    'dual_projected_rhs': project(self.fom.primal_model.rhs, self.RBDual, None),
                    'primal_projected_dual_rhs': project(self.dual_intermediate_fom.rhs, RB, None),
                    'dual_sens_projected_rhs': sens_projected_rhs,
                    'dual_sens_projected_op': sens_projected_op,
                    'primal_sens_projected_dual_rhs_1': sens_projected_dual_rhs_1,
                    'primal_sens_projected_dual_rhs_2': sens_projected_dual_rhs_2,
                    'primal_sens_projected_dual_op': sens_projected_dual_op
                }
            else:
                projected_functionals = {
                    'output_coefficient' : output_functional['output_coefficient'],
                    'linear_part' : project(li_part, RB, None),
                    'bilinear_part' : project(bi_part, RB, RB),
                    'd_u_linear_part' : project(d_u_li_part, RB, None),
                    'd_u_bilinear_part' : project(d_u_bi_part, RB, RB),
                    'dual_projected_d_u_bilinear_part' : project(d_u_bi_part, RB, self.RBDual),
                    'primal_dual_projected_op': project(self.fom.primal_model.operator, RB, self.RBDual),
                    'dual_projected_rhs': project(self.fom.primal_model.rhs, self.RBDual, None),
                    'primal_projected_dual_rhs': project(self.dual_intermediate_fom.rhs, RB, None),
                }
            return projected_functionals
        else:
            Boundary_Functional = self.primal_rom.rhs.operators[1]
            dual_Boundary_Functional = project(self.primal_fom.rhs.operators[1], self.RBDual, None)
            range_ = Boundary_Functional.range
            dual_range_ = dual_Boundary_Functional.range
            class Boundary_squared_half(ImmutableObject):
                def __init__(self, source, range):
                    self.__auto_init(locals())
                    self.linear = False
                def apply2(self, u, v, mu=None):
                    return Boundary_Functional.apply_adjoint(u).to_numpy() * Boundary_Functional.apply_adjoint(v).to_numpy() * 0.5
                def d_mu(self, component, index):
                    return ZeroOperator(self.range, self.source)
            class Boundary_squared(ImmutableObject):
                def __init__(self, source, range):
                    self.__auto_init(locals())
                    self.linear = False
                def apply2(self, u, v, mu=None): 
                    return Boundary_Functional.apply_adjoint(u).to_numpy() * Boundary_Functional.apply_adjoint(v).to_numpy()
                def apply(self, u, mu=None): 
                    return (Boundary_Functional.apply_adjoint(u).to_numpy()[0][0] * Boundary_Functional).as_vector()
                def d_mu(self, component, index):
                    return ZeroOperator(self.range, self.source)
            class dual_projected_Boundary_squared(ImmutableObject):
                def __init__(self, source, range):
                    self.__auto_init(locals())
                    self.linear = False
                def apply2(self, u, v, mu=None): 
                    return Boundary_Functional.apply_adjoint(u).to_numpy() * dual_Boundary_Functional.apply_adjoint(v).to_numpy()
                def apply(self, p, mu=None): 
                    return (dual_Boundary_Functional.apply_adjoint(p).to_numpy()[0][0] * Boundary_Functional).as_vector()
                def apply_adjoint(self, u, mu=None): 
                    return (Boundary_Functional.apply_adjoint(u).to_numpy()[0][0] * dual_Boundary_Functional).as_vector()
                def d_mu(self, component, index):
                    return ZeroOperator(self.range, self.source)
            output_functional = self.fom.output_functional_dict
            li_part = output_functional['linear_part']
            d_u_li_part = output_functional['d_u_linear_part']
            RB = self.RBPrimal
            if self.use_corrected_gradient and self.separated_bases: 
                projected_functionals = {
                    'output_coefficient' : output_functional['output_coefficient'],
                    'linear_part' : project(li_part, RB, None),
                    'bilinear_part' : Boundary_squared_half(range_,range_),
                    'd_u_linear_part' : project(d_u_li_part, RB, None),
                    'd_u_bilinear_part' : Boundary_squared(range_,range_),
                    'dual_projected_d_u_bilinear_part': dual_projected_Boundary_squared(range_, dual_range_),
                    'primal_dual_projected_op': project(self.fom.primal_model.operator, RB, self.RBDual),
                    'dual_projected_rhs': project(self.fom.primal_model.rhs, self.RBDual, None),
                    'primal_projected_dual_rhs': project(self.dual_intermediate_fom.rhs, RB, None),
                    'dual_sens_projected_rhs': sens_projected_rhs,
                    'dual_sens_projected_op': sens_projected_op,
                    'primal_sens_projected_dual_rhs_1': sens_projected_dual_rhs_1,
                    'primal_sens_projected_dual_rhs_2': sens_projected_dual_rhs_2,
                    'primal_sens_projected_dual_op': sens_projected_dual_op
                }
            else:
                projected_functionals = {
                    'output_coefficient' : output_functional['output_coefficient'],
                    'linear_part' : project(li_part, RB, None),
                    'bilinear_part' : Boundary_squared_half(range_,range_),
                    'd_u_linear_part' : project(d_u_li_part, RB, None),
                    'd_u_bilinear_part' : Boundary_squared(range_,range_),
                    'dual_projected_d_u_bilinear_part': dual_projected_Boundary_squared(range_, dual_range_),
                    'primal_dual_projected_op': project(self.fom.primal_model.operator, RB, self.RBDual),
                    'dual_projected_rhs': project(self.fom.primal_model.rhs, self.RBDual, None),
                    'primal_projected_dual_rhs': project(self.dual_intermediate_fom.rhs, RB, None),
                }
            return projected_functionals
    
    def project_product(self):
        projected_product = project(self.opt_product, self.RBPrimal, self.RBPrimal)
        return projected_product

    def assemble_estimator_for_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_primal_subbasis(self, dim):
        raise NotImplementedError

class NonAssembledCoerciveRBReductor(StationaryRBReductor):
    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None):
        assert fom.operator.linear and fom.rhs.linear
        assert isinstance(fom.operator, LincombOperator)
        assert all(not op.parametric for op in fom.operator.operators)
        if fom.rhs.parametric:
            assert isinstance(fom.rhs, LincombOperator)
            assert all(not op.parametric for op in fom.rhs.operators)

        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol)
        self.coercivity_estimator = coercivity_estimator

    def assemble_estimator(self):
        # compute the Riesz representative of (U, .)_L2 with respect to product

        class non_assembled_estimator(ImmutableObject):
            def __init__(self, fom, product, reductor):
                self.__auto_init(locals())
            def estimate(self, U, mu, m):
                U = self.reductor.reconstruct(U)
                riesz = self.product.apply_inverse(self.fom.operator.apply(U, mu) - self.fom.rhs.as_vector(mu))
                sqrt = self.product.apply2(riesz,riesz)
                output = np.sqrt(sqrt)
                return output
        return non_assembled_estimator(self.fom, self.products['RB'], self)

    def assemble_estimator_for_subbasis(self, dims):
        return self._last_rom.estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)

