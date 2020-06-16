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
import time
from copy import deepcopy

from pdeopt.tools import truncated_conj_grad as TruncCG



def projection_onto_range(parameter_space, mu):

    ranges = parameter_space.ranges
    for (key, item) in parameter_space.parameter_type.items():
        range_ = ranges[key]
        if sum(item) < 2:  # these are the cases () and (1,)
            if mu[key] < range_[0]:
                if item == ():
                    mu[key] = range_[0]
                else:
                    mu[key] = [range_[0]]
            if mu[key] > range_[1]:
                if item == ():
                    mu[key] = range_[1]
                else:
                    mu[key] = [range_[1]]
        else:
            for j in range(item[0]):
                if mu[key][j] < range_[0]:
                    mu[key][j] = range_[0]
                if mu[key][j] > range_[1]:
                    mu[key][j] = range_[1]
    return mu

def active_and_inactive_sets(parameter_space, mu, epsilon):

    Act = []

    ranges = parameter_space.ranges
    for (key,item) in parameter_space.parameter_type.items():
        range_ = ranges[key]
        if sum(item) < 2:
            if mu[key] - range_[0] <= epsilon:
                Act.append(1.0)
            elif range_[1] - mu[key] <= epsilon:
                Act.append(1.0)
            else:
                Act.append(0.0)
        else:
            for j in range(item[0]):
                if mu[key][j] - range_[0] <= epsilon:
                    Act.append(1.0)
                elif range_[1] - mu[key][j] <= epsilon:
                    Act.append(1.0)
                else:
                    Act.append(0.0)

    Act = np.array(Act)
    Inact = np.ones(Act.shape) - Act

    return Act, Inact


def armijo_rule(opt_model, parameter_space, TR_parameters, mu_i, Ji, direction):
    j = 0
    condition = True
    while condition and j < TR_parameters['max_iterations_armijo']:
        mu_ip1 = mu_i + (TR_parameters['initial_step_armijo'] ** j) * direction
        mu_ip1_dict = opt_model.primal_model.parse_parameter(opt_model.pre_parse_parameter(mu_ip1))
        mu_ip1_dict = projection_onto_range(parameter_space,mu_ip1_dict)
        mu_ip1 = opt_model.parse_parameter_inverse(mu_ip1_dict)
        Jip1 = opt_model.output_functional_hat(mu_ip1_dict)

        if not TR_parameters['full_order_model']:
            u_cp = opt_model.solve(mu_ip1_dict)
            p_cp = opt_model.solve_dual(mu_ip1_dict)
            est = opt_model.estimate_output_functional_hat(u_cp, p_cp, mu_ip1_dict)
        else:
            est = 0.0

        if  Jip1 <= Ji - (TR_parameters['armijo_alpha'] / ((TR_parameters['initial_step_armijo'] ** j))) * (np.linalg.norm(mu_ip1-mu_i)**2) and abs(est / Jip1) <= TR_parameters['radius']:
            condition = False

        j = j + 1

    if condition:  # This means that we exit the loop because of maximum iteration reached
        print("Maximum iteration for Armijo rule reached")
        mu_ip1 = mu_i
        mu_ip1_dict = opt_model.primal_model.parse_parameter(opt_model.pre_parse_parameter(mu_ip1))
        Jip1 = Ji
        est = TR_parameters['radius']*Ji # so that the Qian-Grepl method stops as well
       

    return mu_ip1, mu_ip1_dict, Jip1, abs(est / Jip1) #the last is needed for the boundary criterium

def compute_new_hessian_approximation(new_mu,old_mu,new_gradient,old_gradient,old_B):

    gk = new_gradient-old_gradient
    pk = new_mu-old_mu

    den = gk.dot(pk)

    if den>0.0:
        Hkgk = old_B.dot(gk)
        coeff = gk.dot(Hkgk)

        Hkgkpkt = np.outer(Hkgk,pk)

        pkHkgkt = np.outer(pk,Hkgk)

        pkpkt = np.outer(pk,pk)

        new_B = old_B + (den+coeff)/(den*den) * pkpkt - (1.0/den) * Hkgkpkt - (1.0/den)*pkHkgkt

    else:
        print("Curvature condition: {}".format(den))
        print("Reset direction to - gradient")
        new_B = np.eye(old_gradient.size)


    return new_B

def compute_modified_hessian_action_matrix_version(H,Active,Inactive,eta):

    etaA = np.multiply(Active, eta)
    etaI = np.multiply(Inactive, eta)

    Hessian_prod = H.dot(etaI)
    Action_of_modified_H = etaA + np.multiply(Inactive, Hessian_prod)

    return Action_of_modified_H


def solve_optimization_subproblem_BFGS(opt_model, parameter_space, mu_k_dict, TR_parameters, timing=False):
    if not TR_parameters['full_order_model']:
        print('___ starting subproblem')
        if 'beta' not in TR_parameters:
            print('Setting beta to the default 0.95')
            TR_parameters['beta'] = 0.95
    else:
        print("Starting parameter {}".format(mu_k_dict))
    
    tic_ = time.time()
    times = []
    mus = []
    Js = []
    FOCs = []

    mu_diff = 1e6
    J_diff = 1e6
    Ji = opt_model.output_functional_hat(mu_k_dict)

    gradient = opt_model.output_functional_hat_gradient(mu_k_dict)
    normgrad = np.linalg.norm(gradient)
    mu_i = opt_model.parse_parameter_inverse(mu_k_dict)
    mu_i_dict = opt_model.primal_model.parse_parameter(opt_model.pre_parse_parameter(mu_i))

    mu_i_1 = mu_i - gradient
    mu_i_1_dict = projection_onto_range(opt_model.parameter_space, opt_model.parse_parameter(opt_model.pre_parse_parameter(mu_i_1)))
    mu_i_1 = opt_model.parse_parameter_inverse(mu_i_1_dict)
    epsilon_i = TR_parameters['epsilon_i']
    if not isinstance(epsilon_i,float):
        epsilon_i = np.linalg.norm(mu_i_1 - mu_i)#/(np.linalg.norm(mu_i)+1e-8)
    B = np.eye(mu_i.size)
    Active_i, Inactive_i = active_and_inactive_sets(opt_model.parameter_space, mu_i_dict, epsilon_i)



    i = 0
    while i < TR_parameters['max_iterations_subproblem']:
        if i>0:
            if not TR_parameters['full_order_model']:
                if boundary_TR_criterium >= TR_parameters['beta']*TR_parameters['radius']:
                    print('boundary criterium of the TR satisfied, so stopping the sub-problem solver')
                    
                    return mu_ip1_dict, Jcp, i, Jip1, FOCs
                if normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['safety_tolerance'] or mu_diff< TR_parameters['safety_tolerance']:
                    print("Subproblem converged: FOC = {}, mu_diff = {}, J_diff = {} ".format(normgrad,mu_diff,J_diff))
                    break

            else:
                if normgrad < TR_parameters['sub_tolerance']:
                    print("Converged: FOC = {}".format(normgrad))
                    break

        if i == 0 and not TR_parameters['full_order_model']:
            print("Computing the approximate Cauchy point and then start the BFGS method")
            direction = -gradient
        else:
            if Inactive_i.sum() == 0.0:
                if TR_parameters["full_order_model"]:
                    print("All indexes are active, I am using -gradient as direction")
                direction = -gradient
            else:
                direction = compute_modified_hessian_action_matrix_version(B, Active_i, Inactive_i, -gradient)
            if np.dot(direction,gradient) > 0:
                print('Not a descendent direction ... taking -gradient as direction')
                direction = -gradient
  
        if TR_parameters["full_order_model"]:
            mu_ip1, mu_ip1_dict, Jip1, _ = armijo_rule(opt_model, parameter_space, TR_parameters, mu_i, Ji, direction)
        else:
            mu_ip1, mu_ip1_dict, Jip1, boundary_TR_criterium = armijo_rule(opt_model, parameter_space, TR_parameters, mu_i, Ji, direction)
            
        if i == 0:
            if not TR_parameters['full_order_model']:
                Jcp = Jip1
            else:
                Jcp = None

        mu_diff = np.linalg.norm(mu_i - mu_ip1) / np.linalg.norm(mu_i)
        J_diff = abs(Ji - Jip1) / abs(Ji)
        old_mu = deepcopy(mu_i)
        mu_i_dict = mu_ip1_dict
        Ji = Jip1

        old_gradient = deepcopy(gradient)
        gradient = opt_model.output_functional_hat_gradient(mu_i_dict)
        mu_box = opt_model.parse_parameter(opt_model.pre_parse_parameter(opt_model.parse_parameter_inverse(mu_i_dict)-gradient))
        first_order_criticity = opt_model.parse_parameter_inverse(mu_i_dict)-opt_model.parse_parameter_inverse(projection_onto_range(parameter_space, mu_box))
        normgrad = np.linalg.norm(first_order_criticity)
        mu_i = opt_model.parse_parameter_inverse(mu_i_dict)
        mu_i_dict = opt_model.primal_model.parse_parameter(opt_model.pre_parse_parameter(mu_i))

        mu_i_1 = mu_i - gradient
        mu_i_1_dict = projection_onto_range(opt_model.parameter_space,opt_model.parse_parameter(opt_model.pre_parse_parameter(mu_i_1)))
        mu_i_1 = opt_model.parse_parameter_inverse(mu_i_1_dict)
        if not isinstance(epsilon_i,float):
            epsilon_i = np.linalg.norm(mu_i_1 - mu_i)
        Active_i, Inactive_i = active_and_inactive_sets(opt_model.parameter_space, mu_i_dict, epsilon_i)
        B = compute_new_hessian_approximation(mu_i, old_mu, gradient, old_gradient, B)

        if TR_parameters["full_order_model"]:
            print("Step {}, functional {} , FOC condition {}".format(mu_ip1, Ji, np.linalg.norm(first_order_criticity)))
       
        times.append(time.time() -tic_)
        mus.append(mu_ip1)
        Js.append(Ji)
        FOCs.append(normgrad)
        i = i + 1

    print("relative differences mu {} and J {}".format(mu_diff, J_diff))

    if timing:
        return mu_ip1_dict, Jcp, i, Jip1, times, mus, Js, FOCs
    else:
        return mu_ip1_dict, Jcp, i, Jip1, FOCs


def modified_hessian_action(mu,Active,Inactive,opt_model,eta):
    
    # Used only by the projected Newton Method

    etaA = np.multiply(Active,eta)
    etaI = np.multiply(Inactive,eta)

    Action_on_I = opt_model.output_functional_hessian_operator(mu, etaI, False)

    Action_of_modified_operator = etaA + np.multiply(Inactive,Action_on_I)

    return Action_of_modified_operator


def solve_optimization_NewtonMethod(opt_model, parameter_space, mu_k_dict, TR_parameters, timing=False):
    #This method is used to compute an accurate approximation of the optimal parameter mu_bar with the FOM.
    # (Eventually also in the global Greedy). It is not used in the TR algorithm in this paper.
   
    print("Starting parameter {}".format(mu_k_dict))    

    if 'global_RB' not in TR_parameters:
        TR_parameters['global_RB']=False

    tic_toc = time.time()
    times = []
    mus = []
    Js = []
    FOCs = []
    Jcp = None

    mu_diff = 1e6
    J_diff = 1e6
    Ji = opt_model.output_functional_hat(mu_k_dict)

    gradient = opt_model.output_functional_hat_gradient(mu_k_dict)
    normgrad = np.linalg.norm(gradient)
    mu_i = opt_model.parse_parameter_inverse(mu_k_dict)
    mu_i_dict = opt_model.primal_model.parse_parameter(opt_model.pre_parse_parameter(mu_i))

    mu_i_1 = mu_i - gradient
    mu_i_1_dict = projection_onto_range(opt_model.parameter_space, opt_model.parse_parameter(opt_model.pre_parse_parameter(mu_i_1)))
    mu_i_1 = opt_model.parse_parameter_inverse(mu_i_1_dict)
    epsilon_i = TR_parameters['epsilon_i']
    if not isinstance(epsilon_i,float):
        epsilon_i = np.linalg.norm(mu_i_1 - mu_i)

    i = 0
    while i < TR_parameters['max_iterations']:
        if i>0:
            if TR_parameters['full_order_model'] or TR_parameters['global_RB']:
                if normgrad < TR_parameters['sub_tolerance']:
                    print("Converged: FOC = {}".format(normgrad))
                    break

        
        Active_i, Inactive_i = active_and_inactive_sets(opt_model.parameter_space, mu_i_dict, epsilon_i)

        if Inactive_i.sum() == 0.0:
            deltamu = gradient
            if TR_parameters["full_order_model"] or TR_parameters['global_RB']:
                print("I am using projected gradient instead of Newton")
        else:
            print("Using truncated CG for the linear system")
            deltamu, itcg,rescg, infocg = TruncCG(A_func=lambda v: modified_hessian_action(mu=mu_i_dict, Active= Active_i, Inactive= Inactive_i, opt_model=opt_model, eta=v), b= gradient, tol = 1.e-10)
            if infocg > 0:
                print("Choosing the gradient as direction")
                deltamu = gradient
            if np.dot(-deltamu,gradient) >= -1.e-14:
                print('Not a descendent direction ... taking gradient as direction')
                deltamu = gradient
        
        mu_ip1, mu_ip1_dict, Jip1, _, = armijo_rule(opt_model, parameter_space, TR_parameters, mu_i, Ji, -deltamu)

        mu_diff = np.linalg.norm(mu_i - mu_ip1) / np.linalg.norm(mu_i)
        J_diff = abs(Ji - Jip1) / abs(Ji)
        mu_i_dict = mu_ip1_dict
        Ji = Jip1

        gradient = opt_model.output_functional_hat_gradient(mu_i_dict)
        mu_box = opt_model.parse_parameter(opt_model.pre_parse_parameter(opt_model.parse_parameter_inverse(mu_i_dict)-gradient))
        first_order_criticity = opt_model.parse_parameter_inverse(mu_i_dict)-opt_model.parse_parameter_inverse(projection_onto_range(parameter_space, mu_box))
        normgrad = np.linalg.norm(first_order_criticity)
        
        mu_i = opt_model.parse_parameter_inverse(mu_i_dict)
        mu_i_dict = opt_model.primal_model.parse_parameter(opt_model.pre_parse_parameter(mu_i))

        mu_i_1 = mu_i - gradient
        mu_i_1_dict = projection_onto_range(opt_model.parameter_space,opt_model.parse_parameter(opt_model.pre_parse_parameter(mu_i_1)))
        mu_i_1 = opt_model.parse_parameter_inverse(mu_i_1_dict)
        if not isinstance(epsilon_i,float):
            epsilon_i = np.linalg.norm(mu_i_1 - mu_i)

        
        print("Step {}, functional {} , FOC condition {}".format(mu_ip1, Ji, np.linalg.norm(first_order_criticity)))
        
        times.append(time.time() -tic_toc)
        mus.append(mu_ip1)
        Js.append(Ji)
        FOCs.append(normgrad)
        
        i = i + 1

    print("relative differences mu {} and J {}".format(mu_diff, J_diff))

    if timing:
        return mu_ip1_dict, Jcp, i, Jip1, times, mus, Js, FOCs
    else:
        return mu_ip1_dict, Jcp, i, Jip1, FOCs, 0


def enrichment_step(mu, reductor, opt_fom=None):
    new_reductor = deepcopy(reductor)
    u, p = new_reductor.extend_bases(mu)
    opt_rom = new_reductor.reduce()
    return opt_rom, new_reductor, u, p


def TR_algorithm(opt_rom, reductor, TR_parameters=None, extension_params=None, opt_fom=None, return_opt_rom=False):
    if TR_parameters is None:
        mu_k = opt_rom.parameter_space.sample_randomly(1)[0]
        TR_parameters = {'radius': 0.1, 'sub_tolerance': 1e-8, 'max_iterations': 30, 'max_iterations_subproblem':400,
                         'starting_parameter': mu_k, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 
                         'armijo_alpha': 1e-4, 'full_order_model': False, 
                         'epsilon_i': 1e-8, 'Qian-Grepl': False, 'safety_tolerance': 1e-16, 'beta': 0.95}
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.1
        if 'sub_tolerance' not in TR_parameters:
            TR_parameters['sub_tolerance'] = 1e-8
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 30
        if 'max_iterations_subproblem' not in TR_parameters:
            TR_parameters['max_iterations_subproblem'] = 400
        if 'starting_parameter' not in TR_parameters:
            TR_parameters['starting_parameter'] = opt_rom.parameter_space.sample_randomly(1)[0]
        if 'max_iterations_armijo' not in TR_parameters:
            TR_parameters['max_iterations_armijo'] = 50
        if 'initial_step_armijo' not in TR_parameters:
            TR_parameters['initial_step_armijo'] = 0.5
        if 'armijo_alpha' not in TR_parameters:
            TR_parameters['armijo_alpha'] = 1.e-4
        if 'full_order_model' not in TR_parameters:
            TR_parameters['full_order_model'] = False
        if 'printing' not in TR_parameters:
            TR_parameters['printing'] = False
        if 'epsilon_i' not in TR_parameters:
            TR_parameters['epsilon_i'] = 1e-8
        if 'Qian-Grepl' not in TR_parameters:
            TR_parameters['Qian-Grepl'] = False
        if 'safety_tolerance' not in TR_parameters:
            TR_parameters['safety_tolerance'] = 1e-16
        if 'beta' not in TR_parameters:
            TR_parameters['beta'] = 0.95
        
        mu_k = TR_parameters['starting_parameter']
        
            

    if extension_params is None:
        extension_params={'Check_suff_and_nec_conditions': True, 'Enlarge_radius': True, 'opt_fom': None }
    elif TR_parameters['Qian-Grepl']:
        extension_params['Check_suff_and_nec_conditions'] = True
        extension_params['Enlarge_radius'] = False
        if 'opt_fom' not in extension_params:
            extension_params['opt_fom'] = None
    else:
        if 'Check_suff_and_nec_conditions' not in extension_params:
            extension_params['Check_suff_and_nec_conditions'] = True
        if 'Enlarge_radius' not in extension_params:
            extension_params['Enlarge_radius'] = True
        if 'opt_fom' not in extension_params:
            extension_params['opt_fom'] = None
    
    if opt_fom is None:
        opt_fom = extension_params['opt_fom']

    if 'FOC_tolerance' not in TR_parameters:
        TR_parameters['FOC_tolerance'] = TR_parameters['sub_tolerance']
        
    if TR_parameters['Qian-Grepl']:
        print('QIAN et al. 2017 Method')
    
    print('starting parameter {}'.format(mu_k))

    # timings
    tic = time.time()
    Js = []
    FOCs = []
    times = []

    parameter_space = opt_rom.parameter_space
    mu_list = []
    mu_list.append(mu_k)
    JFE_list = []
    normgrad = 1e6
    estimate_gradient = 1e6 # Used only for Qian et al. method
    model_has_been_enriched = False
    point_rejected = False
    J_k = opt_rom.output_functional_hat(mu_k)
    print("Starting value of the cost: {}".format(J_k))
    print("******************************* \n")
    k = 0
    while k < TR_parameters['max_iterations']:
        if point_rejected:
            point_rejected = False
            if TR_parameters['radius'] < 2.22*1e-16:
                print('\nTR-radius is below machine precision... stopping')
                break
        else:
            if not TR_parameters['Qian-Grepl']:
                if (normgrad < TR_parameters['FOC_tolerance']):
                    print('\nStopping criteria fulfilled: FOM FOC condition {} '.format(normgrad))
                    break
            else:
                if (normgrad + estimate_gradient < TR_parameters['FOC_tolerance']):
                    print('\nStopping criteria fulfilled: normgrad {} + estimate_gradient {}'.format(normgrad,estimate_gradient))
                    break
        
        mu_kp1, Jcp, j, J_kp1, _  = solve_optimization_subproblem_BFGS(opt_rom, parameter_space, mu_k,
                                                                                 TR_parameters)           
        u_rom = opt_rom.solve(mu_kp1)
        p_rom = opt_rom.solve_dual(mu_kp1, U=u_rom)
        
        estimate_J = opt_rom.estimate_output_functional_hat(u_rom, p_rom, mu_kp1)

        if TR_parameters['Qian-Grepl']:
            estimate_gradient = opt_rom.estimate_output_functional_hat_gradient_norm(mu_kp1, u_rom, p_rom)
        
        if J_kp1 + estimate_J < Jcp:

            print('checked sufficient condition, starting the enrichment')

            opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, opt_fom=extension_params['opt_fom'])
            JFE_list.append(reductor.fom.output_functional_hat(mu_kp1,u))
            model_has_been_enriched = True


            if extension_params['Enlarge_radius']:
                if len(JFE_list) > 2:
                    if (k-1!= 0) and (JFE_list[-2]-JFE_list[-1])/(J_k-J_kp1) > 0.75:
                        TR_parameters['radius'] *= 2
                        print('enlarging the TR radius to {}'.format(TR_parameters['radius']))

            print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
            mu_list.append(mu_kp1)
            times.append(time.time() -tic)
            Js.append(J_kp1)
            mu_k = mu_kp1
            J_k = opt_rom.output_functional_hat(mu_k)

        elif J_kp1 - estimate_J > Jcp:
            print('necessary condition failed')
            TR_parameters['radius'] = TR_parameters['radius'] * 0.5
            print("Shrinking the TR radius to: {} because Jcp {} and J_kp1 {}".format(TR_parameters['radius'], Jcp,
                                                                                      J_kp1))
            point_rejected = True

        else:
            print('enriching to check the sufficient decrease condition')

            new_rom, new_reductor, u, p = enrichment_step(mu_kp1, reductor, opt_fom=extension_params['opt_fom'])
            JFE_list.append(reductor.fom.output_functional_hat(mu_kp1, u))
            model_has_been_enriched = True



            J_kp1 = new_rom.output_functional_hat(mu_kp1)
            print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))

            if J_kp1 > Jcp + 1e-8:    # add a safety tolerance of 1e-8 for avoiding numerical stability effects
                TR_parameters['radius'] = TR_parameters['radius'] * 0.5
                print("Shrinking the TR radius to: {} because Jcp {} and J_kp1 {}".format(TR_parameters['radius']
                                                                                          ,Jcp,J_kp1))
                point_rejected = True
                JFE_list.pop(-1) #We need to remove the value from the list, because we reject the parameter

            else:
                opt_rom = new_rom
                reductor = new_reductor
                mu_list.append(mu_kp1)
                times.append(time.time() -tic)
                Js.append(J_kp1)
                mu_k = mu_kp1
                if extension_params['Enlarge_radius']:
                    if len(JFE_list) > 2:
                        if (k-1!= 0) and (JFE_list[-2]-JFE_list[-1])/(J_k-J_kp1) > 0.75:
                            TR_parameters['radius'] *= 2
                            print('enlarging the TR radius to {}'.format(TR_parameters['radius']))
                J_k = J_kp1


        if model_has_been_enriched and TR_parameters['Qian-Grepl']:
            # Qian et al. method does not use the fom gradient
            model_has_been_enriched = False

        if not point_rejected:
            if model_has_been_enriched:
                print('computing the fom gradient since the model was enriched')
                gradient = reductor.fom.output_functional_hat_gradient(mu_k, U=u, P=p)
                mu_box = opt_rom.parse_parameter(opt_rom.pre_parse_parameter(opt_rom.parse_parameter_inverse(mu_k)-gradient))
                first_order_criticity = opt_rom.parse_parameter_inverse(mu_k)-opt_rom.parse_parameter_inverse(projection_onto_range(parameter_space, mu_box))
                normgrad = np.linalg.norm(first_order_criticity)
                model_has_been_enriched = False
            else:
                estimate_gradient = opt_rom.estimate_output_functional_hat_gradient_norm(mu_k, non_assembled=True)
                gradient = opt_rom.output_functional_hat_gradient(mu_k)
                mu_box = opt_rom.parse_parameter(opt_rom.pre_parse_parameter(opt_rom.parse_parameter_inverse(mu_k)-gradient))
                first_order_criticity = opt_rom.parse_parameter_inverse(mu_k)-opt_rom.parse_parameter_inverse(projection_onto_range(parameter_space, mu_box))
                normgrad = np.linalg.norm(first_order_criticity)


            FOCs.append(normgrad)

            if TR_parameters['Qian-Grepl']:
                print('estimate_gradient {}'.format(estimate_gradient))
            print("First order critical condition: {}".format(normgrad))

            k= k + 1
        print("******************************* \n")
   

    if extension_params['Enlarge_radius']:
        Js = JFE_list  # This is for speeding-up the post-processing computation of the error for the TR method
                       # This procedure does not give additional speed-up to our method, 
                       # but improves only the time of the post-processing step,
                       # to have the plot of the error, which is not counted in the computational time of the method.

    if k >= TR_parameters['max_iterations']:
        print (" WARNING: Maximum number of iteration for the TR algorithm reached")
   
    if 'timings' in extension_params:
        if extension_params['timings']:
            if return_opt_rom:    
                return mu_list, times, Js, FOCs, opt_rom
            else:
                return mu_list, times, Js, FOCs
        else:
            return mu_list, times, Js, FOCs

    return mu_list
