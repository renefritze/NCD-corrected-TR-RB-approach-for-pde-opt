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
#   Luca Mechelli (2020)
#   Tim Keil      (2019 - 2020)
# ~~~

import numpy as np
import scipy
import csv


def compute_errors(opt_fom, J_start, J_opt, mu_start, mu_opt, mus, Js, times, tictoc, FOC= None):
    mu_error = [np.linalg.norm(opt_fom.parse_parameter_inverse(mu_start) - mu_opt)]
    J_error = [J_start - J_opt]
    for mu_i in mus:
        if isinstance(mu_i,dict):
            mu_error.append(np.linalg.norm(opt_fom.parse_parameter_inverse(mu_i) - mu_opt))
        else:
            mu_error.append(np.linalg.norm(mu_i - mu_opt))

    for Ji in Js:
        J_error.append(np.abs(Ji - J_opt))
    times_full = [tictoc]
    for tim in times:
        times_full.append(tim + tictoc)
        
    if FOC is not None:
        if len(FOC)!= len(times_full):
            gradient = opt_fom.output_functional_hat_gradient(mu_start)
            mu_box = opt_fom.parse_parameter(opt_fom.pre_parse_parameter(opt_fom.parse_parameter_inverse(mu_start)-gradient))
            from pdeopt.TR import projection_onto_range
            first_order_criticity = opt_fom.parse_parameter_inverse(mu_start)-opt_fom.parse_parameter_inverse(projection_onto_range(opt_fom.parameter_space, mu_box))
            normgrad = np.linalg.norm(first_order_criticity)
            FOCs= [normgrad]
            FOCs.extend(FOC)
        else:
            FOCs = FOC
        return times_full, J_error, mu_error, FOCs
    
    return times_full, J_error, mu_error

def compute_actual_errors(opt_fom, J_start, J_opt, mus, times, tictoc, mu_start=None, mu_opt=None):
    print('computing actual errors .... ')
    if mu_opt is not None:
        mu_error = [np.linalg.norm(opt_fom.parse_parameter_inverse(mu_start) - mu_opt)]
    J_error = [np.abs(J_start - J_opt)]
    FOCs = []
    for mu in mus:
        if isinstance(mu, dict):
            mu_k = mu
            mu_array = opt_fom.parse_parameter_inverse(mu)
        else:
            mu_k = opt_fom.parse_parameter(opt_fom.pre_parse_parameter(mu))
            mu_array = opt_fom.parse_parameter_inverse(mu_k)
        J_error.append(np.abs(opt_fom.output_functional_hat(mu_k) - J_opt))
        if mu_opt is not None:
            mu_error.append(np.linalg.norm(mu_array - mu_opt))
        gradient = opt_fom.output_functional_hat_gradient(mu_k)
        mu_box = opt_fom.parse_parameter(opt_fom.pre_parse_parameter(opt_fom.parse_parameter_inverse(mu_k)-gradient))
        from pdeopt.TR import projection_onto_range
        first_order_criticity = opt_fom.parse_parameter_inverse(mu_k)-opt_fom.parse_parameter_inverse(projection_onto_range(opt_fom.parameter_space, mu_box))
        normgrad = np.linalg.norm(first_order_criticity)
        FOCs.append(normgrad)
    times_full = [tictoc]
    for tim in times:
        times_full.append(tim + tictoc)
    if len(times_full) != len(J_error):
        J_error = J_error[1:]
    if mu_opt is not None:
        if len(times_full) != len(mu_error):
            mu_error = mu_error[1:]
    if mu_opt is not None:
        return times_full, J_error, FOCs, mu_error
    else:
        return times_full, J_error, FOCs, None

def compute_eigvals(A,B):
    print('WARNING: THIS MIGHT BE VERY EXPENSIVE')
    return scipy.sparse.linalg.eigsh(A, M=B, return_eigenvectors=False)

def save_data(directory, times, J_error, n, mu_error=None, mu_time=None, mu_estimator=None, FOC=None):
    with open('{}/error_{}.txt'.format(directory, n), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in J_error:
            writer.writerow([val])
    if mu_error is not None:
        with open('{}/mu_error_{}.txt'.format(directory, n), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for val in mu_error:
                writer.writerow([val])
    if mu_estimator is not None:
        with open('{}/mu_est_{}.txt'.format(directory, n), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for val in mu_estimator:
                writer.writerow([val])
    if mu_time is not None:
        with open('{}/mu_time_{}.txt'.format(directory, n), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for val in mu_time:
                writer.writerow([val])
    if FOC is not None:
        with open('{}/FOC_{}.txt'.format(directory, n), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for val in FOC:
                writer.writerow([val])
    with open('{}/times_{}.txt'.format(directory, n), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in times:
            writer.writerow([val])

def get_data(directory, n, mu_error_=False, mu_est_=False, FOC=False):
    J_error = []
    times = []
    mu_error = []
    mu_time = []
    mu_est = []
    FOC_ = []
    if mu_error_ is True:
        f = open('{}mu_error_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_error.append(float(val[0]))
    if FOC is True:
        f = open('{}FOC_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            FOC_.append(float(val[0]))
    if mu_est_ is True:
        f = open('{}mu_est_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_est.append(float(val[0]))
        f = open('{}mu_time_{}.txt'.format(directory, n), 'r')
        reader = csv.reader(f)
        for val in reader:
            mu_time.append(float(val[0]))
    f = open('{}error_{}.txt'.format(directory, n), 'r')
    reader = csv.reader(f)
    for val in reader:
        J_error.append(abs(float(val[0])))
    f = open('{}times_{}.txt'.format(directory, n), 'r')
    reader = csv.reader(f)
    for val in reader:
        times.append(float(val[0]))
    if mu_error_:
        if mu_est_:
            if FOC:
                return times, J_error, mu_error, mu_time, mu_est, FOC_
            else:    
                return times, J_error, mu_error, mu_time, mu_est
        else:
            if FOC:
                return times, J_error, mu_error, FOC_
            else:
                return times, J_error, mu_error
    else:
        if FOC:
            return times, J_error, FOC_
        else:
            return times, J_error
    
def truncated_conj_grad(A_func,b,x_0=None,tol=10e-6, maxiter = None, atol = None):
    
    if x_0 is None:
        x_0 = np.zeros(b.size)
    if atol is None:
        atol = tol
    if maxiter is None:
        maxiter = 10*b.size
    
    test = A_func(x_0)
    if len(test) == len(b):
        def action(x):
                return A_func(x)
    else:
        print('wrong input for A in the CG method')
        return
    
    #define r_0, note that test= action(x_0)
    r_k = b-test
    #defin p_0
    p_k = r_k
    count = 0
    #define x_0
    x_k = x_0
    #cause we need the norm more often than one time, we save it
    tmp_r_k_norm = np.linalg.norm(r_k)
    norm_b = np.linalg.norm(b)
    while count < maxiter and tmp_r_k_norm > max(tol*norm_b,atol):
        #save the matrix vector product
        #print(tmp_r_k_norm)
        tmp = action(p_k)
        p_kxtmp = np.dot(p_k,tmp)
        #check if p_k is a descent direction, otherwise terminate
        if p_kxtmp<= 1.e-10*(np.linalg.norm(p_k))**2:
            print("CG truncated at iteration: {} with residual: {}, because p_k is not a descent direction".format(count,tmp_r_k_norm))
            if count>0:
                return x_k, count, tmp_r_k_norm, 0
            else:
                return x_k, count, tmp_r_k_norm, 1
        else:
            #calculate alpha_k
            alpha_k = ((tmp_r_k_norm)**2)/(p_kxtmp)
            #calculate x_k+1
            x_k = x_k + alpha_k*p_k
            #calculate r_k+1
            r_k = r_k - alpha_k*tmp
            #save the new norm of r_k+1
            tmp_r_k1 = np.linalg.norm(r_k)
            #calculate beta_k
            beta_k = (tmp_r_k1)**2/(tmp_r_k_norm)**2
            tmp_r_k_norm = tmp_r_k1
            #calculate p_k+1
            p_k = r_k + beta_k*p_k
            count += 1
    
    if count >= maxiter:
        print("Maximum number of iteration for CG reached, residual= {}".format(tmp_r_k_norm))
        return x_k,count, tmp_r_k_norm, 1
    
    return x_k, count, tmp_r_k_norm, 0


def compute_all_errors_and_estimators_for_all_ROMS(validation_set, opt_fom, opt_rom_1a, opt_rom_2a, opt_rom_3a, opt_rom_4a, opt_rom_5a, reductor_4a, reductor_5a):
    J_errors_1a, DJ_errors_1a, rel_J_errors_1a, rel_DJ_errors_1a, J_estimators_1a, DJ_estimators_1a, effectivities_J_1a, effectivities_DJ_1a = [], [], [], [], [], [], [], []
    J_errors_2a, DJ_errors_2a, rel_J_errors_2a, rel_DJ_errors_2a, J_estimators_2a, DJ_estimators_2a, effectivities_J_2a, effectivities_DJ_2a = [], [], [], [], [], [], [], []
    J_errors_3a, DJ_errors_3a, rel_J_errors_3a, rel_DJ_errors_3a, J_estimators_3a, DJ_estimators_3a, effectivities_J_3a, effectivities_DJ_3a = [], [], [], [], [], [], [], []
    J_errors_4a, DJ_errors_4a, rel_J_errors_4a, rel_DJ_errors_4a, J_estimators_4a, DJ_estimators_4a, effectivities_J_4a, effectivities_DJ_4a = [], [], [], [], [], [], [], []
    J_errors_5a, DJ_errors_5a, rel_J_errors_5a, rel_DJ_errors_5a, J_estimators_5a, DJ_estimators_5a, effectivities_J_5a, effectivities_DJ_5a = [], [], [], [], [], [], [], []
    
    u_mu_errors_4a, rel_u_mu_errors_4a, u_mu_estimators_4a, effectivities_u_mu_4a = [], [], [], []
    u_mu_errors_5a, rel_u_mu_errors_5a, u_mu_estimators_5a, effectivities_u_mu_5a = [], [], [], []
    p_mu_errors_4a, rel_p_mu_errors_4a, p_mu_estimators_4a, effectivities_p_mu_4a = [], [], [], []
    p_mu_errors_5a, rel_p_mu_errors_5a, p_mu_estimators_5a, effectivities_p_mu_5a = [], [], [], []
    
    J, DJ = [], []
    for mu in validation_set:
        print('.', end='', flush=True)
        u_h = opt_fom.solve(mu)
        p_h = opt_fom.solve_dual(mu, U=u_h)
        actual_J = opt_fom.output_functional_hat(mu, U=u_h, P=p_h)
        actual_DJ = opt_fom.output_functional_hat_gradient(mu, U=u_h, P=p_h)
        J.append(actual_J)
        DJ.append(actual_DJ)

        # this is equivalent for all roms
        U = opt_rom_1a.solve(mu)
        P = opt_rom_1a.solve_dual(mu, U=U)
        
        J_1a = opt_rom_1a.output_functional_hat(mu, U=U, P=P)
        J_2a = opt_rom_2a.output_functional_hat(mu, U=U, P=P)
        J_3a = opt_rom_3a.output_functional_hat(mu, U=U, P=P)
        J_4a = opt_rom_4a.output_functional_hat(mu, U=U, P=P)
        J_5a = opt_rom_5a.output_functional_hat(mu, U=U, P=P)
        
        DJ_1a = opt_rom_1a.output_functional_hat_gradient(mu, U=U, P=P)
        DJ_2a = opt_rom_2a.output_functional_hat_gradient(mu, U=U, P=P)
        DJ_3a = opt_rom_3a.output_functional_hat_gradient(mu, U=U, P=P)
        DJ_4a = opt_rom_4a.output_functional_hat_gradient(mu, U=U, P=P)
        DJ_5a = opt_rom_5a.output_functional_hat_gradient(mu, U=U, P=P)
        
        J_estimator_1a = opt_rom_1a.estimate_output_functional_hat(U, P, mu)
        J_estimator_2a = opt_rom_2a.estimate_output_functional_hat(U, P, mu)
        J_estimator_3a = opt_rom_3a.estimate_output_functional_hat(U, P, mu)
        J_estimator_4a = opt_rom_4a.estimate_output_functional_hat(U, P, mu)
        J_estimator_5a = opt_rom_5a.estimate_output_functional_hat(U, P, mu)
        
        DJ_estimator_1a = opt_rom_1a.estimate_output_functional_hat_gradient_norm(mu, U=U, P=P)
        DJ_estimator_2a = opt_rom_2a.estimate_output_functional_hat_gradient_norm(mu, U=U, P=P)
        DJ_estimator_3a = opt_rom_3a.estimate_output_functional_hat_gradient_norm(mu, U=U, P=P)
        DJ_estimator_4a = opt_rom_4a.estimate_output_functional_hat_gradient_norm(mu, U=U, P=P)
        DJ_estimator_5a = opt_rom_5a.estimate_output_functional_hat_gradient_norm(mu, U=U, P=P)

        J_errors_1a.append(np.abs(actual_J - J_1a))
        J_errors_2a.append(np.abs(actual_J - J_2a))
        J_errors_3a.append(np.abs(actual_J - J_3a))
        J_errors_4a.append(np.abs(actual_J - J_4a))
        J_errors_5a.append(np.abs(actual_J - J_5a))
        
        rel_J_errors_1a.append(np.abs(actual_J - J_1a)/actual_J)
        rel_J_errors_2a.append(np.abs(actual_J - J_2a)/actual_J)
        rel_J_errors_3a.append(np.abs(actual_J - J_3a)/actual_J)
        rel_J_errors_4a.append(np.abs(actual_J - J_4a)/actual_J)
        rel_J_errors_5a.append(np.abs(actual_J - J_5a)/actual_J)
    
        DJ_errors_1a.append(np.linalg.norm(actual_DJ - DJ_1a))
        DJ_errors_2a.append(np.linalg.norm(actual_DJ - DJ_2a))
        DJ_errors_3a.append(np.linalg.norm(actual_DJ - DJ_3a))
        DJ_errors_4a.append(np.linalg.norm(actual_DJ - DJ_4a))
        DJ_errors_5a.append(np.linalg.norm(actual_DJ - DJ_5a))

        rel_DJ_errors_1a.append(np.linalg.norm(actual_DJ - DJ_1a)/np.linalg.norm(actual_DJ))
        rel_DJ_errors_2a.append(np.linalg.norm(actual_DJ - DJ_2a)/np.linalg.norm(actual_DJ))
        rel_DJ_errors_3a.append(np.linalg.norm(actual_DJ - DJ_3a)/np.linalg.norm(actual_DJ))
        rel_DJ_errors_4a.append(np.linalg.norm(actual_DJ - DJ_4a)/np.linalg.norm(actual_DJ))
        rel_DJ_errors_5a.append(np.linalg.norm(actual_DJ - DJ_5a)/np.linalg.norm(actual_DJ))
        
        J_estimators_1a.append(J_estimator_1a)
        J_estimators_2a.append(J_estimator_2a)
        J_estimators_3a.append(J_estimator_3a)
        J_estimators_4a.append(J_estimator_4a)
        J_estimators_5a.append(J_estimator_5a)

        DJ_estimators_1a.append(DJ_estimator_1a)
        DJ_estimators_2a.append(DJ_estimator_2a)
        DJ_estimators_3a.append(DJ_estimator_3a)
        DJ_estimators_4a.append(DJ_estimator_4a)
        DJ_estimators_5a.append(DJ_estimator_5a)
        
        effectivities_J_1a.append(J_estimator_1a/J_errors_1a[-1])
        effectivities_J_2a.append(J_estimator_2a/J_errors_2a[-1])
        effectivities_J_3a.append(J_estimator_3a/J_errors_3a[-1])
        effectivities_J_4a.append(J_estimator_4a/J_errors_4a[-1])
        effectivities_J_5a.append(J_estimator_5a/J_errors_5a[-1])
    
        effectivities_DJ_1a.append(DJ_estimator_1a/DJ_errors_1a[-1])
        effectivities_DJ_2a.append(DJ_estimator_2a/DJ_errors_2a[-1])
        effectivities_DJ_3a.append(DJ_estimator_3a/DJ_errors_3a[-1])
        effectivities_DJ_4a.append(DJ_estimator_4a/DJ_errors_4a[-1])
        effectivities_DJ_5a.append(DJ_estimator_5a/DJ_errors_5a[-1])
        
        #sensitivities
        gradient_error_4a = []
        gradient_error_5a = []
        
        rel_gradient_error_4a = []
        rel_gradient_error_5a = []
        
        estimate_4a = []
        estimate_5a = []

        eff_4a = []
        eff_5a = []
        
        # p sensitivity
        p_gradient_error_4a = []
        p_gradient_error_5a = []
        
        rel_p_gradient_error_4a = []
        rel_p_gradient_error_5a = []
        
        estimate_p_4a = []
        estimate_p_5a = []

        eff_p_4a = []
        eff_p_5a = []
        
        mu_as_array = opt_fom.parse_parameter_inverse(mu)
        d = len(mu_as_array)
        eta = np.zeros((d,))
        d_ = 0
        for (key, item) in opt_fom.parameter_space.parameter_type.items():
            index_dict, _ = opt_fom._collect_indices(item)
            for (l, index) in index_dict.items():
                eta[d_] = 1
                
                # u sens
                u_h_eta = opt_fom.solve_for_u_d_mu(key, index, mu, u_h)
            
                u_r_eta_4a = opt_rom_4a.solve_for_u_d_eta(mu, eta, U)
                u_r_eta_5a = opt_rom_5a.solve_for_u_d_mu(key, index, mu, U)

                estimate_4a.append(opt_rom_4a.estimate_u_d_eta(U, u_r_eta_4a, mu, eta))
                estimate_5a.append(opt_rom_5a.estimate_u_d_mu(key, index, U, u_r_eta_5a, mu))
                            
                u_r_eta_4a_h = reductor_4a.primal_sensitivity.reconstruct(u_r_eta_4a)
                u_r_eta_5a_h = reductor_5a.primal_sensitivity[key][index].reconstruct(u_r_eta_5a)

                gradient_error_4a.append(np.sqrt(opt_fom.opt_product.pairwise_apply2(u_h_eta - u_r_eta_4a_h, u_h_eta - u_r_eta_4a_h)))
                gradient_error_5a.append(np.sqrt(opt_fom.opt_product.pairwise_apply2(u_h_eta - u_r_eta_5a_h, u_h_eta - u_r_eta_5a_h)))
                
                rel_gradient_error_4a.append(gradient_error_4a[-1]/opt_fom.opt_product.pairwise_apply2(u_h_eta, u_h_eta))
                rel_gradient_error_5a.append(gradient_error_5a[-1]/opt_fom.opt_product.pairwise_apply2(u_h_eta, u_h_eta))

                eff_4a.append(estimate_4a[-1]/gradient_error_4a[-1])
                eff_5a.append(estimate_5a[-1]/gradient_error_5a[-1])
                
                # p sens
                p_h_eta = opt_fom.solve_for_p_d_mu(key, index, mu, u_h, p_h, u_h_eta)
            
                p_r_eta_4a = opt_rom_4a.solve_for_p_d_eta(mu, eta)
                p_r_eta_5a = opt_rom_5a.solve_for_p_d_mu(key, index, mu)

                estimate_p_4a.append(opt_rom_4a.estimate_p_d_eta(U, P, u_r_eta_4a, p_r_eta_4a, mu, eta))
                estimate_p_5a.append(opt_rom_5a.estimate_p_d_mu(key, index, U, P, u_r_eta_5a, p_r_eta_5a, mu))
                            
                p_r_eta_4a_h = reductor_4a.dual_sensitivity.reconstruct(p_r_eta_4a)
                p_r_eta_5a_h = reductor_5a.dual_sensitivity[key][index].reconstruct(p_r_eta_5a)

                p_gradient_error_4a.append(np.sqrt(opt_fom.opt_product.pairwise_apply2(p_h_eta - p_r_eta_4a_h, p_h_eta - p_r_eta_4a_h)))
                p_gradient_error_5a.append(np.sqrt(opt_fom.opt_product.pairwise_apply2(p_h_eta - p_r_eta_5a_h, p_h_eta - p_r_eta_5a_h)))
                
                rel_p_gradient_error_4a.append(p_gradient_error_4a[-1]/opt_fom.opt_product.pairwise_apply2(p_h_eta, p_h_eta))
                rel_p_gradient_error_5a.append(p_gradient_error_5a[-1]/opt_fom.opt_product.pairwise_apply2(p_h_eta, p_h_eta))

                eff_p_4a.append(estimate_p_4a[-1]/p_gradient_error_4a[-1])
                eff_p_5a.append(estimate_p_5a[-1]/p_gradient_error_5a[-1])

                eta[d_] = 0
                d_ += 1 
        
        u_mu_errors_4a.append(np.linalg.norm(gradient_error_4a)) 
        u_mu_errors_5a.append(np.linalg.norm(gradient_error_5a)) 
        
        rel_u_mu_errors_4a.append(np.linalg.norm(rel_gradient_error_4a))
        rel_u_mu_errors_5a.append(np.linalg.norm(rel_gradient_error_5a))
        
        u_mu_estimators_4a.append(np.linalg.norm(estimate_4a))
        u_mu_estimators_5a.append(np.linalg.norm(estimate_5a))
        
        effectivities_u_mu_4a.append(np.linalg.norm(eff_4a))
        effectivities_u_mu_5a.append(np.linalg.norm(eff_5a))
        
        p_mu_errors_4a.append(np.linalg.norm(p_gradient_error_4a)) 
        p_mu_errors_5a.append(np.linalg.norm(p_gradient_error_5a)) 
        
        rel_p_mu_errors_4a.append(np.linalg.norm(rel_p_gradient_error_4a))
        rel_p_mu_errors_5a.append(np.linalg.norm(rel_p_gradient_error_5a))
        
        p_mu_estimators_4a.append(np.linalg.norm(estimate_p_4a))
        p_mu_estimators_5a.append(np.linalg.norm(estimate_p_5a))
        
        effectivities_p_mu_4a.append(np.linalg.norm(eff_p_4a))
        effectivities_p_mu_5a.append(np.linalg.norm(eff_p_5a))
    
    return J_errors_1a, DJ_errors_1a, rel_J_errors_1a, rel_DJ_errors_1a, J_estimators_1a, DJ_estimators_1a, effectivities_J_1a, effectivities_DJ_1a, \
           J_errors_2a, DJ_errors_2a, rel_J_errors_2a, rel_DJ_errors_2a, J_estimators_2a, DJ_estimators_2a, effectivities_J_2a, effectivities_DJ_2a, \
           J_errors_3a, DJ_errors_3a, rel_J_errors_3a, rel_DJ_errors_3a, J_estimators_3a, DJ_estimators_3a, effectivities_J_3a, effectivities_DJ_3a, \
           J_errors_4a, DJ_errors_4a, rel_J_errors_4a, rel_DJ_errors_4a, J_estimators_4a, DJ_estimators_4a, effectivities_J_4a, effectivities_DJ_4a, \
           J_errors_5a, DJ_errors_5a, rel_J_errors_5a, rel_DJ_errors_5a, J_estimators_5a, DJ_estimators_5a, effectivities_J_5a, effectivities_DJ_5a, \
           J, DJ, \
           u_mu_errors_4a, rel_u_mu_errors_4a, u_mu_estimators_4a, effectivities_u_mu_4a, \
           u_mu_errors_5a, rel_u_mu_errors_5a, u_mu_estimators_5a, effectivities_u_mu_5a, \
           p_mu_errors_4a, rel_p_mu_errors_4a, p_mu_estimators_4a, effectivities_p_mu_4a, \
           p_mu_errors_5a, rel_p_mu_errors_5a, p_mu_estimators_5a, effectivities_p_mu_5a
