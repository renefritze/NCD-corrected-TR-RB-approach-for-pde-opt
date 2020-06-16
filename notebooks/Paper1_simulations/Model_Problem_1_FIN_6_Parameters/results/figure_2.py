import matplotlib.pyplot as plt
# import tikzplotlib
import sys
path = '../../../'
sys.path.append(path)
from pdeopt.tools import get_data

directory = 'Starter744/'

mu_est = False
mu_error = True

colorclass0 =(0.65, 0.00, 0.15)
colorclass1 =(0.84, 0.19, 0.15)
colorclass2 =(0.96, 0.43, 0.26)
colorclass3 =(0.99, 0.68, 0.38)
colorclass4 = 'black'
colorclass5 =(0.67, 0.85, 0.91)

# I want to have these methods in my plot: 
method_tuple = [
                  [2 , '5. FOM proj. BFGS'],
                  [25,'1(a) standard lag.'],
                  [54,'1(b) standard uni.'],
                  [26,'Qian et al. 2017'],
                  ]

if mu_est is False and mu_error is False:
    times_full_0 , J_error_0, FOC_0 = get_data(directory,method_tuple[0][0], FOC=True)
    times_full_1 , J_error_1, FOC_1 = get_data(directory,method_tuple[1][0], FOC=True)
    times_full_2 , J_error_2, FOC_2 = get_data(directory,method_tuple[2][0], FOC=True)
    times_full_3 , J_error_3, FOC_3 = get_data(directory,method_tuple[3][0], FOC=True)
elif mu_est is False and mu_error is True:
    times_full_0 , J_error_0 , mu_error_0 , FOC_0 = get_data(directory,method_tuple[0][0] , mu_error_=mu_error, FOC=True)
    times_full_1 , J_error_1 , mu_error_1 , FOC_1 = get_data(directory,method_tuple[1][0] , mu_error_=mu_error, FOC=True)
    times_full_2 , J_error_2 , mu_error_2 , FOC_2 = get_data(directory,method_tuple[2][0] , mu_error_=mu_error, FOC=True)
    times_full_3 , J_error_3 , mu_error_3 , FOC_3 = get_data(directory,method_tuple[3][0] , mu_error_=mu_error, FOC=True)
elif mu_est is True:
    times_full_0 , J_error_0 , mu_error_0 , times_mu_0 , mu_est_0, FOC_0 = get_data(directory,method_tuple[0][0] , mu_est, mu_est, FOC=True)
    times_full_1 , J_error_1 , mu_error_1 , times_mu_1 , mu_est_1, FOC_1 = get_data(directory,method_tuple[1][0] , mu_est, mu_est, FOC=True)
    times_full_2 , J_error_2 , mu_error_2 , times_mu_2 , mu_est_2, FOC_2 = get_data(directory,method_tuple[2][0] , mu_est, mu_est, FOC=True)
    times_full_3 , J_error_3 , mu_error_3 , times_mu_3 , mu_est_3, FOC_3 = get_data(directory,method_tuple[3][0] , mu_est, mu_est, FOC=True)
    #fix mu_est_2
    times_mu_0 = [ti + times_full_0[0] for ti in times_mu_0] 
    times_mu_1 = [ti + times_full_1[0] for ti in times_mu_1] 
    times_mu_2 = [ti + times_full_2[0] for ti in times_mu_2] 
    times_mu_3 = [ti + times_full_3[0] for ti in times_mu_3] 
if 1:
    timings_figure = plt.figure(figsize=(10,5))
    plt.semilogy(times_full_0 ,J_error_0 , '-', color=colorclass0 , marker='^', label=method_tuple[0][1])
    plt.semilogy(times_full_3 ,J_error_3 , '-', color=colorclass5 , marker='D', label=method_tuple[3][1])
    plt.semilogy(times_full_1 ,J_error_1 , '-', color=colorclass1 , marker='v', label=method_tuple[1][1])
    plt.semilogy(times_full_2 ,J_error_2 , '-', color=colorclass2 , marker='o', label=method_tuple[2][1])
    # plt.xlim([-3,3600])
    # plt.ylim([1e-18, 1e4])
    plt.xlabel('time in seconds [s]',fontsize=14)
    plt.ylabel('$| \hat{\mathcal{J}}_h(\overline{\mu})-\hat{\mathcal{J}}^k_n(\mu_k) |$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([-1,30])
    plt.grid()
    plt.legend(fontsize=10)
    # plt.legend(loc='lower center', fontsize=10)

    # tikzplotlib.save("{}J_error.tex".format(directory))
    # timings_figure.savefig('{}J_error_plot.pdf'.format(directory), format='pdf', bbox_inches="tight")

if 1:
    # Exclude the initial point, from the FOM method
    if len(times_full_0) != len(FOC_0):
         times_full_0_ = times_full_0[1:]
    timings_figure_3 = plt.figure(figsize=(10,5))
    plt.semilogy(times_full_0_, FOC_0 , '-', color=colorclass0 , marker='^', label=method_tuple[0][1])
    plt.semilogy(times_full_3, FOC_3 , '-', color=colorclass5 , marker='D', label=method_tuple[3][1])
    plt.semilogy(times_full_1, FOC_1 , '-', color=colorclass1 , marker='v', label=method_tuple[1][1])
    plt.semilogy(times_full_2, FOC_2 , '-', color=colorclass2 , marker='o', label=method_tuple[2][1])
    
    # plt.xlim([-3,3600])
    # plt.ylim([1e-18, 1e4])
    plt.xlabel('time in seconds [s]',fontsize=14)
    plt.ylabel('FOC condition', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([-1,30])
    plt.grid()
    # plt.legend(loc='lower center', fontsize=10)
    plt.legend(fontsize=10)

    # tikzplotlib.save("{}FOC.tex".format(directory))
    # timings_figure_3.savefig('{}FOC.pdf'.format(directory), format='pdf', bbox_inches="tight")

if mu_error is True:
    timings_figure = plt.figure(figsize=(10,5))
    plt.semilogy(times_full_0 ,mu_error_0 , '-', color=colorclass0 , marker='^', label=method_tuple[0][1])
    plt.semilogy(times_full_3 ,mu_error_3 , '-', color=colorclass5 , marker='D', label=method_tuple[3][1])
    plt.semilogy(times_full_1 ,mu_error_1 , '-', color=colorclass1 , marker='v', label=method_tuple[1][1])
    plt.semilogy(times_full_2 ,mu_error_2 , '-', color=colorclass2 , marker='o', label=method_tuple[2][1])
    plt.xlabel('time in seconds [s]',fontsize=14)
    plt.ylabel('$\| \overline{\mu}-\mu_k \|$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([-1,30])
    plt.grid()
    plt.legend(fontsize=10)
    # plt.legend(loc='lower center', fontsize=10)

    # tikzplotlib.save("{}mu_error.tex".format(directory))
    # timings_figure.savefig('{}mu_error_plot.pdf'.format(directory), format='pdf', bbox_inches="tight")

plt.show()
