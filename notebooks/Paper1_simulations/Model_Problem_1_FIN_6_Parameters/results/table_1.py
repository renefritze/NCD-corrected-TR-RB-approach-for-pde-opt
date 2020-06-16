import sys
path = '../../../'
sys.path.append(path)
from pdeopt.tools import get_data
import numpy as np

directories = [ 'Starter10/', 
                'Starter11/',
                'Starter111/',
                'Starter2/',
                'Starter439/',
                'Starter66/',
                'Starter744/',
                'Starter7552/',
                'Starter758/',
                'Starter7822/']

# I want to have these methods in my tabular: 
method_tuple = [
                  [2 , '5. FOM proj. BFGS'],
                  [25,'1(a) standard lag.'],
                  [54,'1(b) standard uni.'],
                  [26,'Qian et al. 2017'],
                  ]

max_time_0 = 0.
max_time_1 = 0.
max_time_2 = 0.
max_time_3 = 0.


min_time_0 = 10000.
min_time_1 = 10000.
min_time_2 = 10000.
min_time_3 = 10000.


sum_0 = []
sum_1 = []
sum_2 = []
sum_3 = []

max_iter_0 = 0.
max_iter_1 = 0.
max_iter_2 = 0.
max_iter_3 = 0.


min_iter_0 = 10000.
min_iter_1 = 10000.
min_iter_2 = 10000.
min_iter_3 = 10000.

iters_0 = []
iters_1 = []
iters_2 = []
iters_3 = []


sum_FOC_0 = []
sum_FOC_1 = []
sum_FOC_2 = []
sum_FOC_3 = []

# print('THIS IS THE MU FOR EXC10')
mu_opt = [0.01,       0.1,        8.37317446, 6.57227607, 0.46651735, 1.88354107]
mu_opt_norm = np.linalg.norm(mu_opt)

sum_muerror_0 = []
sum_muerror_1 = []
sum_muerror_2 = []
sum_muerror_3 = []


for directory in directories:
    times_full_0 , _ , mu_error_0, FOC_0 = get_data(directory,method_tuple[0][0], mu_error_=True, FOC=True)
    total_time = times_full_0[-1]
    sum_muerror_0.append(mu_error_0[-1])
    sum_FOC_0.append(FOC_0[-1])
    sum_0.append(total_time)
    min_time_0 = min(min_time_0, total_time)
    max_time_0 = max(max_time_0, total_time)
    
    iterations = len(times_full_0)-1
    iters_0.append(iterations)
    min_iter_0 = min(min_iter_0, iterations)
    max_iter_0 = max(max_iter_0, iterations)
    
    times_full_1 , _ , mu_error_1, FOC_1 = get_data(directory,method_tuple[1][0], mu_error_=True, FOC=True)
    total_time = times_full_1[-1]
    sum_muerror_1.append(mu_error_1[-1])
    sum_FOC_1.append(FOC_1[-1])
    sum_1.append(total_time)
    min_time_1 = min(min_time_1, total_time)
    max_time_1 = max(max_time_1, total_time)
    
    iterations = len(times_full_1)-1
    iters_1.append(iterations)
    min_iter_1 = min(min_iter_1, iterations)
    max_iter_1 = max(max_iter_1, iterations)
    
    times_full_2 , _ , mu_error_2, FOC_2  = get_data(directory,method_tuple[2][0], mu_error_=True, FOC=True)
    total_time = times_full_2[-1]
    sum_muerror_2.append(mu_error_2[-1])
    sum_FOC_2.append(FOC_2[-1])
    sum_2.append(total_time)
    min_time_2 = min(min_time_2, total_time)
    max_time_2 = max(max_time_2, total_time)
    
    iterations = len(times_full_2)-1
    iters_2.append(iterations)
    min_iter_2 = min(min_iter_2, iterations)
    max_iter_2 = max(max_iter_2, iterations)
    
    times_full_3 , _ , mu_error_3, FOC_3 = get_data(directory,method_tuple[3][0], mu_error_=True, FOC=True)
    total_time = times_full_3[-1]
    sum_muerror_3.append(mu_error_3[-1])
    sum_FOC_3.append(FOC_3[-1])
    sum_3.append(total_time)
    min_time_3 = min(min_time_3, total_time)
    max_time_3 = max(max_time_3, total_time)
    
    iterations = len(times_full_3)-1
    iters_3.append(iterations)
    min_iter_3 = min(min_iter_3, iterations)
    max_iter_3 = max(max_iter_3, iterations)
    
    
average_times_0 = sum(sum_0)/len(directories)
average_times_1 = sum(sum_1)/len(directories)
average_times_2 = sum(sum_2)/len(directories)
average_times_3 = sum(sum_3)/len(directories)

average_iters_0 = sum(iters_0)/len(directories)
average_iters_1 = sum(iters_1)/len(directories)
average_iters_2 = sum(iters_2)/len(directories)
average_iters_3 = sum(iters_3)/len(directories)

average_mu_error_0 = sum(sum_muerror_0)/len(directories)
average_mu_error_1 = sum(sum_muerror_1)/len(directories)
average_mu_error_2 = sum(sum_muerror_2)/len(directories)
average_mu_error_3 = sum(sum_muerror_3)/len(directories)

average_FOC_0 = sum(sum_FOC_0)/len(directories)
average_FOC_1 = sum(sum_FOC_1)/len(directories)
average_FOC_2 = sum(sum_FOC_2)/len(directories)
average_FOC_3 = sum(sum_FOC_3)/len(directories)

from tabulate import tabulate

print('AVERAGE time of the algorithms min and max:\n ')
headers = ['Method', 'average [s]', 'min [s]', 'max [s]', 'av. speed-up']
table = [[method_tuple[0][1], average_times_0, min_time_0, max_time_0, '--'],
         [method_tuple[3][1], average_times_3, min_time_3, max_time_3, average_times_0/average_times_3],
         [method_tuple[1][1], average_times_1, min_time_1, max_time_1, average_times_0/average_times_1],
         [method_tuple[2][1], average_times_2, min_time_2, max_time_2, average_times_0/average_times_2]
         ]

print(tabulate(table, headers=headers, tablefmt='github', floatfmt='.2f'))
#print(tabulate(table, headers=headers, tablefmt='latex', floatfmt='.2f'))
print()
print('AVERAGE ITERATIONS of the algorithms min and max:\n ')
headers = ['Method', 'average it.', 'min it.', 'max it.', 'av. rel.error.', 'av. FOC cond.']
table = [[method_tuple[0][1], average_iters_0, min_iter_0, max_iter_0, average_mu_error_0/mu_opt_norm, average_FOC_0],
         [method_tuple[3][1], average_iters_3, min_iter_3, max_iter_3, average_mu_error_3/mu_opt_norm, average_FOC_3],
         [method_tuple[1][1], average_iters_1, min_iter_1, max_iter_1, average_mu_error_1/mu_opt_norm, average_FOC_1],
         [method_tuple[2][1], average_iters_2, min_iter_2, max_iter_2, average_mu_error_2/mu_opt_norm, average_FOC_2]
         ]
print(tabulate(table, headers=headers, tablefmt='github', floatfmt='.8f'))
#print(tabulate(table, headers=headers, tablefmt='latex', floatfmt='.8f'))
print()
print()
# all together
headers = ['Method', 'average [s]', 'min [s]', 'max [s]', 'speed-up', 'av. it. ', 'min it.', 'max it.', 'rel.error.', 'FOC cond.']
table = [[method_tuple[0][1], average_times_0, min_time_0, max_time_0, '--'                           , average_iters_0, min_iter_0, max_iter_0, average_mu_error_0/mu_opt_norm, average_FOC_0],
         [method_tuple[3][1], average_times_3, min_time_3, max_time_3, average_times_0/average_times_3, average_iters_3, min_iter_3, max_iter_3, average_mu_error_3/mu_opt_norm, average_FOC_3],
         [method_tuple[1][1], average_times_1, min_time_1, max_time_1, average_times_0/average_times_1, average_iters_1, min_iter_1, max_iter_1, average_mu_error_1/mu_opt_norm, average_FOC_1],
         [method_tuple[2][1], average_times_2, min_time_2, max_time_2, average_times_0/average_times_2, average_iters_2, min_iter_2, max_iter_2, average_mu_error_2/mu_opt_norm, average_FOC_2]
         ]
print(tabulate(table, headers=headers, tablefmt='github', floatfmt='.8f'))
