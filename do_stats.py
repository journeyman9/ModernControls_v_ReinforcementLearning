import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pdb

## Stats -- null hypothesis = equal means assuming equal variance
# if p-val is smaller than .05, reject null hypothesis

# rms_mc_psi_1, rms_mc_psi_2, rms_mc_d2
# rms_rl_psi_1, rms_rl_psi_2, rms_rl_d2

# max_mc_psi_1, max_mc_psi_2, max_mc_d2
# max_rl_psi_1, max_rl_psi_2, max_rl_d2

# mc_min_d, mc_min_psi
# rl_min_d, rl_min_psi

metrics = pd.read_csv(sys.argv[1], sep='\t')

def welch_test(data1, data2, alpha=0.05, tail=2, name1='mu_1', name2='mu_2'):
    data1 = data1.squeeze()
    data2 = data2.squeeze()
    t, p = stats.ttest_ind(data1, data2, equal_var=False)
    
    MU_1 = np.mean(data1)
    SIGMA_1 = np.std(data1)

    MU_2 = np.mean(data2)
    SIGMA_2 = np.std(data2)

    if tail == 1:
        alpha = 2 * alpha

    if p <= alpha:
        print('\nReject null hypothesis (statistically significant) that the means are the same because')
        print('p < alpha: {} < {:.2f}'.format(p, alpha))
        if t < 0: 
            print("Result of the Welch's t-test at level %02g: μ2>μ1, the test passed with p-value = %02g." %(alpha, p))
            print('t < 0: {} < 0'.format(t))
            print(str(name2) + ' = {:.3f} +- {:.3f} > '.format(MU_2, SIGMA_2) + \
                  str(name1) + ' = {:.3f} +- {:.3f}'.format(MU_1, SIGMA_1) + ' with 95% confidence')
        else:
            print("Result of the Welch's t-test level %02g: μ1>μ2, the test passed with p-value = %02g." %(alpha, p))
            print('t >= 0: {} >= 0'.format(t))
            print(str(name1) + ' = {:.3f} +- {:.3f} > '.format(MU_1, SIGMA_1) + \
                  str(name2) + ' = {:.3f} +- {:.3f}'.format(MU_2, SIGMA_2) + ' with 95% confidence')
    else:
        print("\nResults of the Welch's t-test level %02g: there is" + 
              " not enough evidence to prove any order relation between μ1 and μ2." % alpha)
    return MU_1, SIGMA_1, MU_2, SIGMA_2

# RMS
rms_mc_psi_1 = metrics['rms_mc_psi_1'].values
rms_rl_psi_1 = metrics['rms_rl_psi_1'].values
MU_rms_mc_psi_1, SIGMA_rms_mc_psi_1, MU_rms_rl_psi_1, SIGMA_rms_rl_psi_1 = \
    welch_test(rms_mc_psi_1, rms_rl_psi_1, name1='rms_mc_psi_1', name2='rms_rl_psi_1')

rms_mc_psi_2 = metrics['rms_mc_psi_2'].values
rms_rl_psi_2 = metrics['rms_rl_psi_2'].values
MU_rms_mc_psi_2, SIGMA_rms_mc_psi_2, MU_rms_rl_psi_2, SIGMA_rms_rl_psi_2 = \
    welch_test(rms_mc_psi_2, rms_rl_psi_2, name1='rms_mc_psi_2', name2='rms_rl_psi_2')

rms_mc_d2 = metrics['rms_mc_d2'].values
rms_rl_d2 = metrics['rms_rl_d2'].values
MU_rms_mc_d2, SIGMA_rms_mc_d2, MU_rms_rl_d2, SIGMA_rms_rl_d2 = \
    welch_test(rms_mc_d2, rms_rl_d2, name1='rms_mc_d2', name2='rms_rl_d2')

fig1, ax1 = plt.subplots()
index = np.arange(3)
bar_width = 0.35 / 2
opacity = 0.8

mc_rms = [MU_rms_mc_psi_1, MU_rms_mc_psi_2, MU_rms_mc_d2]
mc_rms_std = [SIGMA_rms_mc_psi_1, SIGMA_rms_mc_psi_2, SIGMA_rms_mc_d2]
rects1 = plt.bar(index, mc_rms, bar_width, yerr=mc_rms_std, alpha=opacity, 
                 color='k', capsize=10, label='modern controls')

rl_rms = [MU_rms_rl_psi_1, MU_rms_rl_psi_2, MU_rms_rl_d2]
rl_rms_std = [SIGMA_rms_rl_psi_1, SIGMA_rms_rl_psi_2, SIGMA_rms_rl_d2]

rects2 = plt.bar(index+bar_width, rl_rms, bar_width, yerr=rl_rms_std, 
                 alpha=opacity, color='b', capsize=10, label='reinforcement learning')
plt.xlabel('Error Terms')
plt.ylabel('rms value')
plt.xticks(index+bar_width/2, (r'$\psi_{1} [rad]$', r'$\psi_{2} [rad]$', r'$d_{2} [m]$'))
plt.legend()
plt.tight_layout()
plt.show()

# Max
max_mc_psi_1 = metrics['max_mc_psi_1'].values
max_rl_psi_1 = metrics['max_rl_psi_1'].values

max_mc_psi_2 = metrics['max_mc_psi_2'].values
max_rl_psi_2 = metrics['max_rl_psi_2'].values

max_mc_d2 = metrics['max_mc_d2'].values
max_rl_d2 = metrics['max_rl_d2'].values

# Goal
mc_min_d = metrics['mc_min_d'].values
rl_min_d = metrics['rl_min_d'].values

mc_min_psi = metrics['mc_min_psi'].values
rl_min_psi = metrics['rl_min_psi'].values

