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

PARAMS = [12.192, 11.192, 10.192, 9.192, 8.192]
#PARAMS = [-0.23, -0.26, -0.29, 0.00, 0.29]
#PARAMS = [-1.118, -1.564, -2.012, -2.459, -2.906]

foldername = sys.argv[1]
metrics1 = pd.read_csv('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + 
                        str(PARAMS[0]).replace(".", "_") + '.txt', sep='\t', comment='#')

metrics2 = pd.read_csv('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + 
                        str(PARAMS[1]).replace(".", "_") + '.txt', sep='\t', comment='#')

metrics3 = pd.read_csv('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + 
                        str(PARAMS[2]).replace(".", "_") + '.txt', sep='\t', comment='#')

metrics4 = pd.read_csv('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + 
                        str(PARAMS[3]).replace(".", "_") + '.txt', sep='\t', comment='#')

metrics5 = pd.read_csv('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + 
                        str(PARAMS[4]).replace(".", "_") + '.txt', sep='\t', comment='#')

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
        print('p <= alpha: {} < {:.2f}'.format(p, alpha))
        if t < 0: 
            print('t < 0: {} < 0'.format(t))
            print(str(name2) + ' = {:.3f} +- {:.3f} > '.format(MU_2, SIGMA_2) + \
                  str(name1) + ' = {:.3f} +- {:.3f}'.format(MU_1, SIGMA_1) + ' with 95% confidence')
        else:
            print('t >= 0: {} >= 0'.format(t))
            print(str(name1) + ' = {:.3f} +- {:.3f} > '.format(MU_1, SIGMA_1) + \
                  str(name2) + ' = {:.3f} +- {:.3f}'.format(MU_2, SIGMA_2) + ' with 95% confidence')
    else:
        print('\nNeed more samples for {} and {}'.format(name1, name2) +\
              ' p > alpha: {} > {}'.format(p, alpha))
    return MU_1, SIGMA_1, MU_2, SIGMA_2

# RMS
fig1, ax1 = plt.subplots(1, 3, figsize=(20, 10), sharex=True)
index = np.arange(5)
bar_width = 0.35 / 2
opacity = 0.8

ax1[0].set_ylabel(r'rms $\psi_{1} [rad]$')
ax1[0].set_xlabel(foldername.strip('/') + '[m]')
ax1[0].set_xticks(index+bar_width/2)
ax1[0].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

ax1[1].set_ylabel(r'rms $\psi_{2} [rad]$')
ax1[1].set_xlabel(foldername.strip('/') + '[m]')
ax1[1].set_xticks(index+bar_width/2)
ax1[1].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

ax1[2].set_ylabel(r'rms $d_{2} [m]$')
ax1[2].set_xlabel(foldername.strip('/') + '[m]')
ax1[2].set_xticks(index+bar_width/2)
ax1[2].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_1 ~~~~~~~~~~~~~~~')
rms_mc1_psi_1 = metrics1['rms_mc_psi_1'].values
rms_rl1_psi_1 = metrics1['rms_rl_psi_1'].values
MU_rms_mc1_psi_1, SIGMA_rms_mc1_psi_1, MU_rms_rl1_psi_1, SIGMA_rms_rl1_psi_1 = \
    welch_test(rms_mc1_psi_1, rms_rl1_psi_1, name1='rms_mc1_psi_1', name2='rms_rl1_psi_1')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_2 ~~~~~~~~~~~~~~~')
rms_mc2_psi_1 = metrics2['rms_mc_psi_1'].values
rms_rl2_psi_1 = metrics2['rms_rl_psi_1'].values
MU_rms_mc2_psi_1, SIGMA_rms_mc2_psi_1, MU_rms_rl2_psi_1, SIGMA_rms_rl2_psi_1 = \
    welch_test(rms_mc2_psi_1, rms_rl2_psi_1, name1='rms_mc2_psi_1', name2='rms_rl2_psi_1')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
rms_mc3_psi_1 = metrics3['rms_mc_psi_1'].values
rms_rl3_psi_1 = metrics3['rms_rl_psi_1'].values
MU_rms_mc3_psi_1, SIGMA_rms_mc3_psi_1, MU_rms_rl3_psi_1, SIGMA_rms_rl3_psi_1 = \
    welch_test(rms_mc3_psi_1, rms_rl3_psi_1, name1='rms_mc3_psi_1', name2='rms_rl3_psi_1')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_4 ~~~~~~~~~~~~~~~')
rms_mc4_psi_1 = metrics4['rms_mc_psi_1'].values
rms_rl4_psi_1 = metrics4['rms_rl_psi_1'].values
MU_rms_mc4_psi_1, SIGMA_rms_mc4_psi_1, MU_rms_rl4_psi_1, SIGMA_rms_rl4_psi_1 = \
    welch_test(rms_mc4_psi_1, rms_rl4_psi_1, name1='rms_mc4_psi_1', name2='rms_rl4_psi_1')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_5 ~~~~~~~~~~~~~~~')
rms_mc5_psi_1 = metrics5['rms_mc_psi_1'].values
rms_rl5_psi_1 = metrics5['rms_rl_psi_1'].values
MU_rms_mc5_psi_1, SIGMA_rms_mc5_psi_1, MU_rms_rl5_psi_1, SIGMA_rms_rl5_psi_1 = \
    welch_test(rms_mc5_psi_1, rms_rl5_psi_1, name1='rms_mc5_psi_1', name2='rms_rl5_psi_1')

mc_rms_psi_1 = [MU_rms_mc1_psi_1, MU_rms_mc2_psi_1, MU_rms_mc3_psi_1, MU_rms_mc4_psi_1, MU_rms_mc5_psi_1]
mc_rms_psi_1_std = [SIGMA_rms_mc1_psi_1, SIGMA_rms_mc2_psi_1, SIGMA_rms_mc3_psi_1, 
                    SIGMA_rms_mc4_psi_1, SIGMA_rms_mc5_psi_1]
rects1 = ax1[0].bar(index, mc_rms_psi_1, bar_width, yerr=mc_rms_psi_1_std, alpha=opacity, 
                    color='k', capsize=10, label='modern controls')

rl_rms_psi_1 = [MU_rms_rl1_psi_1, MU_rms_rl2_psi_1, MU_rms_rl3_psi_1, MU_rms_rl4_psi_1, MU_rms_rl5_psi_1]
rl_rms_psi_1_std = [SIGMA_rms_rl1_psi_1, SIGMA_rms_rl2_psi_1, SIGMA_rms_rl3_psi_1, 
                    SIGMA_rms_rl4_psi_1, SIGMA_rms_rl5_psi_1]

rects2 = ax1[0].bar(index+bar_width, rl_rms_psi_1, bar_width, yerr=rl_rms_psi_1_std, 
                    alpha=opacity, color='b', capsize=10, label='reinforcement learning')
ax1[0].legend()

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_1 ~~~~~~~~~~~~~~~')
rms_mc1_psi_2 = metrics1['rms_mc_psi_2'].values
rms_rl1_psi_2 = metrics1['rms_rl_psi_2'].values
MU_rms_mc1_psi_2, SIGMA_rms_mc1_psi_2, MU_rms_rl1_psi_2, SIGMA_rms_rl1_psi_2 = \
    welch_test(rms_mc1_psi_2, rms_rl1_psi_2, name1='rms_mc1_psi_2', name2='rms_rl1_psi_2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_2 ~~~~~~~~~~~~~~~')
rms_mc2_psi_2 = metrics2['rms_mc_psi_2'].values
rms_rl2_psi_2 = metrics2['rms_rl_psi_2'].values
MU_rms_mc2_psi_2, SIGMA_rms_mc2_psi_2, MU_rms_rl2_psi_2, SIGMA_rms_rl2_psi_2 = \
    welch_test(rms_mc2_psi_2, rms_rl2_psi_2, name1='rms_mc2_psi_2', name2='rms_rl2_psi_2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
rms_mc3_psi_2 = metrics3['rms_mc_psi_2'].values
rms_rl3_psi_2 = metrics3['rms_rl_psi_2'].values
MU_rms_mc3_psi_2, SIGMA_rms_mc3_psi_2, MU_rms_rl3_psi_2, SIGMA_rms_rl3_psi_2 = \
    welch_test(rms_mc3_psi_2, rms_rl3_psi_2, name1='rms_mc3_psi_2', name2='rms_rl3_psi_2')


print('~~~~~~~~~~~~~' + foldername.strip('/') + '_4 ~~~~~~~~~~~~~~~')
rms_mc4_psi_2 = metrics4['rms_mc_psi_2'].values
rms_rl4_psi_2 = metrics4['rms_rl_psi_2'].values
MU_rms_mc4_psi_2, SIGMA_rms_mc4_psi_2, MU_rms_rl4_psi_2, SIGMA_rms_rl4_psi_2 = \
    welch_test(rms_mc4_psi_2, rms_rl4_psi_2, name1='rms_mc4_psi_2', name2='rms_rl4_psi_2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_5 ~~~~~~~~~~~~~~~')
rms_mc5_psi_2 = metrics5['rms_mc_psi_2'].values
rms_rl5_psi_2 = metrics5['rms_rl_psi_2'].values
MU_rms_mc5_psi_2, SIGMA_rms_mc5_psi_2, MU_rms_rl5_psi_2, SIGMA_rms_rl5_psi_2 = \
    welch_test(rms_mc5_psi_2, rms_rl5_psi_2, name1='rms_mc5_psi_2', name2='rms_rl5_psi_2')

mc_rms_psi_2 = [MU_rms_mc1_psi_2, MU_rms_mc2_psi_2, MU_rms_mc3_psi_2, MU_rms_mc4_psi_2, MU_rms_mc5_psi_2]
mc_rms_psi_2_std = [SIGMA_rms_mc1_psi_2, SIGMA_rms_mc2_psi_2, SIGMA_rms_mc3_psi_2, 
                    SIGMA_rms_mc4_psi_2, SIGMA_rms_mc5_psi_2]
rects3 = ax1[1].bar(index, mc_rms_psi_2, bar_width, yerr=mc_rms_psi_2_std, alpha=opacity, 
                    color='k', capsize=10, label='modern controls')

rl_rms_psi_2 = [MU_rms_rl1_psi_2, MU_rms_rl2_psi_2, MU_rms_rl3_psi_2, MU_rms_rl4_psi_2, MU_rms_rl5_psi_2]
rl_rms_psi_2_std = [SIGMA_rms_rl1_psi_2, SIGMA_rms_rl2_psi_2, SIGMA_rms_rl3_psi_2, 
                    SIGMA_rms_rl4_psi_2, SIGMA_rms_rl5_psi_2]

rects4 = ax1[1].bar(index+bar_width, rl_rms_psi_2, bar_width, yerr=rl_rms_psi_2_std, 
                    alpha=opacity, color='b', capsize=10, label='reinforcement learning')


print('~~~~~~~~~~~~~' + foldername.strip('/') + '_1 ~~~~~~~~~~~~~~~')
rms_mc1_d2 = metrics1['rms_mc_d2'].values
rms_rl1_d2 = metrics1['rms_rl_d2'].values
MU_rms_mc1_d2, SIGMA_rms_mc1_d2, MU_rms_rl1_d2, SIGMA_rms_rl1_d2 = \
    welch_test(rms_mc1_d2, rms_rl1_d2, name1='rms_mc1_d2', name2='rms_rl1_d2')


print('~~~~~~~~~~~~~' + foldername.strip('/') + '_2 ~~~~~~~~~~~~~~~')
rms_mc2_d2 = metrics2['rms_mc_d2'].values
rms_rl2_d2 = metrics2['rms_rl_d2'].values
MU_rms_mc2_d2, SIGMA_rms_mc2_d2, MU_rms_rl2_d2, SIGMA_rms_rl2_d2 = \
    welch_test(rms_mc2_d2, rms_rl2_d2, name1='rms_mc2_d2', name2='rms_rl2_d2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
rms_mc3_d2 = metrics3['rms_mc_d2'].values
rms_rl3_d2 = metrics3['rms_rl_d2'].values
MU_rms_mc3_d2, SIGMA_rms_mc3_d2, MU_rms_rl3_d2, SIGMA_rms_rl3_d2 = \
    welch_test(rms_mc3_d2, rms_rl3_d2, name1='rms_mc3_d2', name2='rms_rl3_d2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_4 ~~~~~~~~~~~~~~~')
rms_mc4_d2 = metrics4['rms_mc_d2'].values
rms_rl4_d2 = metrics4['rms_rl_d2'].values
MU_rms_mc4_d2, SIGMA_rms_mc4_d2, MU_rms_rl4_d2, SIGMA_rms_rl4_d2 = \
    welch_test(rms_mc4_d2, rms_rl4_d2, name1='rms_mc4_d2', name2='rms_rl4_d2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_5 ~~~~~~~~~~~~~~~')
rms_mc5_d2 = metrics5['rms_mc_d2'].values
rms_rl5_d2 = metrics5['rms_rl_d2'].values
MU_rms_mc5_d2, SIGMA_rms_mc5_d2, MU_rms_rl5_d2, SIGMA_rms_rl5_d2 = \
    welch_test(rms_mc5_d2, rms_rl5_d2, name1='rms_mc5_d2', name2='rms_rl5_d2')

mc_rms_d2 = [MU_rms_mc1_d2, MU_rms_mc2_d2, MU_rms_mc3_d2, MU_rms_mc4_d2, MU_rms_mc5_d2]
mc_rms_d2_std = [SIGMA_rms_mc1_d2, SIGMA_rms_mc2_d2, SIGMA_rms_mc3_d2, 
                    SIGMA_rms_mc4_d2, SIGMA_rms_mc5_d2]
rects5 = ax1[2].bar(index, mc_rms_d2, bar_width, yerr=mc_rms_d2_std, alpha=opacity, 
                    color='k', capsize=10, label='modern controls')

rl_rms_d2 = [MU_rms_rl1_d2, MU_rms_rl2_d2, MU_rms_rl3_d2, MU_rms_rl4_d2, MU_rms_rl5_d2]
rl_rms_d2_std = [SIGMA_rms_rl1_d2, SIGMA_rms_rl2_d2, SIGMA_rms_rl3_d2, 
                    SIGMA_rms_rl4_d2, SIGMA_rms_rl5_d2]

rects6 = ax1[2].bar(index+bar_width, rl_rms_d2, bar_width, yerr=rl_rms_d2_std, 
                    alpha=opacity, color='b', capsize=10, label='reinforcement learning')

plt.tight_layout()
plt.show()

exit()

# Max
max_mc_psi_1 = abs(metrics1['max_mc_psi_1'].values)
max_rl_psi_1 = abs(metrics1['max_rl_psi_1'].values)
MU_max_mc_psi_1, SIGMA_max_mc_psi_1, MU_max_rl_psi_1, SIGMA_max_rl_psi_1 = \
    welch_test(max_mc_psi_1, max_rl_psi_1, name1='max_mc_psi_1', name2='max_rl_psi_1')

max_mc_psi_2 = abs(metrics1['max_mc_psi_2'].values)
max_rl_psi_2 = abs(metrics1['max_rl_psi_2'].values)
MU_max_mc_psi_2, SIGMA_max_mc_psi_2, MU_max_rl_psi_2, SIGMA_max_rl_psi_2 = \
    welch_test(max_mc_psi_2, max_rl_psi_2, name1='max_mc_psi_2', name2='max_rl_psi_2')

max_mc_d2 = abs(metrics1['max_mc_d2'].values)
max_rl_d2 = abs(metrics1['max_rl_d2'].values)
MU_max_mc_d2, SIGMA_max_mc_d2, MU_max_rl_d2, SIGMA_max_rl_d2 = \
    welch_test(max_mc_d2, max_rl_d2, name1='max_mc_d2', name2='max_rl_d2')

fig2, ax2 = plt.subplots()
index = np.arange(3)
bar_width = 0.35 / 2
opacity = 0.8

mc_max = [MU_max_mc_psi_1, MU_max_mc_psi_2, MU_max_mc_d2]
mc_max_std = [SIGMA_max_mc_psi_1, SIGMA_max_mc_psi_2, SIGMA_max_mc_d2]
rects1 = plt.bar(index, mc_max, bar_width, yerr=mc_max_std, alpha=opacity, 
		 color='k', capsize=10, label='Modern Controls')

rl_max = [MU_max_rl_psi_1, MU_max_rl_psi_2, MU_max_rl_d2]
rl_max_std = [SIGMA_max_rl_psi_1, SIGMA_max_rl_psi_2, SIGMA_max_rl_d2]
rects2 = plt.bar(index+bar_width, rl_max, bar_width, yerr=rl_max_std, 
		 alpha=opacity, color='b', capsize=10, label='Reinforcement Learning')
plt.xlabel('Error Terms')
plt.ylabel('max value')
plt.xticks(index+bar_width/2, (r'$\psi_{1} [rad]$', r'$\psi_{2} [rad]$', r'$d_{2} [m]$'))
plt.legend()
plt.tight_layout()

'''
n_bins = 40
fig3, ax3 = plt.subplots(1, 2, sharey=True, tight_layout=True)
ax3[0].hist(rms_mc_psi_1, bins=n_bins)
ax3[0].set_ylabel('Frequency')
ax3[1].hist(rms_rl_psi_1, bins=n_bins)
ax3[0].title.set_text('Modern Controls')
ax3[1].title.set_text('Reinforcement Learning')
ax3[0].set_xlabel(r'rms $\psi_{1} [rad]$')
ax3[1].set_xlabel(r'rms $\psi_{1} [rad]$')
'''

# rms_mc_psi_1, rms_mc_psi_2, rms_mc_d2
# rms_rl_psi_1, rms_rl_psi_2, rms_rl_d2

# max_mc_psi_1, max_mc_psi_2, max_mc_d2
# max_rl_psi_1, max_rl_psi_2, max_rl_d2

# mc_min_d, mc_min_psi
# rl_min_d, rl_min_psi

# Goal
mc_min_d = metrics1['mc_min_d'].values
rl_min_d = metrics1['rl_min_d'].values

mc_min_psi = metrics1['mc_min_psi'].values
rl_min_psi = metrics1['rl_min_psi'].values

