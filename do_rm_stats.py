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

PARAMS = [8.192, 9.192, 10.192, 11.192, 12.192]
#PARAMS = [0.29, 0.00, -0.29, -0.26, -0.23]
#PARAMS = [-2.906, -2.459, -2.012, -1.564, -1.118]

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

# Eliminate rows of panda df if does not make it to goal
metrics1 = metrics1[metrics1["mc_goal_flag"] == True]
metrics1 = metrics1[metrics1["rl_goal_flag"] == True]
metrics1 = metrics1[:100]

metrics2 = metrics2[metrics2["mc_goal_flag"] == True]
metrics2 = metrics2[metrics2["rl_goal_flag"] == True]
metrics2 = metrics2[:100]

metrics3 = metrics3[metrics3["mc_goal_flag"] == True]
metrics3 = metrics3[metrics3["rl_goal_flag"] == True]
metrics3 = metrics3[:100]

metrics4 = metrics4[metrics4["mc_goal_flag"] == True]
metrics4 = metrics4[metrics4["rl_goal_flag"] == True]
metrics4 = metrics4[:100]

metrics5 = metrics5[metrics5["mc_goal_flag"] == True]
metrics5 = metrics5[metrics5["rl_goal_flag"] == True]
metrics5 = metrics5[:100]

print("Number of consistent runs between LQR and DDPG: ", len(metrics5))

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
        print(str(name1) + ' = {:.3f} +- {:.3f}, '.format(MU_1, SIGMA_1) + \
              str(name2) + ' = {:.3f} +- {:.3f}'.format(MU_2, SIGMA_2))
    return MU_1, SIGMA_1, MU_2, SIGMA_2

plt.rcParams.update({'font.size':18})
size_fig = (20, 10)
color_LQR = 'g'
color_DDPG = 'b'

# RMS
fig1, ax1 = plt.subplots(1, 3, figsize=size_fig, sharex=True)
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

ax1[2].set_ylabel(r'rms $y_{2} [m]$')
ax1[2].set_xlabel(foldername.strip('/') + '[m]')
ax1[2].set_xticks(index+bar_width/2)
ax1[2].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

print()
print('***************************')
print('RMS Psi_1')
print('***************************')
print()
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
rects1 = ax1[0].bar(index, mc_rms_psi_1, bar_width, yerr=mc_rms_psi_1_std, 
                    alpha=opacity, color=color_LQR, capsize=10, label='LQR',
                    edgecolor='k')

rl_rms_psi_1 = [MU_rms_rl1_psi_1, MU_rms_rl2_psi_1, MU_rms_rl3_psi_1, MU_rms_rl4_psi_1, MU_rms_rl5_psi_1]
rl_rms_psi_1_std = [SIGMA_rms_rl1_psi_1, SIGMA_rms_rl2_psi_1, SIGMA_rms_rl3_psi_1, 
                    SIGMA_rms_rl4_psi_1, SIGMA_rms_rl5_psi_1]

rects2 = ax1[0].bar(index+bar_width, rl_rms_psi_1, bar_width, 
                    yerr=rl_rms_psi_1_std, alpha=opacity, color=color_DDPG, 
                    capsize=10, label='DDPG', edgecolor='k')
ax1[0].legend()

print()
print('***************************')
print('RMS Psi_2')
print('***************************')
print()
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
rects3 = ax1[1].bar(index, mc_rms_psi_2, bar_width, yerr=mc_rms_psi_2_std, 
                    alpha=opacity, color=color_LQR, capsize=10, label='LQR',
                    edgecolor='k')

rl_rms_psi_2 = [MU_rms_rl1_psi_2, MU_rms_rl2_psi_2, MU_rms_rl3_psi_2, MU_rms_rl4_psi_2, MU_rms_rl5_psi_2]
rl_rms_psi_2_std = [SIGMA_rms_rl1_psi_2, SIGMA_rms_rl2_psi_2, SIGMA_rms_rl3_psi_2, 
                    SIGMA_rms_rl4_psi_2, SIGMA_rms_rl5_psi_2]

rects4 = ax1[1].bar(index+bar_width, rl_rms_psi_2, bar_width, 
                    yerr=rl_rms_psi_2_std, alpha=opacity, color=color_DDPG, 
                    capsize=10, label='DDPG', edgecolor='k')

print()
print('***************************')
print('RMS y2')
print('***************************')
print()
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
rects5 = ax1[2].bar(index, mc_rms_d2, bar_width, yerr=mc_rms_d2_std, 
                    alpha=opacity, color=color_LQR, capsize=10, label='LQR',
                    edgecolor='k')

rl_rms_d2 = [MU_rms_rl1_d2, MU_rms_rl2_d2, MU_rms_rl3_d2, MU_rms_rl4_d2, MU_rms_rl5_d2]
rl_rms_d2_std = [SIGMA_rms_rl1_d2, SIGMA_rms_rl2_d2, SIGMA_rms_rl3_d2, 
                    SIGMA_rms_rl4_d2, SIGMA_rms_rl5_d2]

rects6 = ax1[2].bar(index+bar_width, rl_rms_d2, bar_width, yerr=rl_rms_d2_std, 
                    alpha=opacity, color=color_DDPG, capsize=10, label='DDPG',
                    edgecolor='k')

plt.tight_layout()
plt.savefig('stat_rm_rms.eps', format='eps', dpi=1000)

# Max
fig2, ax2 = plt.subplots(1, 3, figsize=size_fig, sharex=True)
index = np.arange(5)
bar_width = 0.35 / 2
opacity = 0.8

ax2[0].set_ylabel(r'max $\psi_{1} [rad]$')
ax2[0].set_xlabel(foldername.strip('/') + '[m]')
ax2[0].set_xticks(index+bar_width/2)
ax2[0].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

ax2[1].set_ylabel(r'max $\psi_{2} [rad]$')
ax2[1].set_xlabel(foldername.strip('/') + '[m]')
ax2[1].set_xticks(index+bar_width/2)
ax2[1].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

ax2[2].set_ylabel(r'max $y_{2} [m]$')
ax2[2].set_xlabel(foldername.strip('/') + '[m]')
ax2[2].set_xticks(index+bar_width/2)
ax2[2].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

print()
print('***************************')
print('Max psi_1')
print('***************************')
print()
print('~~~~~~~~~~~~~' + foldername.strip('/') + '_1 ~~~~~~~~~~~~~~~')
max_mc1_psi_1 = abs(metrics1['max_mc_psi_1'].values)
max_rl1_psi_1 = abs(metrics1['max_rl_psi_1'].values)
MU_max_mc1_psi_1, SIGMA_max_mc1_psi_1, MU_max_rl1_psi_1, SIGMA_max_rl1_psi_1 = \
    welch_test(max_mc1_psi_1, max_rl1_psi_1, name1='max_mc1_psi_1', name2='max_rl1_psi_1')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_2 ~~~~~~~~~~~~~~~')
max_mc2_psi_1 = abs(metrics2['max_mc_psi_1'].values)
max_rl2_psi_1 = abs(metrics2['max_rl_psi_1'].values)
MU_max_mc2_psi_1, SIGMA_max_mc2_psi_1, MU_max_rl2_psi_1, SIGMA_max_rl2_psi_1 = \
    welch_test(max_mc2_psi_1, max_rl2_psi_1, name1='max_mc2_psi_1', name2='max_rl2_psi_1')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
max_mc3_psi_1 = abs(metrics3['max_mc_psi_1'].values)
max_rl3_psi_1 = abs(metrics3['max_rl_psi_1'].values)
MU_max_mc3_psi_1, SIGMA_max_mc3_psi_1, MU_max_rl3_psi_1, SIGMA_max_rl3_psi_1 = \
    welch_test(max_mc3_psi_1, max_rl3_psi_1, name1='max_mc3_psi_1', name2='max_rl3_psi_1')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_4 ~~~~~~~~~~~~~~~')
max_mc4_psi_1 = abs(metrics4['max_mc_psi_1'].values)
max_rl4_psi_1 = abs(metrics4['max_rl_psi_1'].values)
MU_max_mc4_psi_1, SIGMA_max_mc4_psi_1, MU_max_rl4_psi_1, SIGMA_max_rl4_psi_1 = \
    welch_test(max_mc4_psi_1, max_rl4_psi_1, name1='max_mc4_psi_1', name2='max_rl4_psi_1')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_5 ~~~~~~~~~~~~~~~')
max_mc5_psi_1 = abs(metrics5['max_mc_psi_1'].values)
max_rl5_psi_1 = abs(metrics5['max_rl_psi_1'].values)
MU_max_mc5_psi_1, SIGMA_max_mc5_psi_1, MU_max_rl5_psi_1, SIGMA_max_rl5_psi_1 = \
    welch_test(max_mc5_psi_1, max_rl5_psi_1, name1='max_mc5_psi_1', name2='max_rl5_psi_1')

mc_max_psi_1 = [MU_max_mc1_psi_1, MU_max_mc2_psi_1, MU_max_mc3_psi_1, MU_max_mc4_psi_1, MU_max_mc5_psi_1]
mc_max_psi_1_std = [SIGMA_max_mc1_psi_1, SIGMA_max_mc2_psi_1, SIGMA_max_mc3_psi_1, 
                    SIGMA_max_mc4_psi_1, SIGMA_max_mc5_psi_1]
rects7 = ax2[0].bar(index, mc_max_psi_1, bar_width, yerr=mc_max_psi_1_std, 
                    alpha=opacity, color=color_LQR, capsize=10, label='LQR',
                    edgecolor='k')

rl_max_psi_1 = [MU_max_rl1_psi_1, MU_max_rl2_psi_1, MU_max_rl3_psi_1, MU_max_rl4_psi_1, MU_max_rl5_psi_1]
rl_max_psi_1_std = [SIGMA_max_rl1_psi_1, SIGMA_max_rl2_psi_1, SIGMA_max_rl3_psi_1, 
                    SIGMA_max_rl4_psi_1, SIGMA_max_rl5_psi_1]

rects8 = ax2[0].bar(index+bar_width, rl_max_psi_1, bar_width, 
                    yerr=rl_max_psi_1_std, alpha=opacity, color=color_DDPG,                         capsize=10, label='DDPG', edgecolor='k')

print()
print('***************************')
print('Max psi_2')
print('***************************')
print()
print('~~~~~~~~~~~~~' + foldername.strip('/') + '_1 ~~~~~~~~~~~~~~~')
max_mc1_psi_2 = abs(metrics1['max_mc_psi_2'].values)
max_rl1_psi_2 = abs(metrics1['max_rl_psi_2'].values)
MU_max_mc1_psi_2, SIGMA_max_mc1_psi_2, MU_max_rl1_psi_2, SIGMA_max_rl1_psi_2 = \
    welch_test(max_mc1_psi_2, max_rl1_psi_2, name1='max_mc1_psi_2', name2='max_rl1_psi_2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
max_mc2_psi_2 = abs(metrics2['max_mc_psi_2'].values)
max_rl2_psi_2 = abs(metrics2['max_rl_psi_2'].values)
MU_max_mc2_psi_2, SIGMA_max_mc2_psi_2, MU_max_rl2_psi_2, SIGMA_max_rl2_psi_2 = \
    welch_test(max_mc2_psi_2, max_rl2_psi_2, name1='max_mc2_psi_2', name2='max_rl2_psi_2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
max_mc3_psi_2 = abs(metrics3['max_mc_psi_2'].values)
max_rl3_psi_2 = abs(metrics3['max_rl_psi_2'].values)
MU_max_mc3_psi_2, SIGMA_max_mc3_psi_2, MU_max_rl3_psi_2, SIGMA_max_rl3_psi_2 = \
    welch_test(max_mc3_psi_2, max_rl3_psi_2, name1='max_mc3_psi_2', name2='max_rl3_psi_2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_4 ~~~~~~~~~~~~~~~')
max_mc4_psi_2 = abs(metrics4['max_mc_psi_2'].values)
max_rl4_psi_2 = abs(metrics4['max_rl_psi_2'].values)
MU_max_mc4_psi_2, SIGMA_max_mc4_psi_2, MU_max_rl4_psi_2, SIGMA_max_rl4_psi_2 = \
    welch_test(max_mc4_psi_2, max_rl4_psi_2, name1='max_mc4_psi_2', name2='max_rl4_psi_2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_5 ~~~~~~~~~~~~~~~')
max_mc5_psi_2 = abs(metrics5['max_mc_psi_2'].values)
max_rl5_psi_2 = abs(metrics5['max_rl_psi_2'].values)
MU_max_mc5_psi_2, SIGMA_max_mc5_psi_2, MU_max_rl5_psi_2, SIGMA_max_rl5_psi_2 = \
    welch_test(max_mc5_psi_2, max_rl5_psi_2, name1='max_mc5_psi_2', name2='max_rl5_psi_2')

mc_max_psi_2 = [MU_max_mc1_psi_2, MU_max_mc2_psi_2, MU_max_mc3_psi_2, MU_max_mc4_psi_2, MU_max_mc5_psi_2]
mc_max_psi_2_std = [SIGMA_max_mc1_psi_2, SIGMA_max_mc2_psi_2, SIGMA_max_mc3_psi_2, 
                    SIGMA_max_mc4_psi_2, SIGMA_max_mc5_psi_2]
rects9 = ax2[1].bar(index, mc_max_psi_2, bar_width, yerr=mc_max_psi_2_std, 
                    alpha=opacity, color=color_LQR, capsize=10, label='LQR',
                    edgecolor='k')

rl_max_psi_2 = [MU_max_rl1_psi_2, MU_max_rl2_psi_2, MU_max_rl3_psi_2, MU_max_rl4_psi_2, MU_max_rl5_psi_2]
rl_max_psi_2_std = [SIGMA_max_rl1_psi_2, SIGMA_max_rl2_psi_2, SIGMA_max_rl3_psi_2, 
                    SIGMA_max_rl4_psi_2, SIGMA_max_rl5_psi_2]

rects10 = ax2[1].bar(index+bar_width, rl_max_psi_2, bar_width, 
                    yerr=rl_max_psi_2_std, alpha=opacity, color=color_DDPG, 
                    capsize=10, label='DDPG', edgecolor='k')

print()
print('***************************')
print('Max y_2')
print('***************************')
print()
print('~~~~~~~~~~~~~' + foldername.strip('/') + '_1 ~~~~~~~~~~~~~~~')
max_mc1_d2 = abs(metrics1['max_mc_d2'].values)
max_rl1_d2 = abs(metrics1['max_rl_d2'].values)
MU_max_mc1_d2, SIGMA_max_mc1_d2, MU_max_rl1_d2, SIGMA_max_rl1_d2 = \
    welch_test(max_mc1_d2, max_rl1_d2, name1='max_mc1_d2', name2='max_rl1_d2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_2 ~~~~~~~~~~~~~~~')
max_mc2_d2 = abs(metrics2['max_mc_d2'].values)
max_rl2_d2 = abs(metrics2['max_rl_d2'].values)
MU_max_mc2_d2, SIGMA_max_mc2_d2, MU_max_rl2_d2, SIGMA_max_rl2_d2 = \
    welch_test(max_mc2_d2, max_rl2_d2, name1='max_mc2_d2', name2='max_rl2_d2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
max_mc3_d2 = abs(metrics3['max_mc_d2'].values)
max_rl3_d2 = abs(metrics3['max_rl_d2'].values)
MU_max_mc3_d2, SIGMA_max_mc3_d2, MU_max_rl3_d2, SIGMA_max_rl3_d2 = \
    welch_test(max_mc3_d2, max_rl3_d2, name1='max_mc3_d2', name2='max_rl3_d2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_4 ~~~~~~~~~~~~~~~')
max_mc4_d2 = abs(metrics4['max_mc_d2'].values)
max_rl4_d2 = abs(metrics4['max_rl_d2'].values)
MU_max_mc4_d2, SIGMA_max_mc4_d2, MU_max_rl4_d2, SIGMA_max_rl4_d2 = \
    welch_test(max_mc4_d2, max_rl4_d2, name1='max_mc4_d2', name2='max_rl4_d2')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_5 ~~~~~~~~~~~~~~~')
max_mc5_d2 = abs(metrics5['max_mc_d2'].values)
max_rl5_d2 = abs(metrics5['max_rl_d2'].values)
MU_max_mc5_d2, SIGMA_max_mc5_d2, MU_max_rl5_d2, SIGMA_max_rl5_d2 = \
    welch_test(max_mc5_d2, max_rl5_d2, name1='max_mc5_d2', name2='max_rl5_d2')

mc_max_d2 = [MU_max_mc1_d2, MU_max_mc2_d2, MU_max_mc3_d2, MU_max_mc4_d2, MU_max_mc5_d2]
mc_max_d2_std = [SIGMA_max_mc1_d2, SIGMA_max_mc2_d2, SIGMA_max_mc3_d2, 
                    SIGMA_max_mc4_d2, SIGMA_max_mc5_d2]
rects11 = ax2[2].bar(index, mc_max_d2, bar_width, yerr=mc_max_d2_std, 
                    alpha=opacity, color=color_LQR, capsize=10, label='LQR',
                    edgecolor='k')

rl_max_d2 = [MU_max_rl1_d2, MU_max_rl2_d2, MU_max_rl3_d2, MU_max_rl4_d2, MU_max_rl5_d2]
rl_max_d2_std = [SIGMA_max_rl1_d2, SIGMA_max_rl2_d2, SIGMA_max_rl3_d2, 
                    SIGMA_max_rl4_d2, SIGMA_max_rl5_d2]

rects12 = ax2[2].bar(index+bar_width, rl_max_d2, bar_width, yerr=rl_max_d2_std, 
                    alpha=opacity, color=color_DDPG, capsize=10, label='DDPG',
                    edgecolor='k')

plt.tight_layout()
ax2[0].legend()
plt.savefig('stat_rm_max.eps', format='eps', dpi=1000)

# Goal
fig3, ax3 = plt.subplots(1, 2, figsize=size_fig, sharex=True)
index = np.arange(5)
bar_width = 0.35 / 2
opacity = 0.8

ax3[0].set_ylabel(r' goal $d_{2} [m]$')
ax3[0].set_xlabel(foldername.strip('/') + '[m]')
ax3[0].set_xticks(index+bar_width/2)
ax3[0].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

ax3[1].set_ylabel(r' goal $\psi_{2} [rad]$')
ax3[1].set_xlabel(foldername.strip('/') + '[m]')
ax3[1].set_xticks(index+bar_width/2)
ax3[1].set_xticklabels([PARAMS[0], PARAMS[1], PARAMS[2], PARAMS[3], PARAMS[4]])

print()
print('***************************')
print('minimum d')
print('***************************')
print()
print('~~~~~~~~~~~~~' + foldername.strip('/') + '_1 ~~~~~~~~~~~~~~~')
mc1_min_d = metrics1['mc_min_d'].values
rl1_min_d = metrics1['rl_min_d'].values
MU_mc1_min_d, SIGMA_mc1_min_d, MU_rl1_min_d, SIGMA_rl1_min_d = \
    welch_test(mc1_min_d, rl1_min_d, name1='rl1_min_d', name2='rl1_min_d')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_2 ~~~~~~~~~~~~~~~')
mc2_min_d = metrics2['mc_min_d'].values
rl2_min_d = metrics2['rl_min_d'].values
MU_mc2_min_d, SIGMA_mc2_min_d, MU_rl2_min_d, SIGMA_rl2_min_d = \
    welch_test(mc2_min_d, rl2_min_d, name1='rl2_min_d', name2='rl2_min_d')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
mc3_min_d = metrics3['mc_min_d'].values
rl3_min_d = metrics3['rl_min_d'].values
MU_mc3_min_d, SIGMA_mc3_min_d, MU_rl3_min_d, SIGMA_rl3_min_d = \
    welch_test(mc3_min_d, rl3_min_d, name1='rl3_min_d', name2='rl3_min_d')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_4 ~~~~~~~~~~~~~~~')
mc4_min_d = metrics4['mc_min_d'].values
rl4_min_d = metrics4['rl_min_d'].values
MU_mc4_min_d, SIGMA_mc4_min_d, MU_rl4_min_d, SIGMA_rl4_min_d = \
    welch_test(mc4_min_d, rl4_min_d, name1='rl4_min_d', name2='rl4_min_d')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_5 ~~~~~~~~~~~~~~~')
mc5_min_d = metrics5['mc_min_d'].values
rl5_min_d = metrics5['rl_min_d'].values
MU_mc5_min_d, SIGMA_mc5_min_d, MU_rl5_min_d, SIGMA_rl5_min_d = \
    welch_test(mc5_min_d, rl5_min_d, name1='rl5_min_d', name2='rl5_min_d')


mc_min_d = [MU_mc1_min_d, MU_mc2_min_d, MU_mc3_min_d, MU_mc4_min_d, MU_mc5_min_d]
mc_min_d_std = [SIGMA_mc1_min_d, SIGMA_mc2_min_d, SIGMA_mc3_min_d, 
                    SIGMA_mc4_min_d, SIGMA_mc5_min_d]
rects13 = ax3[0].bar(index, mc_min_d, bar_width, yerr=mc_min_d_std, 
                    alpha=opacity, color=color_LQR, capsize=10, label='LQR',
                    edgecolor='k')

rl_min_d = [MU_rl1_min_d, MU_rl2_min_d, MU_rl3_min_d, MU_rl4_min_d, MU_rl5_min_d]
rl_min_d_std = [SIGMA_rl1_min_d, SIGMA_rl2_min_d, SIGMA_rl3_min_d, 
                    SIGMA_rl4_min_d, SIGMA_rl5_min_d]

rects14 = ax3[0].bar(index+bar_width, rl_min_d, bar_width, yerr=rl_min_d_std, 
                    alpha=opacity, color=color_DDPG, capsize=10, label='DDPG',
                    edgecolor='k')

print()
print('***************************')
print('minimum psi_2')
print('***************************')
print()
print('~~~~~~~~~~~~~' + foldername.strip('/') + '_1 ~~~~~~~~~~~~~~~')
mc1_min_psi = abs(metrics1['mc_min_psi'].values)
rl1_min_psi = abs(metrics1['rl_min_psi'].values)
MU_mc1_min_psi, SIGMA_mc1_min_psi, MU_rl1_min_psi, SIGMA_rl1_min_psi = \
    welch_test(mc1_min_psi, rl1_min_psi, name1='rl1_min_psi', name2='rl1_min_psi')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_2 ~~~~~~~~~~~~~~~')
mc2_min_psi = abs(metrics2['mc_min_psi'].values)
rl2_min_psi = abs(metrics2['rl_min_psi'].values)
MU_mc2_min_psi, SIGMA_mc2_min_psi, MU_rl2_min_psi, SIGMA_rl2_min_psi = \
    welch_test(mc2_min_psi, rl2_min_psi, name1='rl2_min_psi', name2='rl2_min_psi')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_3 ~~~~~~~~~~~~~~~')
mc3_min_psi = abs(metrics3['mc_min_psi'].values)
rl3_min_psi = abs(metrics3['rl_min_psi'].values)
MU_mc3_min_psi, SIGMA_mc3_min_psi, MU_rl3_min_psi, SIGMA_rl3_min_psi = \
    welch_test(mc3_min_psi, rl3_min_psi, name1='rl3_min_psi', name2='rl3_min_psi')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_4 ~~~~~~~~~~~~~~~')
mc4_min_psi = abs(metrics4['mc_min_psi'].values)
rl4_min_psi = abs(metrics4['rl_min_psi'].values)
MU_mc4_min_psi, SIGMA_mc4_min_psi, MU_rl4_min_psi, SIGMA_rl4_min_psi = \
    welch_test(mc4_min_psi, rl4_min_psi, name1='rl4_min_psi', name2='rl4_min_psi')

print('~~~~~~~~~~~~~' + foldername.strip('/') + '_5 ~~~~~~~~~~~~~~~')
mc5_min_psi = abs(metrics5['mc_min_psi'].values)
rl5_min_psi = abs(metrics5['rl_min_psi'].values)
MU_mc5_min_psi, SIGMA_mc5_min_psi, MU_rl5_min_psi, SIGMA_rl5_min_psi = \
    welch_test(mc5_min_psi, rl5_min_psi, name1='rl5_min_psi', name2='rl5_min_psi')

mc_min_psi = [MU_mc1_min_psi, MU_mc2_min_psi, MU_mc3_min_psi, MU_mc4_min_psi, MU_mc5_min_psi]
mc_min_psi_std = [SIGMA_mc1_min_psi, SIGMA_mc2_min_psi, SIGMA_mc3_min_psi, 
                    SIGMA_mc4_min_psi, SIGMA_mc5_min_psi]
rects13 = ax3[1].bar(index, mc_min_psi, bar_width, yerr=mc_min_psi_std, 
                    alpha=opacity, color=color_LQR, capsize=10, label='LQR',
                    edgecolor='k')

rl_min_psi = [MU_rl1_min_psi, MU_rl2_min_psi, MU_rl3_min_psi, MU_rl4_min_psi, MU_rl5_min_psi]
rl_min_psi_std = [SIGMA_rl1_min_psi, SIGMA_rl2_min_psi, SIGMA_rl3_min_psi, 
                    SIGMA_rl4_min_psi, SIGMA_rl5_min_psi]

rects14 = ax3[1].bar(index+bar_width, rl_min_psi, bar_width, 
                    yerr=rl_min_psi_std, alpha=opacity, color=color_DDPG, 
                    capsize=10, label='DDPG', edgecolor='k')

plt.tight_layout()
ax3[0].legend()
plt.savefig('stat_rm_goal.eps', format='eps', dpi=1000)

# Histogram
'''n_bins = 40
fig4, ax4 = plt.subplots(1, 2, sharey=True, tight_layout=True)
ax4[0].hist(rms_mc1_psi_1, bins=n_bins)
ax4[0].set_ylabel('Frequency')
ax4[1].hist(rms_rl1_psi_1, bins=n_bins)
ax4[0].title.set_text('Modern Controls')
ax4[1].title.set_text('Reinforcement Learning')
ax4[0].set_xlabel(r'rms $\psi_{1} [rad]$')
ax4[1].set_xlabel(r'rms $\psi_{1} [rad]$')'''

# rms_mc_psi_1, rms_mc_psi_2, rms_mc_d2
# rms_rl_psi_1, rms_rl_psi_2, rms_rl_d2

# max_mc_psi_1, max_mc_psi_2, max_mc_d2
# max_rl_psi_1, max_rl_psi_2, max_rl_d2

# mc_min_d, mc_min_psi
# rl_min_d, rl_min_psi


## Percent error printouts between LQR and DDPG
def perce(experimental, theoretical):
    return ((experimental - theoretical) / abs(theoretical) ) * 100

print()
print('Percent Error between LQR and DDPG')
print(PARAMS[0])
print('psi_2: {:.2f}, y_2: {:.2f}'.format(perce(MU_rms_rl1_psi_2, 
                                                MU_rms_mc1_psi_2), 
                                          perce(MU_rms_rl1_d2,
                                                MU_rms_mc1_d2)))

print()
print(PARAMS[1])
print('psi_2: {:.2f}, y_2: {:.2f}'.format(perce(MU_rms_rl2_psi_2, 
                                                MU_rms_mc2_psi_2), 
                                          perce(MU_rms_rl2_d2,
                                                MU_rms_mc2_d2)))
print()
print(PARAMS[2])
print('psi_2: {:.2f}, y_2: {:.2f}'.format(perce(MU_rms_rl3_psi_2, 
                                                MU_rms_mc3_psi_2), 
                                          perce(MU_rms_rl3_d2,
                                                MU_rms_mc3_d2)))
print()
print(PARAMS[3])
print('psi_2: {:.2f}, y_2: {:.2f}'.format(perce(MU_rms_rl4_psi_2, 
                                                MU_rms_mc4_psi_2), 
                                          perce(MU_rms_rl4_d2,
                                                MU_rms_mc4_d2)))
print()
print(PARAMS[4])
print('psi_2: {:.2f}, y_2: {:.2f}'.format(perce(MU_rms_rl5_psi_2, 
                                                MU_rms_mc5_psi_2), 
                                          perce(MU_rms_rl5_d2,
                                                MU_rms_mc5_d2)))
print()
print("Percent error from nominal cases")
print(PARAMS[0])
perce_mc1_psi_2 = perce(MU_rms_mc1_psi_2, MU_rms_mc3_psi_2)
perce_rl1_psi_2 = perce(MU_rms_rl1_psi_2, MU_rms_rl3_psi_2)
print('psi_2 LQR: {:.2f}, DDPG: {:.2f}'.format(perce_mc1_psi_2, 
                                               perce_rl1_psi_2))

perce_mc1_y2 = perce(MU_rms_mc1_d2, MU_rms_mc3_d2)
perce_rl1_y2 = perce(MU_rms_rl1_d2, MU_rms_rl3_d2)
print('y_2 LQR: {:.2f}, DDPG: {:.2f}'.format(perce_mc1_y2, perce_rl1_y2))

print()
print(PARAMS[1])
perce_mc2_psi_2 = perce(MU_rms_mc2_psi_2, MU_rms_mc3_psi_2)
perce_rl2_psi_2 = perce(MU_rms_rl2_psi_2, MU_rms_rl3_psi_2)
print('psi_2 LQR: {:.2f}, DDPG: {:.2f}'.format(perce_mc2_psi_2,
                                               perce_rl2_psi_2))

perce_mc2_y2 = perce(MU_rms_mc2_d2, MU_rms_mc3_d2)
perce_rl2_y2 = perce(MU_rms_rl2_d2, MU_rms_rl3_d2)
print('y_2 LQR: {:.2f}, DDPG: {:.2f}'.format(perce_mc2_y2, perce_rl2_y2))

print()
print(PARAMS[3])
perce_mc4_psi_2 = perce(MU_rms_mc4_psi_2, MU_rms_mc3_psi_2)
perce_rl4_psi_2 = perce(MU_rms_rl4_psi_2, MU_rms_rl3_psi_2)
print('psi_2 LQR: {:.2f}, DDPG: {:.2f}'.format(perce_mc4_psi_2,
                                               perce_rl4_psi_2))

perce_mc4_y2 = perce(MU_rms_mc4_d2, MU_rms_mc3_d2)
perce_rl4_y2 = perce(MU_rms_rl4_d2, MU_rms_rl3_d2)
print('y_2 LQR: {:.2f}, DDPG: {:.2f}'.format(perce_mc4_y2, perce_rl4_y2))

print()
print(PARAMS[4])
perce_mc5_psi_2 = perce(MU_rms_mc5_psi_2, MU_rms_mc3_psi_2)
perce_rl5_psi_2 = perce(MU_rms_rl5_psi_2, MU_rms_rl3_psi_2)
print('psi_2 LQR: {:.2f}, DDPG: {:.2f}'.format(perce_mc5_psi_2,
                                               perce_rl5_psi_2))

perce_mc5_y2 = perce(MU_rms_mc5_d2, MU_rms_mc3_d2)
perce_rl5_y2 = perce(MU_rms_rl5_d2, MU_rms_rl3_d2)
print('y_2 LQR: {:.2f}, DDPG: {:.2f}'.format(perce_mc5_y2, perce_rl5_y2))

## Plot radar chart
N = len(PARAMS) - 1
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

radar_labels = [str(PARAMS[0]) + 'm', str(PARAMS[1]) + 'm', 
                str(PARAMS[3]) + 'm', str(PARAMS[4]) + 'm']

fig_radar = plt.figure(figsize=(10, 10))
ax_radar = plt.subplot(1, 1, 1, polar=True)

# psi_2
ax_radar.set_theta_offset(np.pi / 4)
ax_radar.set_theta_direction(-1)

plt.xticks(angles[:-1], radar_labels, size=16)
ax_radar.tick_params(axis='both', which='major', pad=20)
ax_radar.set_rlabel_position(3)
plt.yticks([5, 10, 15, 20], ["5%", "10%", "15%", "20%"], color='grey', size=14)
plt.ylim(0, 25)

#LQR
perce_lqr_psi_2 = [abs(perce_mc1_psi_2), abs(perce_mc2_psi_2), 
                   abs(perce_mc4_psi_2), abs(perce_mc5_psi_2)]
perce_lqr_psi_2 += perce_lqr_psi_2[:1]
ax_radar.plot(angles, perce_lqr_psi_2, 'g', linewidth=1, linestyle='solid', 
              label=r"LQR $\psi_{2e}$")
ax_radar.fill(angles, perce_lqr_psi_2, 'g', alpha=0.1)

#DDPG
perce_ddpg_psi_2 = [abs(perce_rl1_psi_2), abs(perce_rl2_psi_2), 
                   abs(perce_rl4_psi_2), abs(perce_rl5_psi_2)]
perce_ddpg_psi_2 += perce_ddpg_psi_2[:1]
ax_radar.plot(angles, perce_ddpg_psi_2, 'b', linewidth=1, linestyle='solid', 
              label=r"DDPG $\psi_{2e}$")
ax_radar.fill(angles, perce_ddpg_psi_2, 'b', alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.06))

plt.tight_layout()
plt.savefig('radar_psi_2.pdf', format='pdf', dpi=1000)

## y2
fig_radar2 = plt.figure(figsize=(10, 10))
ax_radar2 = plt.subplot(1, 1, 1, polar=True)

ax_radar2.set_theta_offset(np.pi / 4)
ax_radar2.set_theta_direction(-1)

plt.xticks(angles[:-1], radar_labels, size=16)
ax_radar2.tick_params(axis='both', which='major', pad=20)
ax_radar2.set_rlabel_position(3)
plt.yticks([5, 10, 15, 20], ["5%", "10%", "15%", "20%"], color='grey', size=14)
plt.ylim(0, 25)

#LQR
perce_lqr_y2 = [abs(perce_mc1_y2), abs(perce_mc2_y2), 
                   abs(perce_mc4_y2), abs(perce_mc5_y2)]
perce_lqr_y2 += perce_lqr_y2[:1]
ax_radar2.plot(angles, perce_lqr_y2, 'g', linewidth=1, linestyle='solid', 
              label=r"LQR $y_{2e}$")
ax_radar2.fill(angles, perce_lqr_y2, 'g', alpha=0.1)

#DDPG
perce_ddpg_y2 = [abs(perce_rl1_y2), abs(perce_rl2_y2), 
                   abs(perce_rl4_y2), abs(perce_rl5_y2)]
perce_ddpg_y2 += perce_ddpg_y2[:1]
ax_radar2.plot(angles, perce_ddpg_y2, 'b', linewidth=1, linestyle='solid', 
              label=r"DDPG $y_{2e}$")
ax_radar2.fill(angles, perce_ddpg_y2, 'b', alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.06))

plt.tight_layout()
plt.savefig('radar_y2.pdf', format='pdf', dpi=1000)

plt.show()
