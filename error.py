import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys

def get_param(filename):
    para_dic = {}
    with open(filename, 'r') as cmt_file:
        for line in cmt_file:
            if line[0] == '#':
                line = line[1:]
                para = line.split(':')
            if len(para) == 2:
                para_dic[para[0].strip()] = para[1].strip()
    return para_dic

folder_name = sys.argv[1]

## MC
try:
    df_mc = pd.read_csv('./' + folder_name + '/mcTrue.txt', sep='\t', comment='#')
    mc_metrics = get_param('./' + folder_name + '/mcTrue.txt')
except:
    df_mc = pd.read_csv('./' + folder_name + '/mcFalse.txt', sep='\t', comment='#')
    mc_metrics = get_param('./' + folder_name + '/mcFalse.txt')

## RL
try:
    df_rl = pd.read_csv('./' + folder_name + '/rlTrue.txt', sep='\t', comment='#')
    rl_metrics = get_param('./' + folder_name + '/rlTrue.txt')
except:
    df_rl = pd.read_csv('./' + folder_name + '/rlFalse.txt', sep='\t', comment='#')
    rl_metrics = get_param('./' + folder_name + '/rlFalse.txt')

## MC Plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

ax1.plot(df_mc['time'], np.degrees(df_mc['mc_psi_1']), 'k', label='modern controls')
ax1.set_ylabel(r'$\psi_{1} [\degree]$')

ax2.plot(df_mc['time'], np.degrees(df_mc['mc_psi_1']), 'k')
ax2.set_ylabel(r'$\psi_{2} [\degree]$')

ax3.plot(df_mc['time'], df_mc['mc_d2'], 'k')
ax3.set_ylabel(r'$d_{2} [m]$')
ax3.set_xlabel('time [s]')

## RL Plots
ax1.plot(df_rl['time'], np.degrees(df_rl['rl_psi_1']), 'b', label='reinforcement learning')
ax2.plot(df_rl['time'], np.degrees(df_rl['rl_psi_1']), 'b')
ax3.plot(df_rl['time'], df_rl['rl_d2'], 'b')

## 0 line
t = int(float(mc_metrics['t']))
ax1.plot(range(t), 0*np.arange(t), 'r--')
ax2.plot(range(t), 0*np.arange(t), 'r--')
ax3.plot(range(t), 0*np.arange(t), 'r--')

ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, 
           borderaxespad=0, frameon=False)

## Metrics
# rms
fig4, ax4 = plt.subplots()
index = np.arange(3)
bar_width = 0.35 / 2
opacity = 0.8

mc_rms = [float(mc_metrics['rms_psi_1']), float(mc_metrics['rms_psi_2']), 
          float(mc_metrics['rms_d2'])]
rects1 = plt.bar(index, mc_rms, bar_width, alpha=opacity, color='k', label='modern controls')

rl_rms = [float(rl_metrics['rms_psi_1']), float(rl_metrics['rms_psi_2']), 
          float(rl_metrics['rms_d2'])]
rects2 = plt.bar(index+bar_width, rl_rms, bar_width, alpha=opacity, color='b', 
                 label='reinforcement learning')
plt.xlabel('Error Terms')
plt.ylabel('rms value')
plt.xticks(index+bar_width/2, (r'$\psi_{1} [rad]$', r'$\psi_{2} [rad]$', r'$d_{2} [m]$'))
plt.legend()
plt.tight_layout()

# max
fig5, ax5 = plt.subplots()
index = np.arange(3)
bar_width = 0.35 / 2
opacity = 0.8

mc_max = [abs(float(mc_metrics['max_psi_1'])), abs(float(mc_metrics['max_psi_2'])), 
          abs(float(mc_metrics['max_d2']))]
rects1 = plt.bar(index, mc_max, bar_width, alpha=opacity, color='k', label='Modern Controls')

rl_max = [abs(float(rl_metrics['max_psi_1'])), abs(float(rl_metrics['max_psi_2'])), 
          abs(float(rl_metrics['max_d2']))]
rects2 = plt.bar(index+bar_width, rl_max, bar_width, alpha=opacity, color='b', 
                 label='Reinforcement Learning')
plt.xlabel('Error Terms')
plt.ylabel('max value')
plt.xticks(index+bar_width/2, (r'$\psi_{1} [rad]$', r'$\psi_{2} [rad]$', r'$d_{2} [m]$'))
plt.legend()
plt.tight_layout()

# Goal, min_d, min_psi
fig6, ax6 = plt.subplots()
index = np.arange(3)
bar_width = 0.35 / 2
opacity = 0.8

if mc_metrics['goal'] == 'True':
    mc_goal_flag = 1.0
else:
    mc_goal_flag = 0.0

mc_goal = [mc_goal_flag, float(mc_metrics['min_d']), 
           float(mc_metrics['min_psi'])]
rects1 = plt.bar(index, mc_goal, bar_width, alpha=opacity, color='k', label='Modern Controls')

if rl_metrics['goal'] == 'True':
    rl_goal_flag = 1.0
else:
    rl_goal_flag = 0.0

rl_goal = [rl_goal_flag, float(rl_metrics['min_d']), 
           float(rl_metrics['min_psi'])]
rects2 = plt.bar(index+bar_width, rl_goal, bar_width, alpha=opacity, color='b', 
                 label='Reinforcement Learning')
plt.ylabel('Criteria')
plt.xticks(index+bar_width/2, ('Goal [bool]', r'$d2_{min} [m]$', r'$\psi2_{min} [rad]$'))
plt.legend()
plt.tight_layout()

plt.show()
