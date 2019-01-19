import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rcParams

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

plt.rcParams.update({'font.size':16})
rcParams['axes.labelpad'] = 10
size_fig = (10, 10)
color_LQR = 'g'
color_DDPG = 'b'

## MC Plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=size_fig, sharex=True)

ax1.plot(df_mc['time'], np.degrees(df_mc['mc_psi_1']), color_LQR, label='LQR')
ax1.set_ylabel(r'$\psi_{1} [\degree]$')

ax2.plot(df_mc['time'], np.degrees(df_mc['mc_psi_1']), color_LQR)
ax2.set_ylabel(r'$\psi_{2} [\degree]$')

ax3.plot(df_mc['time'], df_mc['mc_d2'], color_LQR)
ax3.set_ylabel(r'$y_{2} [m]$')
ax3.set_xlabel('time [s]')

## RL Plots
ax1.plot(df_rl['time'], np.degrees(df_rl['rl_psi_1']), color_DDPG, label='DDPG')
ax2.plot(df_rl['time'], np.degrees(df_rl['rl_psi_1']), color_DDPG)
ax3.plot(df_rl['time'], df_rl['rl_d2'], color_DDPG)

## 0 line
t = int(float(mc_metrics['t']))
ax1.plot(range(t), 0*np.arange(t), 'r--')
ax2.plot(range(t), 0*np.arange(t), 'r--')
ax3.plot(range(t), 0*np.arange(t), 'r--')

ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=1, 
           borderaxespad=0, frameon=False)

plt.savefig('error.eps', format='eps', dpi=1000)

## Metrics
# rms
fig4, ax4 = plt.subplots()
index = np.arange(3)
bar_width = 0.35 / 2
opacity = 0.8

mc_rms = [float(mc_metrics['rms_psi_1']), float(mc_metrics['rms_psi_2']), 
          float(mc_metrics['rms_d2'])]
rects1 = plt.bar(index, mc_rms, bar_width, alpha=opacity, color=color_LQR, 
                label='LQR', edgecolor='k')

rl_rms = [float(rl_metrics['rms_psi_1']), float(rl_metrics['rms_psi_2']), 
          float(rl_metrics['rms_d2'])]
rects2 = plt.bar(index+bar_width, rl_rms, bar_width, alpha=opacity, 
                 color=color_DDPG, label='DDPG', edgecolor='k')
plt.xlabel('Error Terms')
plt.ylabel('rms value')
plt.xticks(index+bar_width/2, (r'$\psi_{1} [rad]$', r'$\psi_{2} [rad]$', r'$y_{2} [m]$'))
plt.legend()
plt.tight_layout()

plt.savefig('single_rms.eps', format='eps', dpi=1000)

# max
fig5, ax5 = plt.subplots()
index = np.arange(3)
bar_width = 0.35 / 2
opacity = 0.8

mc_max = [abs(float(mc_metrics['max_psi_1'])), abs(float(mc_metrics['max_psi_2'])), 
          abs(float(mc_metrics['max_d2']))]
rects1 = plt.bar(index, mc_max, bar_width, alpha=opacity, color=color_LQR, 
                 label='LQR', edgecolor='k')

rl_max = [abs(float(rl_metrics['max_psi_1'])), abs(float(rl_metrics['max_psi_2'])), 
          abs(float(rl_metrics['max_d2']))]
rects2 = plt.bar(index+bar_width, rl_max, bar_width, alpha=opacity, 
                 color=color_DDPG, label='DDPG', edgecolor='k')
plt.xlabel('Error Terms')
plt.ylabel('max value')
plt.xticks(index+bar_width/2, (r'$\psi_{1} [rad]$', r'$\psi_{2} [rad]$', r'$y_{2} [m]$'))
plt.legend()
plt.tight_layout()

plt.savefig('single_max.eps', format='eps', dpi=1000)

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
           abs(float(mc_metrics['min_psi']))]
rects1 = plt.bar(index, mc_goal, bar_width, alpha=opacity, color=color_LQR, 
                 label='LQR', edgecolor='k')

if rl_metrics['goal'] == 'True':
    rl_goal_flag = 1.0
else:
    rl_goal_flag = 0.0

rl_goal = [rl_goal_flag, float(rl_metrics['min_d']), 
           abs(float(rl_metrics['min_psi']))]
rects2 = plt.bar(index+bar_width, rl_goal, bar_width, alpha=opacity, 
                 color=color_DDPG, label='DDPG', edgecolor='k')
plt.ylabel('Criteria')
plt.xticks(index+bar_width/2, ('Goal [bool]', r'$d2_{min} [m]$', r'$\psi2_{min} [rad]$'))
plt.legend()
plt.tight_layout()

plt.savefig('single_criteria.eps', format='eps', dpi=1000)

## 3D surface of states and Q
fig_q = plt.figure(figsize=(20, 10))
ax_q = Axes3D(fig_q)
surf_q = ax_q.plot_trisurf(df_rl.rl_d2, df_rl.rl_psi_2, df_rl.rl_q, 
                         cmap='viridis', linewidths=0.2)
fig_q.colorbar(surf_q, shrink=0.5, aspect=5)
ax_q.set_xlabel(r'$d2_{min} [m]$')
ax_q.set_ylabel(r'$\psi2_{min} [rad]$')
ax_q.set_zlabel('Q')

plt.savefig('Q_surface.eps', format='eps', dpi=1000)

plt.show()
