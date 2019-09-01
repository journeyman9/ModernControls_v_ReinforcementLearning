import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

foldername = sys.argv[1]

PARAMS = [8.192, 9.192, 10.192, 11.192, 12.192]
#PARAMS = [0.228, 0.114, 0.000, -0.114, -0.228]
#PARAMS = [-2.906, -2.459, -2.012, -1.564, -1.118]
#PARAMS = [0.0, .03, .04, .05, .06]
#PARAMS = [.001, .010, .080, .400, .600]

stats1 = get_param('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + str(PARAMS[0]).replace(".", "_") + '.txt')


stats2 = get_param('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + str(PARAMS[1]).replace(".", "_") + '.txt')

stats3 = get_param('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + str(PARAMS[2]).replace(".", "_") + '.txt')

stats4 = get_param('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + str(PARAMS[3]).replace(".", "_") + '.txt')

stats5 = get_param('./' + foldername + 'stat_me_' + foldername.strip('/') + '_' + str(PARAMS[4]).replace(".", "_") + '.txt')

print()
print("LQR Goal: {}, {}, {}, {}, {}".format(stats1["mc_goal"], stats2["mc_goal"], stats3["mc_goal"], stats4["mc_goal"], stats5["mc_goal"]))
print("DDPG Goal: {}, {}, {}, {}, {}".format(stats1["rl_goal"], stats2["rl_goal"], stats3["rl_goal"], stats4["rl_goal"], stats5["rl_goal"]))

print()
print("LQR Fin: {}, {}, {}, {}, {}".format(stats1["mc_fin"], stats2["mc_fin"], stats3["mc_fin"], stats4["mc_fin"], stats5["mc_fin"]))
print("DDPG Fin: {}, {}, {}, {}, {}".format(stats1["rl_fin"], stats2["rl_fin"], stats3["rl_fin"], stats4["rl_fin"], stats5["rl_fin"]))

print()
print("LQR Jackknife: {}, {}, {}, {}, {}".format(stats1["mc_jackknife"], stats2["mc_jackknife"], stats3["mc_jackknife"], stats4["mc_jackknife"], stats5["mc_jackknife"]))
print("DDPG Jackknife: {}, {}, {}, {}, {}".format(stats1["rl_jackknife"], stats2["rl_jackknife"], stats3["rl_jackknife"], stats4["rl_jackknife"], stats5["rl_jackknife"]))

print()
print("LQR Large Dist: {}, {}, {}, {}, {}".format(stats1["mc_dist_too_large"], stats2["mc_dist_too_large"], stats3["mc_dist_too_large"], stats4["mc_dist_too_large"], stats5["mc_dist_too_large"]))
print("DDPG Large Dist: {}, {}, {}, {}, {}".format(stats1["rl_dist_too_large"], stats2["rl_dist_too_large"], stats3["rl_dist_too_large"], stats4["rl_dist_too_large"], stats5["rl_dist_too_large"]))

print()
print("LQR Large Ang: {}, {}, {}, {}, {}".format(stats1["mc_angle_too_large"], stats2["mc_angle_too_large"], stats3["mc_angle_too_large"], stats4["mc_angle_too_large"], stats5["mc_angle_too_large"]))
print("DDPG Large Ang: {}, {}, {}, {}, {}".format(stats1["rl_angle_too_large"], stats2["rl_angle_too_large"], stats3["rl_angle_too_large"], stats4["rl_angle_too_large"], stats5["rl_angle_too_large"]))
