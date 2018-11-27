import gym
import random
import gym.spaces
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
from datetime import datetime
import gym_truck_backerupper
import time
import csv
import sys
import pandas as pd
import os
import select

SEEDS = [0]
LABEL = 'trailer'
K = np.array([-3.7896, 12.8011, -1.0])
DEMONSTRATIONS = 2

def test_mc(env, K, mc_t_log, mc_psi_1_log, mc_psi_2_log, mc_d2_log, start_rendering): 
    done = False
    s = env.reset()
    q0 = env.q0
    qg = env.qg
    total_reward = 0.0
    steps = 0
    while not done:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            command = input()
            if command == 'render':
                start_rendering = True
                print('set render...')
            else:
                start_rendering = False
                env.close()
                print('hide render...')
        if start_rendering:
            env.render()
        a = K.dot(s[0:3]) + s[3]
        s_, r, done, info = env.step(a)
        
        if done:
            print()
            for key, value in info.items():
                if value:
                    print(key, value)
            print()

        mc_t_log.append(env.t[steps])
        mc_psi_1_log.append(s_[0])
        mc_psi_2_log.append(s_[1])
        mc_d2_log.append(s_[2])

        s = s_
        total_reward += r
        steps += 1
    return total_reward, info, q0, qg, start_rendering

def test_rl(env, policy, state, train_phase, sess, 
            rl_psi_1_log, rl_psi_2_log, rl_d2_log, rl_t_log, q0, qg, start_rendering):
    done = False
    q0_ = q0.copy()
    qg_ = qg.copy()
    if env.v < 0:
        q0_[2] -= np.pi
        qg_[2] -= np.pi
    q0_[2] = np.degrees(q0_[2])
    qg_[2] = np.degrees(qg_[2])
    env.manual_course(q0_, qg_)
    s = env.reset()
    env.manual_track = False
    total_reward = 0.0
    steps = 0
    while not done:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            command = input()
            if command == 'render':
                start_rendering = True
                print('set render...')
            else:
                start_rendering = False
                env.close()
                print('hide render...')
        if start_rendering:
            env.render()
        a = sess.run(policy, feed_dict={state: s.reshape(1, s.shape[0]),
                                        train_phase: False})
        s_, r, done, info = env.step(a)
        
        if done:
            print()
            for key, value in info.items():
                if value:
                    print(key, value)
            print()
        
        rl_t_log.append(env.t[steps])
        rl_psi_1_log.append(s_[0])
        rl_psi_2_log.append(s_[1])
        rl_d2_log.append(s_[2])

        s = s_
        total_reward += r
        steps += 1
    return total_reward, info

def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))

def abs_max(x):
    return max(np.min(x), np.max(x), key=abs)


if __name__ == '__main__':
    env = gym.make('TruckBackerUpper-v0').unwrapped
    #env.manual_velocity(-25.0)
    for seed_idx in range(len(SEEDS)):
        checkpoint_path = "./models/" + LABEL + "_seed_" + str(
                          SEEDS[seed_idx]) + "/my_ddpg.ckpt"
        np.random.seed(SEEDS[seed_idx])
        tf.set_random_seed(SEEDS[seed_idx])
        env.seed(SEEDS[seed_idx])
        
        with tf.Session(graph=tf.Graph()) as sess:
            ## Modern Controls
            rms_mc_psi_1 = []
            rms_mc_psi_2 = []
            rms_mc_d2 = []
            max_mc_psi_1 = []
            max_mc_psi_2 = []
            max_mc_d2 = []
            mc_goal_flag = []
            mc_jackknife = []
            mc_out_of_bounds = []
            mc_times_up = []
            mc_fin = []
            mc_min_d = []
            mc_min_psi = []
            mc_t = []

            ## Reinforcement Learning
            rms_rl_psi_1 = []
            rms_rl_psi_2 = []
            rms_rl_d2 = []
            max_rl_psi_1 = []
            max_rl_psi_2 = []
            max_rl_d2 = []
            rl_goal_flag = []
            rl_jackknife = []
            rl_out_of_bounds = []
            rl_times_up = []
            rl_fin = []
            rl_min_d = []
            rl_min_psi = []
            rl_t = []
            saved = tf.train.import_meta_graph(checkpoint_path + '.meta', clear_devices=True)
            saved.restore(sess, checkpoint_path)
            state = sess.graph.get_tensor_by_name('Actor/s:0')
            train_phase = sess.graph.get_tensor_by_name('Actor/train_phase_actor:0')
            learned_policy = sess.graph.get_tensor_by_name(
                                    'Actor/pi_online_network/pi_hat/Mul_4:0')
            start_rendering = False

            for demo in range(DEMONSTRATIONS):
                ## Modern Controls
                mc_psi_1_log = []
                mc_psi_2_log = []
                mc_d2_log = []
                mc_t_log = []
                r, info, q0, qg, start_rendering = test_mc(
                                env, K, mc_t_log, mc_psi_1_log, mc_psi_2_log, 
                                mc_d2_log, start_rendering)

                rms_mc_psi_1.append(rms(mc_psi_1_log))
                rms_mc_psi_2.append(rms(mc_psi_2_log))
                rms_mc_d2.append(rms(mc_d2_log))
                max_mc_psi_1.append(abs_max(mc_psi_1_log))
                max_mc_psi_2.append(abs_max(mc_psi_2_log))
                max_mc_d2.append(abs_max(mc_psi_2_log))
                mc_goal_flag.append(info['goal'])
                mc_jackknife.append(info['jackknife'])
                mc_out_of_bounds.append(info['out_of_bounds'])
                mc_times_up.append(info['times_up'])
                mc_fin.append(info['fin'])
                mc_min_d.append(info['min_d'])
                mc_min_psi.append(info['min_psi'])
                mc_t.append(info['t'])
                df = pd.DataFrame({'time' : mc_t_log,
                                   'mc_psi_1' : mc_psi_1_log, 
                                   'mc_psi_2' : mc_psi_2_log, 
                                   'mc_d2' : mc_d2_log})
                if not os.path.exists('./run' + str(demo)):
                    os.mkdir('./run' + str(demo))
                with open('./run' + str(demo) + '/' + 'mc' + 
                          str(mc_goal_flag[demo]) + '.txt', 'w') as mc_filename:
                    mc_filename.write('# goal: {}\n'.format(mc_goal_flag[demo]) +
                                      '# jackknife: {}\n'.format(mc_jackknife[demo]) + 
                                      '# out_of_bounds: {}\n'.format(mc_out_of_bounds[demo]) +
                                      '# times_up: {}\n'.format(mc_times_up[demo]) +
                                      '# fin: {}\n'.format(mc_fin[demo]) +
                                      '# t: {:.3f}\n'.format(mc_t[demo]) +
                                      '# min_d: {:.3f} \n'.format(mc_min_d[demo]) +
                                      '# min_psi: {:.3f}\n\n'.format(mc_min_psi[demo]) +
                                      '# rms_psi_1: {:.3f}\n'.format(rms_mc_psi_1[demo]) +
                                      '# rms_psi_2: {:.3f}\n'.format(rms_mc_psi_2[demo]) +
                                      '# rms_d2: {:.3f}\n'.format(rms_mc_d2[demo]) +
                                      '# max_psi_1: {:.3f}\n'.format(max_mc_psi_1[demo]) +
                                      '# max_psi_2: {:.3f}\n'.format(max_mc_psi_2[demo]) +
                                      '# max_d2: {:.3f}\n'.format(max_mc_d2[demo]) +
                                      '# reward: {:.3f}\n'.format(r) + 
                                      '# q0: {}\n'.format(q0) + 
                                      '# qg: {}\n\n'.format(qg))
                    
                    df.to_csv(mc_filename, sep='\t', index=False, mode='a')

                ## Reinforcement Learning
                rl_psi_1_log = []
                rl_psi_2_log = []
                rl_d2_log = []
                rl_t_log = []
                r, info = test_rl(env, learned_policy, state, train_phase, sess,
                            rl_psi_1_log, rl_psi_2_log, rl_d2_log, rl_t_log, q0, qg,
                            start_rendering)
                rms_rl_psi_1.append(rms(rl_psi_1_log))
                rms_rl_psi_2.append(rms(rl_psi_2_log))
                rms_rl_d2.append(rms(rl_d2_log))
                max_rl_psi_1.append(abs_max(rl_psi_1_log))
                max_rl_psi_2.append(abs_max(rl_psi_2_log))
                max_rl_d2.append(abs_max(rl_psi_2_log))
                rl_goal_flag.append(info['goal'])
                rl_jackknife.append(info['jackknife'])
                rl_out_of_bounds.append(info['out_of_bounds'])
                rl_times_up.append(info['times_up'])
                rl_fin.append(info['fin'])
                rl_min_d.append(info['min_d'])
                rl_min_psi.append(info['min_psi'])
                rl_t.append(info['t'])
                df = pd.DataFrame({'time' : rl_t_log,
                                   'rl_psi_1' : rl_psi_1_log, 
                                   'rl_psi_2' : rl_psi_2_log, 
                                   'rl_d2' : rl_d2_log})
                with open('./run' + str(demo) + '/' + 'rl' + 
                          str(rl_goal_flag[demo]) + '.txt', 'w') as rl_filename:
                    rl_filename.write('# goal: {}\n'.format(rl_goal_flag[demo]) +
                                      '# jackknife: {}\n'.format(rl_jackknife[demo]) + 
                                      '# out_of_bounds: {}\n'.format(rl_out_of_bounds[demo]) +
                                      '# times_up: {}\n'.format(rl_times_up[demo]) +
                                      '# fin: {}\n'.format(rl_fin[demo]) +
                                      '# t: {:.3f}\n'.format(rl_t[demo]) +
                                      '# min_d: {:.3f} \n'.format(rl_min_d[demo]) +
                                      '# min_psi: {:.3f}\n\n'.format(rl_min_psi[demo]) +
                                      '# rms_psi_1: {:.3f}\n'.format(rms_rl_psi_1[demo]) +
                                      '# rms_psi_2: {:.3f}\n'.format(rms_rl_psi_2[demo]) +
                                      '# rms_d2: {:.3f}\n'.format(rms_rl_d2[demo]) +
                                      '# max_psi_1: {:.3f}\n'.format(max_rl_psi_1[demo]) +
                                      '# max_psi_2: {:.3f}\n'.format(max_rl_psi_2[demo]) +
                                      '# max_d2: {:.3f}\n'.format(max_rl_d2[demo]) +
                                      '# reward: {:.3f}\n\n'.format(r) +
                                      '# q0: {}\n'.format(q0) + 
                                      '# qg: {}\n\n'.format(qg))
                    
                    df.to_csv(rl_filename, sep='\t', index=False, mode='a')

            env.close()
        tf.reset_default_graph()
