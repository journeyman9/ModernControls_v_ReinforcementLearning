import gym
import random
import gym.spaces
import numpy as np
import tensorflow as tf
tf.contrib.resampler
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
import scipy.stats as stats
import matplotlib.pyplot as plt
import ast


SEED = 9
SEED_ID = [0]
LABEL = 'lp17_3_to_25'

#PARAM_LABEL = 'wheelbase'
#PARAMS = [8.192, 9.192, 10.192, 11.192, 12.192]

#PARAM_LABEL = 'hitch'
#PARAMS = [0.228, 0.114, 0.000, -0.114, -0.228]

#PARAM_LABEL = 'velocity'
#PARAMS = [-2.906, -2.459, -2.012, -1.564, -1.118]

#PARAM_LABEL = 'sensor_noise'
#PARAMS = [0.0, .03, .04, .05, .06]

PARAM_LABEL = 'control_frequency'
PARAMS = [.001, .010, .080, .400, .500, .600]

K = np.array([-24.7561, 94.6538, -7.8540]) 
DEMONSTRATIONS = 100

def test_mc(env, K, mc_t_log, mc_psi_1_log, mc_psi_2_log, mc_d2_log, mc_a_log,
            start_rendering, lesson_idx): 
    done = False
    if len(lesson_plan) > 0:
        course = lesson_plan[lesson_idx]
        env.manual_course(course[0], course[1])
        lesson_idx += 1
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
        
        if PARAM_LABEL == 'control_frequency':
            if (env.sim_i-1) % int(PARAMS[param_idx] / env.dt) == 0:
                a = np.clip(K.dot(s), env.action_space.low, 
                            env.action_space.high)
                a_last = a.copy()
            else:
                a = a_last.copy()
        else:
            a = np.clip(K.dot(s), env.action_space.low, 
                        env.action_space.high)
        s_, r, done, info = env.step(a)
        
        if done:
            print()
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("LQR")
            env.tog ^= 1
            for key, value in info.items():
                if value:
                    print(key, value)
            print()

        mc_t_log.append(env.t[steps])
        mc_psi_1_log.append(s_[0])
        mc_psi_2_log.append(s_[1])
        mc_d2_log.append(s_[2])
        mc_a_log.append(a)

        s = s_
        total_reward += r
        steps += 1
    return total_reward, info, q0, qg, start_rendering, lesson_idx

def test_rl(env, policy, q_value, state_a, state_c, action, train_phase_a, train_phase_c, sess, 
            rl_psi_1_log, rl_psi_2_log, rl_d2_log, rl_t_log, rl_a_log, rl_q_log, 
            q0, qg, start_rendering):
    done = False
    q0_ = q0.copy()
    qg_ = qg.copy()
    if env.v1x < 0:
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
        
        if PARAM_LABEL == 'control_frequency':
            if (env.sim_i-1) % int(PARAMS[param_idx] / env.dt) == 0:
                a = sess.run(policy, feed_dict={state_a: s.reshape(1, 
                             s.shape[0]),train_phase_a: False})[0]
                a_last = a.copy()
            else:
                a = a_last.copy()
        else:
            a = sess.run(policy, feed_dict={state_a: s.reshape(1, s.shape[0]),
                                            train_phase_a: False})[0]
        s_, r, done, info = env.step(a)
        
        if done:
            print()
            print("DDPG")
            env.tog ^= 1
            for key, value in info.items():
                if value:
                    print(key, value)
            print()
        
        rl_t_log.append(env.t[steps])
        rl_psi_1_log.append(s_[0])
        rl_psi_2_log.append(s_[1])
        rl_d2_log.append(s_[2])
        rl_a_log.append(a[0])
        rl_q_log.append(sess.run(q_value, feed_dict={state_c: s.reshape(1, s.shape[0]),
                                                     action: a.reshape(1, a.shape[0]),
                                                     train_phase_c: False})[0, 0])

        s = s_
        total_reward += r
        steps += 1
    return total_reward, info, start_rendering

def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))

def abs_max(x):
    return max(np.min(x), np.max(x), key=abs)


if __name__ == '__main__':
    env = gym.make('TruckBackerUpper-v0').unwrapped
    #env.manual_params(L2=12.192, h=-0.23)
    #env.manual_velocity(-25.0)
    start_rendering = False
    env.tog ^= 1
    lesson_plan = []
    
    if len(sys.argv) >=2:
        with open(sys.argv[1], newline='') as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\n')
            for row in readCSV:
                if row[0].startswith('#'):
                    continue
                else:
                    lesson_plan.append(ast.literal_eval(row[0]))
            print('~~~~~~~~~~~~~~~~~~~~~')
            print('Lesson Planned')
            print('~~~~~~~~~~~~~~~~~~~~~')
        DEMONSTRATIONS = len(lesson_plan)
         
    for param_idx in range(len(PARAMS)):
        ''' ''' 
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        env.seed(SEED)
        lesson_idx = 0
        if PARAM_LABEL == 'wheelbase':
            env.manual_params(L2=PARAMS[param_idx], h=-0.29)
        elif PARAM_LABEL == 'hitch':
            env.manual_params(L2=10.192, h=PARAMS[param_idx])
        elif PARAM_LABEL == 'velocity':
            env.manual_velocity(v=PARAMS[param_idx])
        elif PARAM_LABEL == 'sensor_noise':
            env.add_sensor_noise(sn=PARAMS[param_idx])
        elif PARAM_LABEL == 'control_frequency':
            env.dt = .001
            env.num_steps = int((env.t_final - env.t0)/env.dt) + 1
        else:
            pass
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
        mc_a = []
        mc_dist_too_large = []
        mc_angle_too_large = []

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
        rl_a = []
        rl_q = []
        rl_dist_too_large = []
        rl_angle_too_large = []
    
        for demo in range(DEMONSTRATIONS):
            ## Modern Controls
            mc_psi_1_log = []
            mc_psi_2_log = []
            mc_d2_log = []
            mc_t_log = []
            mc_a_log = []
            r, info, q0, qg, start_rendering, lesson_idx = test_mc(
                            env, K, mc_t_log, mc_psi_1_log, mc_psi_2_log, 
                            mc_d2_log, mc_a_log, start_rendering, lesson_idx)

            rms_mc_psi_1.append(rms(mc_psi_1_log))
            rms_mc_psi_2.append(rms(mc_psi_2_log))
            rms_mc_d2.append(rms(mc_d2_log))
            max_mc_psi_1.append(abs_max(mc_psi_1_log))
            max_mc_psi_2.append(abs_max(mc_psi_2_log))
            max_mc_d2.append(abs_max(mc_d2_log))
            mc_goal_flag.append(info['goal'])
            mc_jackknife.append(info['jackknife'])
            mc_out_of_bounds.append(info['out_of_bounds'])
            mc_times_up.append(info['times_up'])
            mc_fin.append(info['fin'])
            mc_min_d.append(info['min_d'])
            mc_min_psi.append(info['min_psi'])
            mc_t.append(info['t'])
            mc_dist_too_large.append(info['dist_too_large'])
            mc_angle_too_large.append(info['angle_too_large'])
            df = pd.DataFrame({'time' : mc_t_log,
                               'mc_psi_1' : mc_psi_1_log, 
                               'mc_psi_2' : mc_psi_2_log, 
                               'mc_d2' : mc_d2_log,
                               'mc_a' : mc_a_log})
            parent = './' + str(PARAM_LABEL)
            child1 =  './' + str(PARAM_LABEL) + '/' + str(PARAMS[param_idx]).replace(".", "_")
            child2 = './' + str(PARAM_LABEL) + '/' + str(PARAMS[param_idx]).replace(".", "_") + '/run' + str(demo)
            if not os.path.exists(parent):
                os.mkdir(parent)
            if not os.path.exists(child1):
                os.mkdir(child1)
            if not os.path.exists(child2):
                os.mkdir(child2)
            with open(child2 + '/' + 'mc' + str(mc_goal_flag[demo]) + '.txt', 'w') as mc_filename:
                mc_filename.write('# goal: {}\n'.format(mc_goal_flag[demo]) +
                                  '# jackknife: {}\n'.format(mc_jackknife[demo]) + 
                                  '# out_of_bounds: {}\n'.format(mc_out_of_bounds[demo]) +
                                  '# times_up: {}\n'.format(mc_times_up[demo]) +
                                  '# fin: {}\n'.format(mc_fin[demo]) +
                                  '# t: {:.3f}\n'.format(mc_t[demo]) +
                                  '# dist_too_large: {}\n'.format(mc_dist_too_large[demo]) +
                                  '# angle_too_large: {}\n'.format(mc_angle_too_large[demo]) +
                                  '# min_d: {:.3f} \n'.format(mc_min_d[demo]) +
                                  '# min_psi: {:.3f}\n\n'.format(mc_min_psi[demo]) +
                                  '# rms_psi_1: {:.3f}\n'.format(rms_mc_psi_1[demo]) +
                                  '# rms_psi_2: {:.3f}\n'.format(rms_mc_psi_2[demo]) +
                                  '# rms_d2: {:.3f}\n'.format(rms_mc_d2[demo]) +
                                  '# max_psi_1: {:.3f}\n'.format(max_mc_psi_1[demo]) +
                                  '# max_psi_2: {:.3f}\n'.format(max_mc_psi_2[demo]) +
                                  '# max_d2: {:.3f}\n'.format(max_mc_d2[demo]) +
                                  '# reward: {:.3f}\n\n'.format(r) + 
                                  '# q0: {}\n'.format(q0) + 
                                  '# qg: {}\n\n'.format(qg))
                
                df.to_csv(mc_filename, sep='\t', index=False, mode='a')

            ## Reinforcement Learning
            for seed_idx in range(len(SEED_ID)):
                print('RL TRAINED SEED {}'.format(seed_idx+1))
                checkpoint_path = "./models/" + LABEL + "_seed_" + str(
                                  SEED_ID[seed_idx]) + "/my_ddpg.ckpt"
                with tf.Session(graph=tf.Graph()) as sess:
                    saved = tf.train.import_meta_graph(
                                checkpoint_path + '.meta', clear_devices=True)
                    saved.restore(sess, checkpoint_path)
                    state_a = sess.graph.get_tensor_by_name('Actor/s:0')
                    state_c = sess.graph.get_tensor_by_name('Critic/s:0')
                    action = sess.graph.get_tensor_by_name('Critic/a:0')
                    q_value = sess.graph.get_tensor_by_name('Critic/Q_online_network/Q_hat/add:0')
                    train_phase_a = sess.graph.get_tensor_by_name('Actor/train_phase_actor:0')
                    train_phase_c = sess.graph.get_tensor_by_name('Critic/train_phase_critic:0')
                    learned_policy = sess.graph.get_tensor_by_name(
                                            'Actor/pi_online_network/pi_hat/Mul_4:0')
                    rl_psi_1_log = []
                    rl_psi_2_log = []
                    rl_d2_log = []
                    rl_t_log = []
                    rl_a_log = []
                    rl_q_log = []
                    r, info, start_rendering = test_rl(env, learned_policy, 
                                      q_value, state_a, state_c, action, 
                                      train_phase_a, train_phase_c, sess, 
                                      rl_psi_1_log, rl_psi_2_log,
                                      rl_d2_log, rl_t_log, rl_a_log, rl_q_log, 
                                      q0, qg, start_rendering)
                    if seed_idx < 1:
                        rms_rl_psi_1.append(rms(rl_psi_1_log))
                        rms_rl_psi_2.append(rms(rl_psi_2_log))
                        rms_rl_d2.append(rms(rl_d2_log))
                        max_rl_psi_1.append(abs_max(rl_psi_1_log))
                        max_rl_psi_2.append(abs_max(rl_psi_2_log))
                        max_rl_d2.append(abs_max(rl_d2_log))
                        rl_goal_flag.append(info['goal'])
                        rl_jackknife.append(info['jackknife'])
                        rl_out_of_bounds.append(info['out_of_bounds'])
                        rl_times_up.append(info['times_up'])
                        rl_fin.append(info['fin'])
                        rl_min_d.append(info['min_d'])
                        rl_min_psi.append(info['min_psi'])
                        rl_t.append(info['t'])
                        rl_dist_too_large.append(info['dist_too_large'])
                        rl_angle_too_large.append(info['angle_too_large'])
                    df = pd.DataFrame({'time' : rl_t_log,
                                       'rl_psi_1' : rl_psi_1_log, 
                                       'rl_psi_2' : rl_psi_2_log, 
                                       'rl_d2' : rl_d2_log, 
                                       'rl_a' : rl_a_log,
                                       'rl_q' : rl_q_log})
                    if seed_idx < 1:
                        with open(child2 + '/' + 'rl' + str(rl_goal_flag[demo]) + 
                                  '.txt', 'w') as rl_filename:
                            rl_filename.write('# goal: {}\n'.format(rl_goal_flag[demo]) +
                                              '# jackknife: {}\n'.format(rl_jackknife[demo]) + 
                                              '# out_of_bounds: {}\n'.format(
                                                                    rl_out_of_bounds[demo]) +
                                              '# times_up: {}\n'.format(rl_times_up[demo]) +
                                              '# fin: {}\n'.format(rl_fin[demo]) +
                                              '# t: {:.3f}\n'.format(rl_t[demo]) +
                                              '# dist_too_large: {}\n'.format(rl_dist_too_large[demo]) + 
                                              '# angle_too_large: {}\n'.format(rl_angle_too_large[demo]) +
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
                    else:
                        with open(child2 + '/' + 'rl' + str(info['goal']) + '_' + 
                                  str(SEED_ID[seed_idx]) +'.txt', 
                                                                'w') as rl_filename:
                            rl_filename.write('# goal: {}\n'.format(info['goal']) +
                                              '# jackknife: {}\n'.format(info['jackknife']) + 
                                              '# out_of_bounds: {}\n'.format(
                                                                        info['out_of_bounds']) +
                                              '# times_up: {}\n'.format(info['times_up']) +
                                              '# fin: {}\n'.format(info['fin']) +
                                              '# t: {:.3f}\n'.format(info['t']) +
                                              '# min_d: {:.3f} \n'.format(info['min_d']) +
                                              '# min_psi: {:.3f}\n\n'.format(info['min_psi']) +
                                              '# rms_psi_1: {:.3f}\n'.format(rms(rl_psi_1_log)) +
                                              '# rms_psi_2: {:.3f}\n'.format(rms(rl_psi_2_log)) +
                                              '# rms_d2: {:.3f}\n'.format(rms(rl_d2_log)) +
                                              '# max_psi_1: {:.3f}\n'.format(
                                                                        abs_max(rl_psi_1_log)) +
                                              '# max_psi_2: {:.3f}\n'.format(
                                                                        abs_max(rl_psi_2_log)) +
                                              '# max_d2: {:.3f}\n'.format(abs_max(rl_d2_log)) +
                                              '# reward: {:.3f}\n\n'.format(r) +
                                              '# q0: {}\n'.format(q0) + 
                                              '# qg: {}\n\n'.format(qg))
                            
                            df.to_csv(rl_filename, sep='\t', index=False, mode='a')

                tf.reset_default_graph()

            #env.close()

        ## Stats -- null hypothesis = equal means assuming equal variance
        # if p-val is smaller than .05, reject null hypothesis

        # rms_mc_psi_1, rms_mc_psi_2, rms_mc_d2
        # rms_rl_psi_1, rms_rl_psi_2, rms_rl_d2

        # max_mc_psi_1, max_mc_psi_2, max_mc_d2
        # max_rl_psi_1, max_rl_psi_2, max_rl_d2

        # mc_min_d, mc_min_psi
        # rl_min_d, rl_min_psi

        df = pd.DataFrame({'rms_mc_psi_1' : rms_mc_psi_1, 'rms_mc_psi_2' : rms_mc_psi_2, 
                           'rms_mc_d2' : rms_mc_d2, 
                           'rms_rl_psi_1' : rms_rl_psi_1, 'rms_rl_psi_2': rms_rl_psi_2,
                           'rms_rl_d2' : rms_rl_d2,
                           'max_mc_psi_1' : max_mc_psi_1, 'max_mc_psi_2' : max_mc_psi_2,
                           'max_mc_d2' : max_mc_d2, 
                           'max_rl_psi_1': max_rl_psi_1, 'max_rl_psi_2' : max_rl_psi_2,
                           'max_rl_d2' : max_rl_d2,
                           'mc_min_d': mc_min_d, 'mc_min_psi' : mc_min_psi,
                           'rl_min_d': rl_min_d, 'rl_min_psi' : rl_min_psi,
                           'mc_goal_flag' : mc_goal_flag, 'mc_jackknife' : mc_jackknife,
                           'mc_out_of_bounds' : mc_out_of_bounds, 'mc_times_up' : mc_times_up,
                           'mc_dist_too_large' : mc_dist_too_large,
                           'mc_angle_too_large' : mc_angle_too_large,
                           'mc_fin' : mc_fin, 'mc_t': mc_t, 
                           'rl_goal_flag' : rl_goal_flag, 'rl_jackknife' : rl_jackknife,
                           'rl_out_of_bounds' : rl_out_of_bounds, 'rl_times_up' : rl_times_up,
                           'rl_dist_too_large' : rl_dist_too_large,
                           'rl_angle_too_large' : rl_angle_too_large,
                           'rl_fin' : rl_fin, 'rl_t' : rl_t})
        
        with open('./' + str(PARAM_LABEL) + '/' + 'stat_me_' + PARAM_LABEL + '_' + 
                  str(PARAMS[param_idx]).replace(".", "_") + '.txt', 'w') as filename:
            filename.write('# mc_goal: {}\n'.format(sum(mc_goal_flag)) +
                              '# mc_jackknife: {}\n'.format(sum(mc_jackknife)) + 
                              '# mc_out_of_bounds: {}\n'.format(sum(mc_out_of_bounds)) +
                              '# mc_times_up: {}\n'.format(sum(mc_times_up)) +
                              '# mc_dist_too_large: {}\n'.format(sum(mc_dist_too_large)) +
                              '# mc_angle_too_large: {}\n'.format(sum(mc_angle_too_large)) + 
                              '# mc_fin: {}\n\n'.format(sum(mc_fin)) + 
                              '# rl_goal: {}\n'.format(sum(rl_goal_flag)) +
                              '# rl_jackknife: {}\n'.format(sum(rl_jackknife)) + 
                              '# rl_out_of_bounds: {}\n'.format(sum(rl_out_of_bounds)) +
                              '# rl_times_up: {}\n'.format(sum(rl_times_up)) +
                              '# rl_dist_too_large: {}\n'.format(sum(rl_dist_too_large)) +
                              '# rl_angle_too_large: {}\n'.format(sum(rl_angle_too_large)) +
                              '# rl_fin: {}\n\n'.format(sum(rl_fin)) )

            df.to_csv(filename, sep='\t', index=False, mode='a')
