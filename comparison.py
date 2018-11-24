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

SEEDS = [0]

def test_rl(self, env, policy, state, train_phase, sess):
    done = False
    s = env.reset()
    total_reward = 0.0
    steps = 0
    while not done:
        env.render()
        a = k.dot(s[0:3])
        s_, r, done, info = env.step(a)
        s = s_
        total_reward += r
        steps += 1


if __name__ == '__main__':
    env = gym.make('TruckBackerUpper-v0')
    k = np. rray([-3.7896, 12.8011, -1.0])
    checkpoint_path = "./models/" + LABEL + "seed_" + str(
                      SEEDS[seed_idx]) + "/my_ddpg.ckpt"
    with tf.Session(graph=tf.Graph()) as sess:
        saved = tf.train.import_meta_graph(checkpoint_path + '.meta', clear_devices=True)
        saved.restore(sess, checkpoint_path)
        np.random.seed(SEEDS[seed_idx])
        tf.set_random_seed(SEEDS[seed_idx])
        env.seed(SEEDS[seed_idx])
    for seed_idx in range(len(SEEDS)):
        
