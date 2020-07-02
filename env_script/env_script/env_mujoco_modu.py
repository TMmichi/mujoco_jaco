#!/usr/bin/env python

import os
import numpy as np
import time

from gym import spaces
from gym.utils import seeding

from env_script.env_mujoco_util import JacoMujocoEnvUtil


class JacoMujocoEnv(JacoMujocoEnvUtil):
    def __init__(self, **kwargs):
        #self.debug = kwargs['debug']
        super().__init__(**kwargs)

        ### ------------  RL SETUP  ------------ ###
        self.current_steps = 0
        self.num_envs = 1
        try:
            self.max_steps = kwargs['steps_per_batch'] * \
                kwargs['batches_per_episodes']
        except Exception:
            print("Using default max_steps: 500")
            self.max_steps = 500

        self.state_shape = [6]
        self.obs_max = 2
        obs = np.array([self.obs_max]*self.state_shape[0])
        self.observation_space = spaces.Box(-obs, obs)
        self.prev_obs = [0,0,0,0,0,0]
        self.action_space_max = 0.7 * 2
        act = np.array([self.action_space_max]*6)
        self.action_space = spaces.Box(-act, act)
        self.seed()

        ### ------------  LOGGING  ------------ ###
        log_dir = "/home/modulab/mujoco_jaco/logs"
        os.makedirs(log_dir, exist_ok=True)
        self.joint_angle_log = open(log_dir+"/log.txt", 'w')


    def reset(self):
        self.current_steps = 0
        return self._reset()

    def get_state_shape(self):
        return self.state_shape

    def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.action_space_max

    def step(self, action, log=True):
        num_step_pass = 40
        # actions = np.clip(actions,-self.action _space_max, self.action_space_max)
        action = np.clip(action, -self.action_space_max, self.action_space_max)
        self.take_action(action)
        for _ in range(num_step_pass):
            self._step_simulation()

        self.make_observation()
        reward_val = self._get_reward()
        done, additional_reward, wb = self.terminal_inspection()
        total_reward = reward_val + additional_reward

        self.logging(self.obs, self.prev_obs, action, wb, total_reward) if log else None
        self.prev_obs = self.obs

        return self.obs, total_reward, done, {0: 0}

    def logging(self, obs, prev_obs, action, wb, reward):
        write_str = "Act:"
        for i in range(len(action)):
            write_str += "\t{0:2.3f}".format(action[i])
        write_str += "\t| Obs:" 
        for i in range(len(obs)):
            write_str += self._colored_string(obs[i],prev_obs[i],action[i])
        write_str += "\t| wb = {0:2.3f} | \033[92mReward:\t{1:1.5f}\033[0m".format(wb,reward)
        print(write_str, end='\r')
        self.joint_angle_log.writelines(write_str+"\n")

    def _colored_string(self, obs_val, prev_obs_val, action):
        if int(np.sign(obs_val-prev_obs_val)) == int(np.sign(action)):
            return "\t\033[92m{0:2.3f}\033[0m".format(obs_val)
        else:
            return "\t\033[91m{0:2.3f}\033[0m".format(obs_val)

    def terminal_inspection(self):
        self.current_steps += 1
        if self.current_steps < self.max_steps:
            return self._get_terminal_inspection()
        else:
            return True, 0, 0

    def make_observation(self):
        self.obs = self._get_observation()[0]
        assert self.state_shape[0] == self.obs.shape[0], \
            "State shape from state generator ({0}) and observations ({1}) differs. Possible test code error. You should fix it.".format(
                self.state_shape[0], self.obs.shape[0])

    def take_action(self, a):
        self._take_action(a)
        return 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
