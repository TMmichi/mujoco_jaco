#!/usr/bin/env python

import os, sys
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

        try:
            self.state_shape = kwargs['stateGen'].get_state_shape()
        except Exception:
            self.state_shape = [6]
        self.obs_max = 4
        obs = np.array([self.obs_max]*self.state_shape[0])
        self.observation_space = spaces.Box(-obs, obs)
        self.prev_obs = [0,0,0,0,0,0]
        # 0.007 (m) -> multiplied by factor of 2, will later be divided into 2 @ step
        self.action_space_max = 0.7 * 2
        # unit action (1) from the policy = 0.5 (cm) in real world
        # x,y,z,r,p,y, finger {1,2}, finger 3
        act = np.array([self.action_space_max]*6)
        self.action_space = spaces.Box(-act, act)  # Action space: [-1.4, 1.4]
        self.seed()

        ### ------------  LOGGING  ------------ ###
        if kwargs['train']:
            log_dir = kwargs['log_dir']
            self.joint_angle_log = open(log_dir+"/training_log.txt", 'w')


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
        # TODO: Determine how many time steps should be proceed when called
        # moveit trajectory planning (0.15) + target angle following (0.3 - 0.15?)
        # -> In real world, full timesteps are used for conducting action (No need for finding IK solution)
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
        write_log = write_str
        for i in range(len(obs)):
            write_str += self._colored_string(obs[i],prev_obs[i],action[i])
            write_log += "\t{0:2.3f}".format(obs[i])
        write_str += "\t| wb = {0:2.3f} | \033[92mReward:\t{1:1.5f}\033[0m".format(wb,reward)
        write_log += "\t| wb = {0:2.3f} | Reward:\t{1:1.5f}".format(wb,reward)
        print(write_str, end='\r')
        self.joint_angle_log.writelines(write_log+"\n")

    def _colored_string(self, obs_val, prev_obs_val, action):
        if int(np.sign(obs_val-prev_obs_val)) == int(np.sign(action)):
            return "\t\033[92m{0:2.3f}\033[0m".format(obs_val)
        else:
            return "\t\033[91m{0:2.3f}\033[0m".format(obs_val)

    def terminal_inspection(self):
        # TODO: terminal state definition
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


if __name__ == "__main__":
    env_test_class = JacoMujocoEnv()
    env_test_class.make_observation()
    obs = env_test_class.gripper_pose
    print("First: \t\t", obs[:6])
    itera = True
    if itera:
        for i in range(100):
            iter = 0
            if i % 4 == 0:
                target_pos = env_test_class.obs[:6] + \
                    np.array([0.02, 0.01, 0.01, 0.1, 0.1, 0.1])
                env_test_class.take_action(np.array([2, 1, 1, 2, 2, 2]))
            elif i % 4 == 1:
                target_pos = env_test_class.obs[:6] + \
                    np.array([-0.02, -0.01, -0.01, -0.1, -0.1, -0.1])
                env_test_class.take_action(np.array([-2, -1, -1, -2, -2, -2]))
            elif i % 4 == 2:
                target_pos = env_test_class.obs[:6] + \
                    np.array([0.02, -0.01, -0.01, 0.1, -0.1, -0.1])
                env_test_class.take_action(np.array([2, -1, -1, 2, -2, -2]))
            elif i % 4 == 3:
                target_pos = env_test_class.obs[:6] + \
                    np.array([-0.02, 0.01, 0.01, -0.1, 0.1, 0.1])
                env_test_class.take_action(np.array([-2, 1, 1, -2, 2, 2]))
            while True:
                env_test_class._step_simulation()
                env_test_class.make_observation()
                pos = env_test_class.obs[:6]
                if np.linalg.norm(pos[:3]-target_pos[:3]) < 0.003 and abs(pos[3]-target_pos[3]) < 0.01 and abs(pos[4]-target_pos[4]) < 0.01 and abs(pos[5]-target_pos[5] < 0.01):
                    print("Reached Pose:\t", pos)
                    print("Reached")
                    break
                if iter % 100 == 0:
                    print("Current Pose:\t", pos)
                    print("Target Pose:\t", target_pos)
                iter += 1
    else:
        iter = 0
        target_pos = np.array([0.2, 0.3, 0.3, 1, 0, 0])
        action_p = (target_pos[:3] - env_test_class.obs[:3])*100
        action_o = (target_pos[3:] - env_test_class.obs[3:])*20
        env_test_class.take_action(np.hstack([action_p, action_o]))
        while True:
            env_test_class._step_simulation()
            env_test_class.make_observation()
            pos = env_test_class.obs[:6]
            # if np.linalg.norm(pos[:3]-target_pos[:3]) < 0.005 and np.linalg.norm(pos[3:]-target_pos[3:]) < 0.01:
            if np.linalg.norm(pos[:3]-target_pos[:3]) < 0.005 and abs(pos[3]-target_pos[3]) < 0.01 and abs(pos[4]-target_pos[4]) < 0.01 and abs(pos[5]-target_pos[5] < 0.01):
                print("Reached Pose:\t", pos)
                print("Reached")
                time.sleep(10)
                break
            if iter % 100 == 0:
                print("Current Pose:\t", pos)
                print("Target Pose:\t", target_pos)
            iter += 1
