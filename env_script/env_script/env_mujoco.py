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
        ## Env Steps
        self.current_steps = 0
        try:
            self.max_steps = kwargs['steps_per_batch'] * \
                kwargs['batches_per_episodes']
        except Exception:
            print("Using default max_steps: 500")
            self.max_steps = 500

        ## Observations
        try:
            self.state_shape = kwargs['stateGen'].get_state_shape()
        except Exception:
            self.state_shape = [6]
        self.obs_max = 2
        obs = np.array([self.obs_max]*self.state_shape[0])
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)
        self.prev_obs = [0,0,0,0,0,0]

        ## Actions
        # unit action from the policy = 1 (cm) in real world
        # max distance diff/s = 1.5 / 0.1 = 15 (cm)
        self.pose_action_space_max = 1.5
        self.gripper_action_space_max = 10  # Open
        self.gripper_action_space_min = 0   # Closed
        pose_act = np.array([self.pose_action_space_max]*6)             # x,y,z,r,p,y
        gripper_act_max = np.array([self.gripper_action_space_max]*2)   # g1, g2
        gripper_act_min = np.array([self.gripper_action_space_min]*2)
        self.act_max = np.hstack([pose_act, gripper_act_max])
        self.act_min = np.hstack([-pose_act, gripper_act_min])
        self.action_space = spaces.Box(self.act_min, self.act_max, dtype=np.float32)
        self.seed()

        ### ------------  LOGGING  ------------ ###
        self.log_save = False
        if kwargs['train_log']:
            log_dir = kwargs['log_dir']
            self.log_save = True
            self.joint_angle_log = open(log_dir+"/training_log.txt", 'w')


    def reset(self):
        self.current_steps = 0
        return self._reset()

    def get_state_shape(self):
        return self.state_shape
    
    def get_num_observation(self):
        return self.observation_space.shape[0]

    def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.pose_action_space_max

    def step(self, action, log=True):
        num_step_pass = 100  #was 20                                      # 0.2s per step
        action = np.clip(action, self.act_min, self.act_max)
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
        if self.interface.sim.data.get_sensor('boxx') >14.715:
            print(self.interface.sim.data.get_sensor('boxx'))
            quit()
        print(self.interface.sim.data.sensordata.dtype)
        print(self.interface.sim.data.get_sensor("boxx"))
        

        print(self.interface.sim.data.get_sensor("td_touch_0"))
        print(self.interface.sim.data.get_sensor("td_touch_1"))
        print(self.interface.sim.data.get_sensor("td_touch_2"))
        print(self.interface.sim.data.get_sensor("td_touch_3"))
        print(self.interface.sim.data.get_sensor("td_touch_4"))
        print(self.interface.sim.data.get_sensor("td2_touch_0"))
        print(self.interface.sim.data.get_sensor("td2_touch_1"))
        print(self.interface.sim.data.get_sensor("td2_touch_2"))
        print(self.interface.sim.data.get_sensor("td2_touch_3"))
        print(self.interface.sim.data.get_sensor("td2_touch_4"))
        print(self.interface.sim.data.get_sensor("tp_touch_0"))
        print(self.interface.sim.data.get_sensor("tp_touch_1"))
        print(self.interface.sim.data.get_sensor("tp_touch_2"))
        print(self.interface.sim.data.get_sensor("tp_touch_3"))
        print(self.interface.sim.data.get_sensor("tp_touch_4"))
        print(self.interface.sim.data.get_sensor("tp2_touch_0"))
        print(self.interface.sim.data.get_sensor("tp2_touch_1"))
        print(self.interface.sim.data.get_sensor("tp2_touch_2"))
        print(self.interface.sim.data.get_sensor("tp2_touch_3"))
        print(self.interface.sim.data.get_sensor("tp2_touch_4"))

        self.interpolation("tp")
        self.interpolation("tp2")
        self.interpolation("td")
        self.interpolation("td2")
        self.interpolation("ip")
        self.interpolation("ip2")
        self.interpolation("id")
        self.interpolation("id2")
        self.interpolation("pp")
        self.interpolation("pp2")
        self.interpolation("pd")
        self.interpolation("pd2")

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
        if self.log_save:
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
        self.obs = self._get_observation()[0]   # Gripper pose of the first jaco2
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
