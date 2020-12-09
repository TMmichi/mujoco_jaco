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
        self.num_step_pass = 30  #0.15s per step

        ## Observations
        end_effector_pose_max = [2]*6   #[0:6]
        gripper_angle_max = [10]*2      #[6:8]
        obj_max = [2]*3                 #[8:11]
        dest_max = [2]*3                #[11:14]
        reach_max = [np.pi]*6           #[14:17]
        obs_max = np.hstack([end_effector_pose_max, gripper_angle_max, obj_max, dest_max, reach_max])
        obs_min = -obs_max
        obs_min[6:8] = 0
        self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)
        try:
            self.state_shape = kwargs['stateGen'].get_state_shape()
        except Exception:
            self.state_shape = self.observation_space.shape[0]
        self.prev_obs = [0]*self.state_shape

        ## Actions
        # unit action from the policy = 1 (cm) in real world
        # max distance diff/s = 1cm / 0.15s
        # max angular diff/s = 0.05rad / 0.15s
        self.pose_action_space_max = 1
        # 10: max open, 0: max closed
        # incremental angle
        # takes 2 seconds to fully stretch/grasp the gripper
        self.gripper_action_space_max = 0.5
        self.gripper_action_space_min = -0.5
        if kwargs.get('task', None) == 'reaching':
            pose_act = np.array([self.pose_action_space_max]*6)             # x,y,z,r,p,y
            self.act_max = pose_act
            self.act_min = -pose_act
        elif kwargs.get('task', 'grasping') == 'grasping':
            pose_act = np.array([self.pose_action_space_max]*6)             # x,y,z,r,p,y
            gripper_act_max = np.array([self.gripper_action_space_max]*2)   # g1, g2
            gripper_act_min = np.array([self.gripper_action_space_min]*2)
            self.act_max = np.hstack([pose_act, gripper_act_max])
            self.act_min = np.hstack([-pose_act, gripper_act_min])
        elif kwargs.get('task', None) == 'carrying':
            pose_act = np.array([self.pose_action_space_max]*6)             # x,y,z,r,p,y
            gripper_act_max = np.array([self.gripper_action_space_max]*2)   # g1, g2
            gripper_act_min = np.array([self.gripper_action_space_min]*2)
            self.act_max = np.hstack([pose_act, gripper_act_max])
            self.act_min = np.hstack([-pose_act, gripper_act_min])
        self.action_space = spaces.Box(self.act_min, self.act_max, dtype=np.float32)
        self.wb = 0
        self.seed()

        ### ------------  LOGGING  ------------ ###
        self.log_save = False
        # if kwargs['train_log']:
        #     log_dir = kwargs['log_dir']
        #     self.log_save = True
        #     self.joint_angle_log = open(log_dir+"/training_log.txt", 'w')


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
        action = np.clip(action, self.act_min, self.act_max)
        self.take_action(action)
        for _ in range(self.num_step_pass):
            self._step_simulation()

        self.make_observation()
        reward_val = self._get_reward()
        done, additional_reward, self.wb = self.terminal_inspection()
        total_reward = reward_val + additional_reward
        if self.current_steps % 10 == 0:
            self.logging(self.obs, self.prev_obs, action, self.wb, total_reward) if log else None
        self.prev_obs = self.obs
        if np.any(np.isnan(self.obs)) or np.any(np.isnan(total_reward)):
            print("WARNING: NAN in obs, resetting.")
            self.obs = self.reset()
            total_reward = 0
            done = True

        return self.obs, total_reward, done, {0: 0}
    
    def get_wb(self):
        return self.wb

    def logging(self, obs, prev_obs, action, wb, reward):
        write_str = "Act:"
        for i in range(len(action)):
            write_str += " {0: 2.3f}".format(action[i])
        write_str += "\t| Obs:" 
        write_log = write_str
        for i in range(6):
            write_str += self._colored_string(obs[i],prev_obs[i],action[i])
            write_log += ", {0: 2.3f}".format(obs[i])
        write_str += "\t| wb = {0: 2.3f} | \033[92mReward:\t{1:1.5f}\033[0m".format(wb,reward)
        write_log += "\t| wb = {0: 2.3f} | Reward:\t{1:1.5f}".format(wb,reward)
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
        assert self.state_shape == self.obs.shape[0], \
            "State shape from state generator ({0}) and observations ({1}) differs. Possible test code error. You should fix it.".format(
                self.state_shape, self.obs.shape[0])

    def take_action(self, a):
        self._take_action(a)
        return 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    def close(self):
        pass


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
