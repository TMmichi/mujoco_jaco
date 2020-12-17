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
        self.max_steps = 2500
        if kwargs.get('task', None) in ['reaching','grasping', 'carrying', 'releasing', 'pushing']:
            self.task_max_steps = 500
        elif kwargs.get('task', None) in ['picking', 'placing']:
            self.task_max_steps = 1200
        else:
            self.task_max_steps = 2500
        self.num_step_pass = 80  #0.15s per step

        ## Observations
        # Not touched, Inner touched, Outer touched, grasped
        touch_index = [3]
        # normalized into 1
        end_effector_position_max = [1]*3           
        # normalized into 1
        end_effector_orientation_max = [1]*3    
        # normalized into 0.5
        gripper_angle_max = [0.5]
        prev_position_action_max = [2]*3            
        prev_orientation_action_max = [2]*3         
        # normalized into 1
        obj_position_max = [1]*3                    
        # normalized into 1
        obj_orientation_max = [1]*3
        # normalized into 1
        dest_max = [1]*3
        # normalized into 1
        reach_position_max = [1]*3                  
        # normalized into 1
        reach_orientation_max = [1]*3           
        if kwargs.get('prev_action', False):
            obs_max = np.hstack([
                touch_index,                          #[0]
                end_effector_position_max,          #[0:3]
                end_effector_orientation_max,       #[3:6]
                gripper_angle_max,                  #[6:8]
                prev_position_action_max,           #[8:11]
                prev_orientation_action_max,        #[11:14]
                obj_position_max,                   #[14:17]
                obj_orientation_max,                #[17:20]
                dest_max,                           #[20:23]
                reach_position_max,                 #[23:26]
                reach_orientation_max])             #[26:29]
        else:
            obs_max = np.hstack([
                touch_index,                        #[0]
                end_effector_position_max,          #[1:4]
                end_effector_orientation_max,       #[4:7]
                gripper_angle_max,                  #[7:8]
                obj_position_max,                   #[8:11]
                obj_orientation_max,                #[11:14]
                dest_max,                           #[14:17]
                reach_position_max,                 #[17:20]
                reach_orientation_max])             #[20:23]
        obs_min = -obs_max
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
        elif kwargs.get('task', None) in ['grasping', 'carrying']:
            pose_act = np.array([self.pose_action_space_max]*6)             # x,y,z,r,p,y
            gripper_act_max = np.array([self.gripper_action_space_max])     # g
            gripper_act_min = np.array([self.gripper_action_space_min])
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
        self.prev_action = action
        self.take_action(action)
        for _ in range(self.num_step_pass):
            self._step_simulation()

        obs = self.make_observation()
        reward_val = self._get_reward()
        done, additional_reward, self.wb = self.terminal_inspection()
        total_reward = reward_val + additional_reward
        if self.current_steps % 10 == 0:
            self.logging(obs, self.prev_obs, action, self.wb, total_reward) if log else None
        self.prev_obs = obs
        if np.any(np.isnan(obs)) or np.any(np.isnan(total_reward)):
            print("WARNING: NAN in obs, resetting.")
            self.obs = self.reset()
            total_reward = 0
            done = True

        return obs, total_reward, done, {0: 0}
    
    def get_wb(self):
        return self.wb

    def logging(self, obs, prev_obs, action, wb, reward):
        write_str = "Act:"
        for i in range(len(action)):
            write_str += " {0: 2.3f}".format(action[i])
        write_str += "\t| Obs:" 
        write_log = write_str
        write_str += str(int(obs[0]))
        for i in range(1,8):
            write_str += self._colored_string(obs[i],prev_obs[i],action[i-1])
            write_log += ", {0: 2.3f}".format(obs[i])
        write_str += "\t| wb = {0: 2.3f} | \033[92mReward:\t{1:1.5f}\033[0m".format(wb,reward)
        write_log += "\t| wb = {0: 2.3f} | Reward:\t{1:1.5f}".format(wb,reward)
        print(write_str, end='\r')

    def _colored_string(self, obs_val, prev_obs_val, action):
        if int(np.sign(obs_val-prev_obs_val)) == int(np.sign(action)):
            return "\t\033[92m{0:2.3f}\033[0m".format(obs_val)
        else:
            return "\t\033[91m{0:2.3f}\033[0m".format(obs_val)

    def terminal_inspection(self):
        # TODO: terminal state definition
        self.current_steps += 1
        if self.current_steps < self.task_max_steps:
            return self._get_terminal_inspection()
        else:
            return True, 0, 0

    def make_observation(self):
        obs = self._get_observation()[0]
        assert self.state_shape == obs.shape[0], \
            "State shape from state generator ({0}) and observations ({1}) differs. Possible test code error. You should fix it.".format(
                self.state_shape, obs.shape[0])
        return obs

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
