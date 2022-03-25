#!/usr/bin/env python

import os, sys
import numpy as np
import time

from gym import spaces
from gym.utils import seeding

from env_script.env_mujoco_util import JacoMujocoEnvUtil
from terminal_log import *


class JacoMujocoEnv(JacoMujocoEnvUtil):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ### ------------  RL SETUP  ------------ ###
        ## Env Steps
        self.current_steps = 0
        self.max_steps = 2500
        self.task = kwargs.get('task', None)
        if self.task in ['reaching','grasping', 'carrying', 'releasing', 'pushing']:
            self.task_max_steps = 500
        elif self.task in ['picking', 'placing']:
            self.task_max_steps = 700
        else:
            self.task_max_steps = 1500
        self.skip_frames = 50  #0.05s per step
        seed = kwargs.get('seed', None)
        self.seed(seed)
        logger_path = "./logger_csv"
        # logger_name = "/SR_"+self.task+"_abalation4_"+str(seed)+".csv"
        logger_name = "/SR_"+self.task+'_'+str(seed)+".csv"
        self.file_logger = open(logger_path+logger_name, 'w')

        ## Observations
        # Not touched, Inner touched, Outer touched, grasped
        touch_index = [3]
        #
        joint_pos_max = [10] * 6
        joint_vel_max = [10] * 6
        # normalized into 1
        end_effector_position_max = [1]*3           
        # normalized into 1
        end_effector_orientation_max = [1]*3    
        # normalized into 0.5
        gripper_angle_max = [1]
        prev_position_action_max = [2]*3            
        prev_orientation_action_max = [2]*3         
        # normalized into 1
        obj_position_max = [1]*3                    
        # normalized into 1
        obj_orientation_max = [1]*3
        obj_dist_max = [3]
        # normalized into 1
        dest_max = [1]*3
        # normalized into 1
        reach_position_max = [1]*3                  
        # normalized into 1
        reach_orientation_max = [1]*3
        # normalized into 1
        dest_ori_max = [1]*3
        if self.ctrl_type == 'torque':
            obs_max = np.hstack([
                touch_index,                            #[0]
                end_effector_position_max,              #[1:4]
                end_effector_orientation_max,           #[4:7]
                gripper_angle_max,                      #[7:8]
                obj_position_max,                       #[8:11]
                obj_orientation_max,                    #[11:14]
                dest_max,                               #[14:17]
                reach_position_max,                     #[17:20]
                reach_orientation_max,                  #[20:23]
                dest_ori_max]).repeat(self.n_robots)    #[24:27]
        elif self.ctrl_type == 'velocity':
            obs_max = np.hstack([
                touch_index,                        #[0]
                joint_pos_max,
                joint_vel_max,
                end_effector_position_max,          #[1:4]
                end_effector_orientation_max,       #[4:7]
                gripper_angle_max,                  #[7:8]
                obj_position_max,                   #[8:11]
                obj_orientation_max,                #[11:14]
                obj_dist_max,
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
        self.gripper_action_space_max = 1
        self.gripper_action_space_min = -1
        if self.task in ['reaching','pushing']:
            pose_act = np.array([self.pose_action_space_max]*6)             # x,y,z,r,p,y
            self.act_max = pose_act
            self.act_min = -pose_act
        elif self.task in ['grasping', 'carrying', 'picking', 'releasing', 'placing', 'pickAndplace', 'bimanipulation']:
            pose_act = np.array([self.pose_action_space_max]*6)             # x,y,z,r,p,y
            gripper_act_max = np.array([self.gripper_action_space_max])     # g
            gripper_act_min = np.array([self.gripper_action_space_min])
            self.act_max = np.hstack([pose_act, gripper_act_max]).repeat(self.n_robots)
            self.act_min = np.hstack([-pose_act, gripper_act_min]).repeat(self.n_robots)

        self.action_space = spaces.Box(self.act_min, self.act_max, dtype=np.float32)
        self.wb = 0
        self.accum_rew = 0
        self.accum_succ = [0] * 25
        self.metadata = None

        ### ------------  LOGGING  ------------ ###
        self.log_save = False


    def reset(self):
        self.current_steps = 0
        self.accum_rew = 0
        return self._reset()

    def get_state_shape(self):
        return self.state_shape
    
    def get_num_observation(self):
        return self.observation_space.shape[0]

    def get_num_action(self):
        return self.action_space.shape[0]

    def get_action_bound(self):
        return self.pose_action_space_max

    def step(self, action, weight=None, subgoal=None, log=True, id=None):
        action = np.clip(action, self.act_min, self.act_max)
        self.take_action(action, weight, subgoal, id)
        for _ in range(self.skip_frames):
            self._step_simulation()

        obs = self.make_observation()
        reward_val = self._get_reward()
        done, additional_reward, self.wb, succ = self.terminal_inspection()
        total_reward = reward_val + additional_reward
        # if self.current_steps % 10 == 0:
        #     logging(obs, self.prev_obs, action, self.wb, total_reward) if log else None
        self.prev_obs = obs
        self.accum_rew += total_reward

        if done:
            self.accum_succ.pop()
            self.accum_succ.insert(0,succ)
            sr = "{0:3.3f}".format(sum(self.accum_succ)/25 * 100)
            print('sr: ',sr)
            self.file_logger.writelines(sr+"\n")
            self.file_logger.flush()

        
        return obs, total_reward, done, {0: 0}
    
    def get_wb(self):
        return self.wb

    def terminal_inspection(self):
        self.current_steps += 1
        if self.current_steps < self.task_max_steps:
            return self._get_terminal_inspection()
        else:
            print("\033[91m \nTime Out \033[0m")
            return True, -10, 0, 0

    def make_observation(self):
        obs = self._get_observation()
        assert self.state_shape == obs.shape[0], \
            "State shape from state generator ({0}) and observations ({1}) differs. Possible test code error. You should fix it.".format(
                self.state_shape, obs.shape[0])
        return obs

    def take_action(self, a, weight=None, subgoal=None, id=None):
        self._take_action(a, weight, subgoal, id)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    def close(self):
        pass
