#!/usr/bin/env python

import os, sys
import numpy as np
import time

from gym import spaces
from gym.utils import seeding

from env_script.env_mujoco_util import JacoMujocoEnvUtil


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
            self.task_max_steps = 1200
        self.skip_frames = 50  #0.05s per step
        self.seed(kwargs.get('seed', None))

        ## Observations
        # Not touched, Inner touched, Outer touched, grasped
        touch_index = [3]
        # EE position, normalized into 1
        end_effector_position_max = [1]*3           
        # EE orientation, normalized into 1
        end_effector_orientation_max = [1]*3    
        # Gripper angle, normalized into 1
        gripper_angle_max = [1]
        # Object position, normalized into 1
        obj_position_max = [1]*3                    
        # Object orientation, normalized into 1
        obj_orientation_max = [1]*3
        # Object desetination position, normalized into 1
        dest_max = [1]*3
        # Reaching target position, normalized into 1
        reach_position_max = [1]*3                  
        # Reaching target orientation, normalized into 1
        reach_orientation_max = [1]*3
        # Object destination orientation, normalized into 1
        dest_ori_max = [1]*3
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
        obs_min = -obs_max
        self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)
        try:
            self.state_shape = kwargs['stateGen'].get_state_shape()
        except Exception:
            self.state_shape = self.observation_space.shape[0]

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

    def step(self, action, weight=None, subgoal=None, id=None):
        action = np.clip(action, self.act_min, self.act_max)
        self.take_action(action, weight, subgoal, id)
        for _ in range(self.skip_frames):
            self._step_simulation()

        obs = self.make_observation()
        reward_val = self._get_reward()
        done, additional_reward, self.wb = self.terminal_inspection()
        total_reward = reward_val + additional_reward
        self.accum_rew += total_reward

        print("Return: ", self.accum_rew) if done else None
            
        return obs, total_reward, done, {0: 0}
    
    def get_wb(self):
        return self.wb

    def terminal_inspection(self):
        self.current_steps += 1
        if self.current_steps < self.task_max_steps:
            return self._get_terminal_inspection()
        else:
            print("\033[91m \nTime Out \033[0m")
            return True, -10, 0

    def make_observation(self):
        obs = self._get_observation()
        assert self.state_shape == obs.shape[0], \
            "State shape from state generator ({0}) and observations ({1}) differs.\
                Possible test code error. Should be fixed.".format(
                self.state_shape, obs.shape[0])
        return obs

    def take_action(self, a, weight=None, subgoal=None, id=None):
        self._take_action(a, weight, subgoal, id)
    
    def close(self):
        return None
