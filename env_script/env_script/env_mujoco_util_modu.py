#!/usr/bin/env python

import os
import sys
import time
from math import pi
from random import sample, randint, uniform

import numpy as np
from numpy.random import uniform as uniform_np

if __name__ != "__main__":
    from env_script.assets.mujoco_config_modu import MujocoConfig
    from env_script.mujoco_modu import Mujoco


class JacoMujocoEnvUtil:
    def __init__(self, controller=True, **kwargs):
        ### ------------  MODEL CONFIGURATION  ------------ ###
        self.jaco = MujocoConfig('jaco2')
        self.interface = Mujoco(self.jaco, dt=0.005)
        self.interface.connect()
        self.ctrl_type = self.jaco.ctrl_type
        self.base_position = self._get_property('link1','position')

        ### ------------  REWARD  ------------ ###
        self.goal = self._sample_goal()
        self.num_episodes = 0


    def _step_simulation(self):
        fb = self.interface.get_feedback()
        self.current_jointstate_1 = fb['q']
        if self.ctrl_type == "torque":
            self.interface.send_signal(np.hstack([self.target_signal, [0, 0, 0]]))
        elif self.ctrl_type == "velocity":
            self.interface.send_signal(np.hstack([fb['q'], self.target_signal, [0, 0, 0]]))
        elif self.ctrl_type == "position":
            self.interface.send_signal(np.hstack([fb['q'] + self.target_signal, fb['dq'], [0, 0, 0]]))
    
    def _reset(self, target_angle=None):
        self.num_episodes = 0
        self.gripper_angle_1 = 0.35
        self.gripper_angle_2 = 0.35
        if target_angle == None:
            random_init_angle = [uniform_np(-pi/2, pi/2), 3.75, uniform_np(
                1.5, 2.5), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3)]
        else:
            random_init_angle = target_angle
        self.interface.set_joint_state(random_init_angle, [0]*6)
        for _ in range(3):
            fb = self.interface.get_feedback()
            self.interface.send_forces([0]*9)
        self.current_jointstate = fb['q']
        self.goal = self._sample_goal()
        obs = self._get_observation()
        dist_diff = np.linalg.norm(obs[0][:3] - self.goal[0])
        self.ref_reward = (3 - dist_diff * 1.3)
        return obs[0]

    def _get_observation(self):
        self.gripper_pose = self._get_property('hand','pose')
        observation = self.gripper_pose- np.hstack([self.goal,[0,0,0]])
        return np.array(observation)

    def _get_reward(self):
        dist_diff = np.linalg.norm(self.gripper_pose[:3] - self.goal)
        if dist_diff > 0.5:
            return 0
        else:
            reward = ((3 - dist_diff * 1.3)) * 0.1
        return reward

    def _sample_goal(self):
        target_pos = [uniform(0.25, 0.35) * sample([-1, 1], 1)[0]
                    for i in range(2)] + [uniform(0.1, 0.4)]
        return np.array(target_pos)

    def _get_terminal_inspection(self):
        self.num_episodes += 1
        dist_diff = np.linalg.norm(self.gripper_pose[:3] - self.goal)
        wb = np.linalg.norm(self._get_property('hand','position') - self.base_position)
        if pi - 0.1 < self.interface.get_feedback()['q'][2] < pi + 0.1:
            print("\033[91m \nUn wanted joint angle - possible singular state \033[0m")
            return True, -5, wb
        else:
            if wb > 0.9:
                print("\033[91m \nWorkspace out of bound \033[0m")
                return True, -5, wb
            else:
                if dist_diff < 0.15:  # TODO: Shape terminal inspection
                    print("\033[92m Target Reached \033[0m")
                    return True, 200 - (self.num_episodes*0.1), wb
                else:
                    return False, 0, wb

    def _take_action(self, a):
        _ = self._get_observation()
        # NOTE
        # If Position: Joint Angle Increments (rad)
        # If Velocity: Joint Velocity (rad/s)
        # If Torque: Joint Torque (Nm)
        self.target_signal = a

    def _get_property(self, subject, prop):
        if prop == 'position':
            out = self.interface.get_xyz(subject)
            return np.copy(out)
        elif prop == 'orientation':
            out = self.interface.get_orientation(subject)
            return np.copy(out)
        elif prop == 'pose':
            pos = self.interface.get_xyz(subject)
            ori = self.interface.get_orientation(subject)
            out = np.append(pos,ori)
            return np.copy(out)


if __name__ == "__main__":
    from assets.mujoco_config_modu import MujocoConfig
    from mujoco_modu import Mujoco

    pos = True
    vel = not pos
    if pos:
        jaco = MujocoConfig('jaco2_position')
    elif vel:
        jaco = MujocoConfig('jaco2_velocity')
    #jaco = MujocoConfig('jaco2')
    interface = Mujoco(jaco, dt=0.005)
    interface.connect()
    interface.set_joint_state([1, 2, 1.5, 1.5, 1.5, 1.5], [0, 0, 0, 0, 0, 0])
    fb = interface.get_feedback()
    if vel:
        inc = .1
        fb['dq'] += np.array([inc] * 6)
        print(fb['dq'])
        mod = True
        fb_new = np.array([0]*6)
        while True:
            fb_pos = interface.get_feedback()
            interface.send_signal(np.hstack([fb_pos['q'], fb['dq'], [0, 0, 0]]))
            if mod == False:    
                fb_new += np.array(interface.get_feedback()['dq'])
                print(fb_new/2)
            else:
                fb_new = np.array(interface.get_feedback()['dq'])
            mod = not mod
    elif pos:
        inc = 0.1
        fb['q'] += np.array([inc] * 6)
        print(fb['q'])
        iter = 0
        while True:
            iter += 1
            interface.send_signal(np.hstack([fb['q'], fb['dq'], [0, 0, 0]]))
            if np.linalg.norm(fb['q'] - interface.get_feedback()['q']) < 0.1:
                print(iter * 0.005)
                break
                #print(interface.get_feedback()['q'])


