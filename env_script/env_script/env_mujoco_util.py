#!/usr/bin/env python

import os
import sys
import time
from math import pi
from random import sample, randint, uniform

import numpy as np
from numpy.random import uniform as uniform_np

#from abr_control.arms.mujoco_config_multi import MujocoConfig
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_control.controllers import OSC
from abr_control.utils import transformations


class JacoMujocoEnvUtil:
    def __init__(self, **kwargs):

        self.n_robots = 1
        if self.n_robots == 1:
            xml_name = 'jaco2'
        elif self.n_robots == 2:
            xml_name = 'jaco2_dual_include'
        elif self.n_robots == 3:
            xml_name = 'jaco2_tri'
        else:
            raise NotImplementedError("[ERROR] xml_file of the given number of robots doesn't exist")
        self.jaco = MujocoConfig(xml_name,n_robots=self.n_robots)
        self.interface = Mujoco(self.jaco, dt=0.005)
        self.interface.connect()

        self.controller = True
        self.control_type = self.jaco.ctrl_type
        if self.controller and self.control_type == "torque":
            self.ctr = OSC(self.jaco, kp=50, ko=180, kv=20, vmax=[0.2,0.5236], ctrlr_dof=[
                       True, True, True, True, True, True])

        self.target_pos = self._reset()
        self.base_position = self._get_property('link1','position')

        ### ------------  STATE GENERATION  ------------ ###
        try:
            self.state_gen = kwargs['stateGen']
        except Exception:
            self.state_gen = None
        self.image_buffersize = 5
        self.image_buff = []
        self.pressure_buffersize = 100
        self.pressure_state = []

        self.depth_trigger = True
        self.pressure_trigger = True
        self.data_buff = []
        self.data_buff_temp = [0, 0, 0]

    ### ------------  REWARD  ------------ ###
        self.goal = self._sample_goal()
        self.num_episodes = 0
        try:
            self.reward_method = kwargs['reward_method']
            self.reward_module = kwargs['reward_module']
        except Exception:
            self.reward_method = None
            self.reward_module = None

    def _step_simulation(self):
        fb = self.interface.get_feedback()
        self.current_jointstate_1 = fb['q']
        if self.controller:
            u = self._controller_generate(fb)
            self.interface.send_forces(np.hstack([u, [0, 0, 0]]))
        else:
            if self.control_type == "torque":
                self.interface.send_signal(np.hstack([self.target_torque, [0, 0, 0]]))
            elif self.control_type == "velocity":
                self.interface.send_signal(np.hstack([self.target_velocity, [0, 0, 0]]))
            elif self.control_type == "position":
                self.interface.send_signal(np.hstack([self.target_position, [0, 0, 0]]))
    
    def _controller_generate(self, fb):
        return self.ctr.generate(
            q=fb['q'],
            dq=fb['dq'],
            target=self.target_pos
        )
    
    def _reset(self, target_angle=None):
        self.num_episodes = 0
        self.gripper_angle_1 = 0.35
        self.gripper_angle_2 = 0.35
        if target_angle == None:
            random_init_angle = [uniform_np(-pi/2, pi/2), 3.75, uniform_np(
                1.5, 2.5), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3)]
            #random_init_angle = [1,1,1,1,1,1]
            random_init_angle *= self.n_robots
            if self.n_robots > 1:
                random_init_angle[7] = random_init_angle[1] + pi/4
                random_init_angle[6] = random_init_angle[0] + pi/4
        else:
            random_init_angle = target_angle
        self.interface.set_joint_state(random_init_angle, [0]*6*self.n_robots)
        for _ in range(3):
            fb = self.interface.get_feedback()
            _ = self._controller_generate(fb)
            self.interface.send_forces([0]*9*self.n_robots)
        self.current_jointstate = fb['q']
        self.goal = self._sample_goal()
        obs = self._get_observation()
        #TODO: additional term for dist_diff
        dist_diff = np.linalg.norm(obs[0][:3] - self.goal[0])
        self.ref_reward = (3 - dist_diff * 1.3)
        return obs[0]

    def _get_observation(self):
        test = True  # TODO: Remove test
        if test:
            self.gripper_pose = self._get_property('hand','pose')
            observation = []
            for i in range(self.n_robots):
                observation.append(self.gripper_pose[i] - np.hstack([self.goal[i],[0,0,0]]))
        else:
            data_from_callback = []
            observation = self.state_gen.generate(data_from_callback)
        return np.array(observation)

    def _get_reward(self):
        # TODO: Reward from IRL
        if self.reward_method == "l2":
            dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.goal[0])
            if dist_diff > 0.5:
                return 0
            else:
                reward = ((3 - dist_diff * 1.3)) * 0.1  # TODO: Shape reward
            return reward
        elif self.reward_method == "":
            return self.reward_module(self.gripper_pose, self.goal)
        else:
            print("\033[31mConstant Reward. SHOULD BE FIXED\033[0m")
            return 30

    def _sample_goal(self):
        goal = []
        for _ in range(self.n_robots):
            target_pos = [uniform(0.25, 0.35) * sample([-1, 1], 1)[0]
                        for i in range(2)] + [uniform(0.1, 0.4)]
            goal.append(target_pos)
        # TODO: Target pose -> make object in Mujoco
        return np.array(goal)

    def _get_terminal_inspection(self):
        self.num_episodes += 1
        dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.goal[0])
        wb = np.linalg.norm(self._get_property('hand','position')[0] - self.base_position[0])
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
        self.target_pos = self.gripper_pose[0] + np.hstack([a[:3]/100,a[3:]/20])

    def _get_property(self, subject, prop):
        out = []
        for i in range(self.n_robots):
            if prop == 'position':
                if self.n_robots == 1:
                    out.append(self.interface.get_xyz(subject))
                else:
                    prefix = "_"+str(i+1)
                    out.append(self.interface.get_xyz(subject+prefix))
                return np.copy(out)
            elif prop == 'orientation':
                if self.n_robots == 1:
                    orientation_quat = self.interface.get_orientation(subject)
                    out.append(transformations.euler_from_quaternion(orientation_quat,'rxyz'))
                else:
                    prefix = "_"+str(i+1)
                    orientation_quat = self.interface.get_orientation(subject+prefix)
                    out.append(transformations.euler_from_quaternion(orientation_quat,'rxyz'))
                return np.copy(out)
            elif prop == 'pose':
                if self.n_robots == 1:
                    pos = self.interface.get_xyz(subject)
                    orientation_quat = self.interface.get_orientation(subject)
                    ori = transformations.euler_from_quaternion(orientation_quat,'rxyz')
                    pose = np.append(pos,ori)
                    out.append(pose)
                else:
                    prefix = "_"+str(i+1)
                    pos = self.interface.get_xyz(subject+prefix)
                    orientation_quat = self.interface.get_orientation(subject+prefix)
                    ori = transformations.euler_from_quaternion(orientation_quat,'rxyz')
                    pose = np.append(pos,ori)
                    out.append(pose)
                return np.copy(out)

    def _get_depth(self):
        pass

    def _get_pressure(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    jaco = MujocoConfig('jaco2')
    interface = Mujoco(jaco, dt=0.01)
    interface.connect()

    ctr = OSC(jaco, kp=2, ctrlr_dof=[True, True, True, False, False, False])
    interface.set_joint_state([1, 2, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
    for _ in range(2):
        fb = interface.get_feedback()
        u = ctr.generate(
            q=fb['q'],
            dq=fb['dq'],
            target=np.hstack([0, 0, -0.15, 0, 0, 0])
        )
        interface.send_forces([0, 0, 0, 0, 0, 0, 0, 0, 0])
    print(interface.get_xyz('hand'))
    target_pos = interface.get_xyz('hand')
    target_or = np.array([0, 0, 0], dtype=np.float16)
    for _ in range(10):
        while True:
            fb = interface.get_feedback()
            u = ctr.generate(
                q=fb['q'],
                dq=fb['dq'],
                target=np.hstack([target_pos, target_or])
            )
            a = interface.get_xyz('hand')
            b = interface.get_orientation('hand')
            # print(a)1
            interface.send_forces(np.hstack([u, [0, 0, 0]]))
            if np.linalg.norm(a[:3] - target_pos[:3]) < 0.01:
                print("Reached")
                break
        target_pos += np.array([0.01, 0.01, 0.01])
        target_or += np.array([0.1, 0.1, 0.1])
