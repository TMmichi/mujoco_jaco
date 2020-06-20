#!/usr/bin/env python

import os
import sys
import time
from math import pi
from random import sample, randint, uniform

import numpy as np
from numpy.random import uniform as uniform_np
from scipy.spatial.transform import Rotation as R

#from abr_control.arms.mujoco_config_multi import MujocoConfig
from abr_control.arms.mujoco_config_multi_mod import MujocoConfig
from abr_control.interfaces.mujoco_dual import Mujoco
from abr_control.controllers import OSC


class JacoMujocoEnvUtil:
    def __init__(self, **kwargs):

        self.n_robots = 2
        if self.n_robots == 2:
            xml_name = 'jaco2_dual_include'
        elif self.n_robots == 3:
            xml_name = 'jaco2_tri'
        self.jaco = MujocoConfig(xml_name,n_robots=self.n_robots)
        self.interface = Mujoco(self.jaco, dt=0.005)
        self.interface.connect()
        self.ctr = OSC(self.jaco, kp=100, kv=9, vmax=[0.2,0.5236], ctrlr_dof=[
                       True, True, True, False, False, False])
        self.target_pos = self._reset()[:6]
        self.base_position = self.interface.get_xyz('link1')

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
        u = self.ctr.generate(
            q=fb['q'],
            dq=fb['dq'],
            target=self.target_pos
        )
        self.interface.send_forces(np.hstack([u, [0, 0, 0]]))
    
    def _reset(self, target_angle=None):
        self.num_episodes = 0
        self.gripper_angle_1 = 0.35
        self.gripper_angle_2 = 0.35
        if target_angle == None:
            #random_init_angle = [uniform_np(-pi/2, pi/2), 3.75, uniform_np(
            #    1.5, 2.5), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3)]
            random_init_angle = [1,1,1,1,1,1]
            random_init_angle *= self.n_robots
            random_init_angle[7] = random_init_angle[1] + pi/4
            random_init_angle[6] = random_init_angle[0] + pi/4
        else:
            random_init_angle = target_angle
        self.interface.set_joint_state(random_init_angle, [0]*6*self.n_robots)
        for _ in range(3):
            fb = self.interface.get_feedback()
            u = self.ctr.generate(
                q=fb['q'],
                dq=fb['dq'],
                target=np.hstack([[-0.25, 0, -0.15, 0, 0, 0],[0.25, 0, -0.15, 0, 0, 0]])
            )
            self.interface.send_forces([0]*9*self.n_robots)
        self.current_jointstate = fb['q']
        self.goal = self._sample_goal()
        obs = self._get_observation()
        dist_diff = np.linalg.norm(obs[:3] - self.goal)
        self.ref_reward = (3 - dist_diff * 1.3)
        return obs

    def _get_observation(self):
        test = True  # TODO: Remove test
        if test:
            self.gripper_pose = []
            observation = []
            for i in range(self.n_robots):
                prefix = "_"+str(i+1)
                position = self.interface.get_xyz('hand'+prefix)
                orientation_quat = self.interface.get_orientation('hand'+prefix)
                r = R.from_quat(orientation_quat)
                orientation_euler = r.as_euler('xyz')
                pose = np.append(position, orientation_euler)
                self.gripper_pose.append(pose)
                obs_ind = np.append(position-self.goal[i],orientation_euler)
                observation.append(obs_ind)
            self.gripper_pose = np.array(self.gripper_pose)
            observation = np.array(observation)
        else:
            data_from_callback = []
            observation = self.state_gen.generate(data_from_callback)
        #time.sleep(10)
        return observation

    def _get_reward(self):
        # TODO: Reward from IRL
        if self.reward_method == "l2":
            dist_diff = np.linalg.norm(self.gripper_pose[:3] - self.goal)
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
        for i in range(self.n_robots):
            target_pos = [uniform(0.2, 0.5) * sample([-1, 1], 1)[0]
                        for i in range(2)] + [uniform(0.1, 0.4)]
            goal.append(target_pos)
        # TODO: Target pose -> make object in Mujoco
        return np.array(goal)

    def _get_terminal_inspection(self):
        self.num_episodes += 1
        dist_diff = np.linalg.norm(self.gripper_pose[:3] - self.goal)
        wb = np.linalg.norm(self.interface.get_xyz('hand') - self.base_position)
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
        self.target_pos = self.gripper_pose + np.hstack([a[:3]/100,a[3:]/20])

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
