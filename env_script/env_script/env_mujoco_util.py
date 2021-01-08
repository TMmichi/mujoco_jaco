#!/usr/bin/env python

import os
import sys
import time
from pathlib import Path
from math import pi
from random import sample, randint, uniform

import scipy.interpolate

import numpy as np
import cv2
from numpy.random import uniform as uniform_np
from matplotlib import pyplot as plt

if __name__ != "__main__":
    from abr_control.controllers import OSC
    from abr_control.utils import transformations
    from env_script.assets.mujoco_config import MujocoConfig
    from env_script.mujoco import Mujoco


class JacoMujocoEnvUtil:
    def __init__(self, controller=True, **kwargs):
        ### ------------  MODEL CONFIGURATION  ------------ ###
        self.n_robots = kwargs.get('n_robots', 1)
        # robot_file = kwargs.get('robot_file', "jaco2_curtain_torque")
        robot_file = kwargs.get('robot_file', "jaco2_torque")
        if robot_file == None:
            n_robot_postfix = ['', '_dual', '_tri']    
            try:
                xml_name = 'jaco2'+n_robot_postfix[self.n_robots-1]
            except Exception:
                raise NotImplementedError("\n\t\033[91m[ERROR]: xml_file of the given number of robots doesn't exist\033[0m")
        else:
            xml_name = robot_file
        self.jaco = MujocoConfig(xml_name, n_robots=self.n_robots)
        self.interface = Mujoco(self.jaco, dt=0.001, visualize=kwargs.get('visualize', False), create_offscreen_rendercontext=False)
        self.interface.connect()
        self.skip_frames = 50
        self.base_position = self.__get_property('link1', 'position')
        self.gripper_angle_1 = 0.5
        self.gripper_angle_2 = 0.5
        self.gripper_angle_1_array = np.zeros(self.skip_frames)
        self.gripper_angle_2_array = np.zeros(self.skip_frames)
        self.gripper_iter = 0
        self.object_z = 0.1898
        self.action_in = False
        self.ctrl_type = self.jaco.ctrl_type
        print("control type: ", self.ctrl_type)
        self.target_signal = [0,0,0,0,0,0]
        self.task = kwargs.get('task', None)
        self.obs_prev_action = kwargs.get('prev_action', False)
        self.num_episodes = 0

        ### ------------  STATE GENERATION  ------------ ###
        self.package_path = str(Path(__file__).resolve().parent.parent)
        self.state_gen = kwargs.get('stateGen', None)
        self.image_buffersize = 5
        self.image_buff = []
        self.pressure_buffersize = 100
        self.pressure_buff = []
        self.data_buff = []

        ### ------------  CONTROLLER SETUP  ------------ ###
        self.controller = controller
        if self.controller:
            ctrl_dof = [True, True, True, True, True, True]
            # self.ctr = OSC(self.jaco, kp=50, ko=180, kv=20, vmax=[0.2, 0.5236], ctrlr_dof=ctrl_dof)
            # self.ctr = OSC(self.jaco, kp=50, ko=180, kv=20, vmax=[0.3, 0.7854], ctrlr_dof=ctrl_dof)
            self.ctr = OSC(self.jaco, kp=50, ko=180, kv=20, vmax=[0.4, 1.0472], ctrlr_dof=ctrl_dof)
            self.target_pos = self._reset()
        else:
            _ = self._reset()

        ### ------------  REWARD  ------------ ###
        self.reward_method = kwargs.get('reward_method', None)
        self.reward_module = kwargs.get('reward_module', None)
        

    def _step_simulation(self):
        fb = self.interface.get_feedback()
        if self.controller:
            u = self.__controller_generate(fb)
            if self.action_in:
                self.interface.send_forces(
                    np.hstack([u, 
                            [
                                self.gripper_angle_1_array[self.gripper_iter], 
                                self.gripper_angle_1_array[self.gripper_iter],
                                self.gripper_angle_1_array[self.gripper_iter]
                            ]]))
            else:
                self.interface.send_forces(
                    np.hstack([u, [self.gripper_angle_1, self.gripper_angle_1, self.gripper_angle_1]]))
            self.gripper_iter += 1
        else:
            if self.ctrl_type == "torque":
                self.interface.send_signal(np.hstack([self.target_signal, [0, 0, 0]]))
            elif self.ctrl_type == "velocity":
                # print('sent vel: ',self.target_signal)
                self.interface.send_signal(
                    np.hstack([fb['q'], self.target_signal, 
                                [
                                    self.gripper_angle_1_array[self.gripper_iter], 
                                    self.gripper_angle_1_array[self.gripper_iter],
                                    self.gripper_angle_1_array[self.gripper_iter]
                                ]]))
                # print('vel: ',self.interface.sim.data.qvel[self.interface.joint_vel_addrs])
            elif self.ctrl_type == "position":
                self.interface.send_signal(np.hstack([fb['q'] + self.target_signal, fb['dq'], [0, 0, 0]]))

    def __controller_generate(self, fb):
        return self.ctr.generate(
            q=fb['q'],
            dq=fb['dq'],
            target=self.target_pos
        )

    def _reset(self, target_angle=None):
        self.num_episodes = 0
        self.gripper_iter = 0
        self.touch_index = 0
        self.action_in = False
        self.gripper_angle_1 = 0.5
        self.gripper_angle_2 = 0.5
        # self.curr_action = np.zeros((6))
        self.prev_action = np.zeros((6))
        # self.interface.viewer._paused = True
        init_angle = self._create_init_angle()
        self.interface.set_joint_state(init_angle, [0]*6*self.n_robots)
        self.reaching_goal, self.obj_goal, self.dest_goal = self.__sample_goal()
        
        if self.task in ['carrying', 'releasing', 'placing']:
            pos = self.__get_property('EE_obj', 'pose')[0]
            quat = transformations.quaternion_from_euler(pos[3], pos[4], pos[5], axes='rxyz')
            self.interface.set_obj_xyz(pos[:3], quat)
        else:
            quat = [0,0,0,0]
            self.interface.set_obj_xyz(self.obj_goal[0],quat)

        if self.task in ['grasping', 'carrying']:
            self.__get_gripper_pose()
            x,y,z = self.obj_goal[0] - self.gripper_pose[0][:3]
            alpha = -np.arcsin(y / np.sqrt(y**2+z**2)) * np.sign(x)
            beta = np.arccos(x / np.linalg.norm([x,y,z])) * np.sign(x)
            gamma = uniform(-0.1, 0.1)
            reach_goal_ori = np.array([alpha, beta, gamma], dtype=np.float16)
            self.target_pos = np.reshape(np.hstack([self.obj_goal,np.repeat([reach_goal_ori],self.n_robots,axis=0)]),(-1))
            if self.controller:
                while True:
                    self.interface.stop_obj()
                    self._step_simulation()
                    self.__get_gripper_pose()
                    dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.obj_goal[0])
                    grip = transformations.unit_vector(
                        transformations.quaternion_from_euler(
                            self.gripper_pose[0][3], self.gripper_pose[0][4], self.gripper_pose[0][5], axes="rxyz"))
                    tar = transformations.unit_vector(
                    transformations.quaternion_from_euler(
                        self.reaching_goal[0][3], self.reaching_goal[0][4], self.reaching_goal[0][5], axes="rxyz"))
                    angle_diff = grip-tar
                    ang_diff = np.linalg.norm(angle_diff)
                    if dist_diff < 0.2:
                        self.target_pos = np.reshape(np.hstack([[self.gripper_pose[0][:3]],np.repeat([reach_goal_ori],self.n_robots,axis=0)]),(-1))
                        break
                    if ang_diff < np.pi/6:
                        break
                while True:
                    self._step_simulation()
                    self.__get_gripper_pose()
                    self.target_pos = np.reshape(np.hstack([self.obj_goal,np.repeat([reach_goal_ori],self.n_robots,axis=0)]),(-1))
                    dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.obj_goal[0])
                    if dist_diff < 0.15:
                        break
        elif self.task in ['placing', 'releasing']:
            pass
        elif self.task == 'reaching':
            self.interface.set_mocap_xyz("target_reach", self.reaching_goal[0][:3])
            self.interface.set_mocap_orientation("target_reach", transformations.quaternion_from_euler(
                self.reaching_goal[0][3], self.reaching_goal[0][4], self.reaching_goal[0][5], axes="rxyz"))
            self.target_pos = np.reshape(np.hstack([self.reaching_goal]),(-1))
        obs = self._get_observation()
        return obs[0]
    
    def _create_init_angle(self):
        if self.task in ['reaching', 'picking']:
            random_init_angle = [uniform_np(-pi/2, pi/2), 3.75, uniform_np(
                    1.5, 2.5), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3)]
            random_init_angle *= self.n_robots
        elif self.task in ['carrying', 'grasping']:
            angle0 = np.random.choice([uniform_np(3*pi/8,pi/2), uniform_np(pi/2,5*pi/8)])
            # angle0 = np.random.choice([uniform_np(-pi/8,pi/8)])
            random_init_angle = [angle0, 3.85, uniform_np(
                    1, 1.1), uniform_np(2, 2.1), uniform_np(0.8, 2.3), uniform_np(-1.2, -1.1)]
            random_init_angle *= self.n_robots
        elif self.task in ['releasing', 'placing', 'pushing']:
            # TODO: Find init angle close to the dest position
            random_init_angle = []
        return random_init_angle

    def __sample_goal(self):
        reach_goal = []
        obj_goal = []
        dest_goal = []
        for i in range(self.n_robots):
            reach_goal_pos = np.array([uniform(0.3, 0.42) * sample([-1, 1], 1)[0]
                          for _ in range(2)] + [uniform(0.3, 0.5)])
            xyz = reach_goal_pos - self.base_position[i]
            xyz /= np.linalg.norm(xyz)
            x,y,z = xyz
            alpha = -np.arcsin(y / np.sqrt(y**2+z**2)) * np.sign(x)
            beta = np.arccos(x / np.linalg.norm([x,y,z])) * np.sign(x)
            gamma = uniform(-0.1, 0.1)
            reach_goal_ori = np.array([alpha, beta, gamma], dtype=np.float16)

            reach_goal.append(np.hstack([reach_goal_pos, reach_goal_ori]))
            obj_goal_pos = [uniform(-0.1,0.1), 0.65+uniform(-0.08,0.02), self.object_z]
            # obj_goal_pos = [uniform(-0.2,0.2), 0.65+uniform(-0.1,0.05), self.object_z]
            # obj_goal_pos = [-0.65+uniform(-0.1,0.1), uniform(-0.1,0.1), self.object_z]
            obj_goal.append(obj_goal_pos)
            dest_goal_pos = [0.5+uniform(-0.05,0.05),0.2+uniform(-0.05,0.05),0.3]
            dest_goal.append(dest_goal_pos)
        return np.array(reach_goal), np.array(obj_goal), np.array(dest_goal)

    def _get_observation(self):
        if self.state_gen == None:
            self.__get_gripper_pose()
            self.touch_index = self._get_touch()
            # print(self.touch_index)
            observation = []
            for i in range(self.n_robots):
                xyz = self.interface.get_xyz('object_body' ) - self.gripper_pose[0][:3]
                obj_diff = np.linalg.norm(xyz)
                if self.controller:
                    if self.obs_prev_action:
                        observation.append(np.hstack([
                            self.touch_index,
                            self.gripper_pose[i][:3],
                            self.gripper_pose[i][3:]/np.pi,
                            [(self.gripper_angle_1-0.65)/0.35], 
                            self.prev_action[:6],
                            self.__get_property('object_body','pose')[0][:3],
                            self.__get_property('object_body','pose')[0][3:]/np.pi,
                            self.dest_goal[i], 
                            self.reaching_goal[i][:3],
                            self.reaching_goal[i][3:]/np.pi
                        ]))
                    else:
                        # Observation dimensions: 
                        # touchidx     proprio          object       destination        reaching
                        #    0,    1,2,3,4,5,6, 7,  8,9,10,11,12,13,  14,15,16,     17,18,19,20,21,22
                        observation.append(np.hstack([
                            self.touch_index,
                            self.gripper_pose[i][:3],
                            self.gripper_pose[i][3:]/np.pi,
                            # [(self.gripper_angle_1-0.65)/0.35],
                            [(self.gripper_angle_1-0.8)/0.2],
                            self.__get_property('object_body','pose')[0][:3],
                            self.__get_property('object_body','pose')[0][3:]/np.pi,
                            self.dest_goal[i], 
                            self.reaching_goal[i][:3],
                            self.reaching_goal[i][3:]/np.pi
                        ]))
                else:
                    # print('pos: ',self.interface.sim.data.qpos[self.interface.joint_pos_addrs])
                    # print('vel: ',self.interface.sim.data.qvel[self.interface.joint_vel_addrs])
                    observation.append(np.hstack([
                        self.touch_index,
                        self.interface.sim.data.qpos[self.interface.joint_pos_addrs],
                        self.interface.sim.data.qvel[self.interface.joint_vel_addrs],
                        self.gripper_pose[i][:3],
                        self.gripper_pose[i][3:]/np.pi,
                        [(self.gripper_angle_1-0.8)/0.2],
                        self.__get_property('object_body','pose')[0][:3],
                        self.__get_property('object_body','pose')[0][3:]/np.pi,
                        obj_diff,
                        self.dest_goal[i], 
                        self.reaching_goal[i][:3],
                        self.reaching_goal[i][3:]/np.pi
                    ]))
        else:
            image, depth = self._get_camera()
            data = [image, depth]
            observation = self.state_gen.generate(data)
        return np.array(observation, dtype=np.float32)
    
    def _get_camera(self):
        self.interface.offscreen.render(width=640, height=480, camera_id=0)
        image, depth = self.interface.offscreen.read_pixels(width=640, height=480, depth=True)
        image = np.flip(image, 0)
        depth = np.flip(depth, 0)
        #plt.imsave(self.package_path+"/test.jpg", image)
        #plt.imsave(self.package_path+"/test_depth.jpg", depth)
        return image, depth
    
    def _get_pressure(self):
        pass

    def __get_gripper_pose(self):
        self.gripper_pose = self.__get_property('EE', 'pose')

    def _get_reward(self):
        if self.reward_method is None:
            if self.task == 'reaching':
                dist_coef = 5
                dist_th = 1
                angle_coef = 2
                angle_th = np.pi/6
                scale_coef = 0.05

                dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.reaching_goal[0][:3])
                grip = transformations.unit_vector(
                    transformations.quaternion_from_euler(
                        self.gripper_pose[0][3], self.gripper_pose[0][4], self.gripper_pose[0][5], axes="rxyz"))
                tar = transformations.unit_vector(
                    transformations.quaternion_from_euler(
                        self.reaching_goal[0][3], self.reaching_goal[0][4], self.reaching_goal[0][5], axes="rxyz"))
                angle_diff = grip-tar
                ang_diff = np.linalg.norm(angle_diff)

                # Exponential reward
                reward = dist_coef*np.exp(-1/dist_th*dist_diff)/2
                reward += angle_coef*np.exp(-1/angle_th*ang_diff)/(2*(dist_diff*15+1))

                # Negative Rewards
                # Gripper too close to the robot base
                wb_th = 0.15
                wb = np.linalg.norm(self.__get_property('EE', 'position')[0] - self.base_position[0])
                if wb < wb_th:
                    reward -= (wb_th - wb)
                # Gripper too low
                z_th = 0.1
                z = self.gripper_pose[0][2]
                if z < z_th:
                    reward -= (z_th - z)
                # self.curr_action = np.copy(self.prev_action)
                return scale_coef * reward

            elif self.task == 'grasping':
                dist_coef = 5
                dist_th = 0.2
                angle_coef = 2
                angle_th = np.pi/6
                grasp_coef = 5
                grasp_value = 0.5
                height_coef = 100
                scale_coef = 0.05
                
                roll_e,pitch_e,yaw_e = self.__get_property('EE','pose')[0][3:]
                ee_vec = self._get_rotation(roll_e, pitch_e, yaw_e, [0,0,-1], True)
                xyz = self.interface.get_xyz('object_body' ) - self.gripper_pose[0][:3]
                obj_diff = np.linalg.norm(xyz)
                xyz /= obj_diff
                x,y,z = xyz
                pitch = -np.arccos(z)
                yaw = -np.arccos(-x/(np.sqrt(1-z**2)))
                roll = 0
                target_vec = self._get_rotation(roll, pitch, yaw, [0,0,1], False)
                target_vec[2] *= -1
                angle_diff = np.linalg.norm(ee_vec - target_vec)

                # distance reward
                reward = dist_coef * np.exp(-1/dist_th * obj_diff)/2
                # angle reward
                # reward += angle_coef * np.exp(-1/angle_th * angle_diff)/2
                reward += angle_coef * np.exp(-1/angle_th * angle_diff)/2 / (obj_diff*15+1)
                # gripper in-touch reward
                if self.touch_index == 1:
                    reward += grasp_coef * grasp_value * 0.3
                # gripper out-touch negative reward
                elif self.touch_index == 2:
                    reward -= grasp_coef * grasp_value * 0.3
                # gripper grasp reward
                elif self.touch_index == 3:
                    reward += grasp_coef * grasp_value
                # pick up reward
                reward +=  height_coef * (self.interface.get_xyz('object_body')[2] - self.object_z)
                return reward * scale_coef

            elif self.task == 'picking':
                return 0
            elif self.task == 'carrying':
                return 0
            elif self.task == 'releasing':
                return 0
            elif self.task == 'placing':
                return 0
            elif self.task == 'pushing':
                return 0
        elif self.reward_method is not None:
            return self.reward_module(self.gripper_pose, self.reaching_goal)
        else:
            print("\033[31mConstant Reward. SHOULD BE FIXED\033[0m")
            return 30
    
    def _get_rotation(self, roll, pitch, yaw, vec, inv):
        mat_rol = np.array([
                        [1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]
                    ])
        mat_pit = np.array([
                        [np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]
                    ])
        mat_yaw = np.array([
                        [np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1],
                    ])
        rot_ee = np.matmul(mat_yaw, np.matmul(mat_pit, mat_rol))
        if not inv:
            return np.matmul(rot_ee, vec)
        else:
            return np.matmul(rot_ee.T, vec)

    def _get_touch(self):
        slicenum = 13
        touch_array = np.zeros(20)
        for i in range(19):
            touch_array[i] = self.interface.sim.data.get_sensor(str(i)+"_touch")
        touch_array[-1] = self.interface.sim.data.get_sensor("EE_touch")
        # print(touch_array)
        thumb = 1 if np.any(touch_array[1:5][touch_array[1:5]>0.001]) else 0
        index = 1 if np.any(touch_array[5:9][touch_array[5:9]>0.001]) else 0
        pinky = 1 if np.any(touch_array[9:13][touch_array[9:13]>0.001]) else 0
        # Grasped
        if (thumb and index) == 1 or (thumb and pinky) == 1:
            return 3
        # Inner Touch
        if np.any(touch_array[:slicenum][touch_array[:slicenum]>0.001]):
            return 1
        # Outer Touch
        elif np.any(touch_array[slicenum:][touch_array[slicenum:]>0.001]):
            return 2
        # Else
        else:
            return 0

    def _get_terminal_inspection(self):
        self.num_episodes += 1
        dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.reaching_goal[0][:3])
        obj_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.interface.get_xyz('object_body'))
        relx, rely, relz = self.__get_property('EE', 'position')[0] - self.base_position[0]
        wb = np.linalg.norm(self.__get_property('EE', 'position')[0] - self.base_position[0])
        if pi - 0.1 < self.interface.get_feedback()['q'][2] < pi + 0.1:
            print("\033[91m \nUn wanted joint angle - possible singular state \033[0m")
            return True, -20, wb
        else:
            # OB termination
            if abs(relx) >= 1 or abs(rely) >= 1 or abs(relz) >= 1 or wb > 0.9:
                print("\033[91m \nWorkspace out of bound \033[0m")
                return True, -50, wb
            else:
                if self.task == 'reaching':
                    grip = transformations.unit_vector(
                        transformations.quaternion_from_euler(
                            self.gripper_pose[0][3], self.gripper_pose[0][4], self.gripper_pose[0][5], axes="rxyz"))
                    tar = transformations.unit_vector(
                        transformations.quaternion_from_euler(
                            self.reaching_goal[0][3], self.reaching_goal[0][4], self.reaching_goal[0][5], axes="rxyz"))
                    angle_diff = grip-tar
                    ang_diff = np.linalg.norm(angle_diff)
                    if dist_diff < 0.05 and ang_diff < np.pi/6: 
                        print("\033[92m Target Reached \033[0m")
                        return True, 200 - (self.num_episodes*0.1), wb
                    else:
                        return False, 0, wb
                elif self.task == 'grasping':
                    # Too Far
                    if obj_diff > 0.2:
                        print("\033[91m \nGripper too far away from the object \033[0m")
                        return True, -20, wb
                    # Grasped
                    if (self.interface.get_xyz('object_body')[2] > self.object_z + 0.07) and self.touch_index in [1,3]:
                    # if (self.interface.get_xyz('object_body')[2] > self.object_z + 0.12) and self.touch_index in [1,3]:
                        print("\033[92m Grasping Succeeded \033[0m")
                        return True, 200 - (self.num_episodes*0.1), wb
                    # Dropped
                    elif self.interface.get_xyz('object_body')[2] < 0.1:
                        print("\033[91m Dropped \033[0m")
                        return True, -20, wb
                    # Else
                    else:
                        return False, 0, wb
                elif self.task == 'picking':
                    return True, 0, wb
                elif self.task == 'carrying':
                    return True, 0, wb
                elif self.task == 'releasing':
                    return True, 0, wb
                elif self.task == 'placing':
                    return True, 0, wb
                elif self.task == 'pushing':
                    return True, 0, wb

    def _take_action(self, a):
        self.action_in = True
        self.gripper_iter = 0
        self.__get_gripper_pose()
        # self.curr_action = np.array(a)
        if np.any(np.isnan(np.array(a))):
            print("WARNING, nan in action", a)
        if self.controller:
            # Action: Gripper Pose Increments (m,rad)
            # Action scaled to 0.01m, 0.05 rad
            # self.target_pos = self.gripper_pose[0] + np.hstack([a[:3]/100, a[3:6]/20])
            self.target_pos = self.gripper_pose[0] + np.hstack([a[:3]/25, a[3:6]/5])
            # print(self.target_pos[3:], self.gripper_pose[0][3:])
            if len(a) == 8:
                prev1 = self.gripper_angle_1
                prev2 = self.gripper_angle_2
                self.gripper_angle_1 += a[6]
                self.gripper_angle_2 += a[7]
                self.gripper_angle_1 = max(min(self.gripper_angle_1,1),0.3)
                self.gripper_angle_2 = max(min(self.gripper_angle_2,1),0.3)
                self.gripper_angle_1_array = np.linspace(prev1, self.gripper_angle_1, self.skip_frames)
                self.gripper_angle_2_array = np.linspace(prev2, self.gripper_angle_2, self.skip_frames)
            elif len(a) == 7:
                prev1 = self.gripper_angle_1
                self.gripper_angle_1 += a[6]/10
                self.gripper_angle_1 = max(min(self.gripper_angle_1,1),0.6)
                self.gripper_angle_1_array = np.linspace(prev1, self.gripper_angle_1, self.skip_frames)
            elif len(a) == 6:
                self.gripper_angle_1 = 0.5
                self.gripper_angle_2 = 0.5
            self.interface.set_mocap_xyz("hand", self.target_pos[:3])
            self.interface.set_mocap_orientation("hand", transformations.quaternion_from_euler(
                self.target_pos[3], self.target_pos[4], self.target_pos[5], axes="rxyz"))
            # self.interface.set_mocap_orientation("target_reach", transformations.quaternion_from_euler(
            #     self.target_pos[3], self.target_pos[4], self.target_pos[5], axes="rxyz"))
        else:
            if len(a) == 8:
                self.target_signal = a[:6]/0.8
                prev1 = self.gripper_angle_1
                prev2 = self.gripper_angle_2
                self.gripper_angle_1 += a[6]
                self.gripper_angle_2 += a[7]
                self.gripper_angle_1 = max(min(self.gripper_angle_1,1),0.3)
                self.gripper_angle_2 = max(min(self.gripper_angle_2,1),0.3)
                self.gripper_angle_1_array = np.linspace(prev1, self.gripper_angle_1, self.skip_frames)
                self.gripper_angle_2_array = np.linspace(prev2, self.gripper_angle_2, self.skip_frames)
            elif len(a) == 7:
                self.target_signal = a[:6]/0.8
                prev1 = self.gripper_angle_1
                self.gripper_angle_1 += a[6]/10
                self.gripper_angle_1 = max(min(self.gripper_angle_1,1),0.3)
                self.gripper_angle_1_array = np.linspace(prev1, self.gripper_angle_1, self.skip_frames)
            elif len(a) == 6:
                self.target_signal = a[:6]/0.8
                
    def __get_property(self, subject, prop):
        out = []
        for i in range(self.n_robots):
            if prop == 'position':
                if self.n_robots == 1:
                    out.append(self.interface.get_xyz(subject))
                else:
                    prefix = "_"+str(i+1)
                    out.append(self.interface.get_xyz(subject+prefix))
            elif prop == 'orientation':
                if self.n_robots == 1:
                    orientation_quat = self.interface.get_orientation(subject)
                    out.append(transformations.euler_from_quaternion(
                        orientation_quat, 'rxyz'))
                else:
                    prefix = "_"+str(i+1)
                    orientation_quat = self.interface.get_orientation(subject+prefix)
                    out.append(transformations.euler_from_quaternion(orientation_quat, 'rxyz'))
            elif prop == 'pose':
                if self.n_robots == 1:
                    pos = self.interface.get_xyz(subject)
                    orientation_quat = self.interface.get_orientation(subject)
                    ori = transformations.euler_from_quaternion(orientation_quat, 'rxyz')
                    pose = np.append(pos, ori)
                    out.append(pose)
                else:
                    prefix = "_"+str(i+1)
                    pos = self.interface.get_xyz(subject+prefix)
                    orientation_quat = self.interface.get_orientation(subject+prefix)
                    ori = transformations.euler_from_quaternion(orientation_quat, 'rxyz')
                    pose = np.append(pos, ori)
                    out.append(pose)
        return np.copy(out)



if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../abr_control')))
    from assets.mujoco_config import MujocoConfig
    from mujoco import Mujoco
    from abr_control.controllers import OSC
    from abr_control.utils import transformations

    mobile = False
    if not mobile:
        pos = False
        vel = False
        torque = True
        dual = False and torque
        controller = True
        if pos:
            jaco = MujocoConfig('jaco2_position')
        elif vel:
            jaco = MujocoConfig('jaco2_velocity')
        elif torque:
            if dual:
                jaco = MujocoConfig('jaco2_dual_torque', n_robots=2)
                #jaco = MujocoConfig('jaco2_tri_torque', n_robots=3)
            else:
                jaco = MujocoConfig('jaco2_torque')
        interface = Mujoco(jaco, dt=0.005, visualize=True)
        interface.connect()

        if controller:
            ctrl_dof = [True, True, True, True, True, True]
            ctr = OSC(jaco, kp=50, ko=180, kv=20, vmax=[0.3, 0.7854], ctrlr_dof=ctrl_dof)
            #ctr = OSC(jaco, kp=50, ko=180, kv=20, vmax=[2, 5.236], ctrlr_dof=ctrl_dof)
            if dual:
                interface.set_joint_state([1.5, 2, 1.1, 1, 1, 1, 1, 1.1, 1, 1.5, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                interface.set_joint_state([1, 2, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
            for _ in range(2):
                fb = interface.get_feedback()
                if dual:
                    target = np.hstack([0, 0, -0.15, 0, 0, 0, 0, 0, -0,15, 0, 0, 0])
                else:
                    target=np.hstack([0, 0, -0.15, 0, 0, 0])
                u = ctr.generate(
                    q=fb['q'],
                    dq=fb['dq'],
                    target=target                        
                )
                if dual:
                    interface.send_forces([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                else:
                    interface.send_forces([0, 0, 0, 0, 0, 0, 0, 0, 0])
            if dual:
                target_pos1 = interface.get_xyz('EE_1')
                target_pos2 = interface.get_xyz('EE_2')
            else:
                #target_pos = interface.get_xyz('EE') + np.array([0.1, 0.1, 0.1])
                target_pos = np.array([0.4, 0.4, 0.4])
            rand_pos = np.array([uniform(0.3, 0.4) * sample([-1.2, 1.2], 1)[0] for i in range(2)] + [uniform(0.3, 0.5)])
            x,y,z = rand_pos - interface.get_xyz('link1')
            alpha = -np.arcsin(y / np.sqrt(y**2+z**2)) * np.sign(x)
            beta = np.arccos(x / np.linalg.norm([x,y,z])) * np.sign(x)
            target_or = np.array([alpha, beta, uniform(-0.1, 0.1)], dtype=np.float16)
            print(target_or)
            quat = transformations.quaternion_from_euler(target_or[0], target_or[1], target_or[2], axes='rxyz')
            interface.set_mocap_xyz("hand", np.array(rand_pos))
            interface.set_mocap_orientation("hand", quat)
            target_pos = rand_pos
            for _ in range(10):
                while True:
                    fb = interface.get_feedback()
                    if dual:
                        target = np.hstack([target_pos1, target_or, target_pos2, target_or])
                    else:
                        target = np.hstack([target_pos, target_or])
                    u = ctr.generate(
                        q=fb['q'],
                        dq=fb['dq'],
                        target=target
                    )
                    if dual:
                        a1 = interface.get_xyz('EE_1')
                        a2 = interface.get_xyz('EE_2')
                        interface.send_forces(np.hstack([u[:6], [0, 0, 0], u[6:], [0, 0, 0]]))
                        if np.linalg.norm(a1[:] - target_pos1[:]) < 0.01 and np.linalg.norm(a2[:] - target_pos2[:]):
                            print("Reached")
                            break
                    else:
                        a = interface.get_xyz('EE')
                        interface.send_forces(np.hstack([u[:6], [0, 0, 0]]))
                        if np.linalg.norm(a[:] - target_pos[:]) < 0.01:
                            #print("Reached")
                            quat = interface.get_orientation('EE')
                            print(transformations.euler_from_quaternion(quat, 'rxyz'))
                            interface.set_mocap_orientation("hand", quat)
                            #break
                if dual:
                    target_pos1 += np.array([0.01, 0.01, 0.01])
                    target_pos2 += np.array([0.01, 0.01, 0.01])
                else:
                    #target_pos += np.array([0.03, 0.03, 0.03])
                    pass
                #target_or += np.array([0.1, 0.1, 0.1])
        else:
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
                        # print(interface.get_feedback()['q'])

    else:
        jaco = MujocoConfig('mobilejaco')
        interface = Mujoco(jaco, dt=0.005)
        interface.connect()
        ctr = OSC(jaco, kp=2, ctrlr_dof=[True, True, True, True, True, True])
        interface.set_joint_state([1, 2, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
        for _ in range(2):
            fb = interface.get_feedback()
            u = ctr.generate(
                q=fb['q'],
                dq=fb['dq'],
                target=np.hstack([0, 0, -0.15, 0, 0, 0])
            )
            interface.send_forces([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        target_pos = interface.get_xyz('EE') - interface.get_xyz('base_link')
        target_or = np.array([0, 0, 0], dtype=np.float16)
        vel_left = 10
        vel_right = 10
        for i in range(10):
            #vel_left *= -1
            while True:
                fb = interface.get_feedback()
                u = ctr.generate(
                    q=fb['q'],
                    dq=fb['dq'],
                    target=np.hstack([target_pos + interface.get_xyz('base_link'), target_or])
                )
                a = interface.get_xyz('EE') - interface.get_xyz('base_link')
                b = interface.get_orientation('EE')
                # print(a)
                interface.send_forces(np.hstack([[vel_left, vel_right, vel_left, vel_right], u, [0, 0, 0]]))
                if np.linalg.norm(a[:] - target_pos[:]) < 0.01:
                    print("Reached")
                    break
            target_pos += np.array([0.01, 0.01, 0.01])
            target_or += np.array([0.1, 0.1, 0.1])
