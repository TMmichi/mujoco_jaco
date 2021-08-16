#!/usr/bin/env python

import os, sys
import time
from pathlib import Path
from math import pi

import numpy as np
from numpy.random import uniform, choice

if __name__ != "__main__":
    from abr_control.controllers import OSC
    from abr_control.utils import transformations
    from env_script.assets.mujoco_config import MujocoConfig
    from env_script.mujoco import Mujoco


class JacoMujocoEnvUtil:
    def __init__(self, controller=True, **kwargs):
        ### ------------  MODEL CONFIGURATION  ------------ ###
        self.seed(kwargs.get('seed', None))
        self.n_robots = kwargs.get('n_robots', 1)
        robot_file = kwargs.get('robot_file', "jaco2_curtain_torque")
        if robot_file == None:
            n_robot_postfix = ['', '_dual', '_tri']    
            try:
                xml_name = 'jaco2'+n_robot_postfix[self.n_robots-1]
            except Exception:
                raise NotImplementedError(
                    "\n\t\033[91m[ERROR]: xml_file of the given number of robots doesn't exist\033[0m")
        else:
            xml_name = robot_file
        self.jaco = MujocoConfig(xml_name, n_robots=self.n_robots)
        self.interface = Mujoco(self.jaco, 
                                dt=0.001, 
                                visualize=kwargs.get('visualize', True), 
                                create_offscreen_rendercontext=False)
        self.interface.connect()
        self.manipulator_name = ['Left','Right']

        self.skip_frames = 50
        self.base_position = self.__get_property('base_link', 'position')
        self.base_orientation = self.__get_property('base_link', 'orientation')
        print(self.base_orientation)
        self.gripper_angle = np.ones(self.n_robots)*0.6
        self.gripper_angle_array = np.zeros((self.n_robots, self.skip_frames))
        self.gripper_iter = 0
        self.object_z = 0.1898
        self.action_in = False
        self.ctrl_type = self.jaco.ctrl_type
        print("control type: ", self.ctrl_type)
        self.task = kwargs.get('task', None)
        self.num_episodes = 0
        self.goal_buffer = kwargs.get('init_buffer', None)
        self.subgoal_obs = kwargs.get('subgoal_obs', False)
        self.rulebased_subgoal = kwargs.get('rulebased_subgoal',False)
        self.auxiliary = kwargs.get('auxiliary', False)
        self.binary = False
        print('subgoal observation: ', self.subgoal_obs)

        ### ------------  STATE GENERATION  ------------ ###
        self.package_path = str(Path(__file__).resolve().parent.parent)
        self.state_gen = kwargs.get('stateGen', None)
        self.image_buffersize = 5
        self.image_buff = []
        self.pressure_buffersize = 100
        self.pressure_buff = []
        self.data_buff = []

        ### ------------  CONTROLLER SETUP  ------------ ###
        ctrl_dof = [True, True, True, True, True, True]
        self.ctr = OSC(self.jaco, kp=50, ko=180, kv=20, vmax=[0.4, 1.0472], ctrlr_dof=ctrl_dof)
        self._reset()
        self.target_pos = self.gripper_pose[0]

        ### ------------  REWARD  ------------ ###
        self.reward_method = kwargs.get('reward_method', None)
        self.reward_module = kwargs.get('reward_module', None)
    

    def _step_simulation(self):
        fb = self.interface.get_feedback()
        u = self.__controller_generate(fb)
        force_signal = np.array([])
        for i in range(self.n_robots):
            if self.action_in:
                griper_angle = [self.gripper_angle_array[i][self.gripper_iter]]*3
            else:
                griper_angle = [self.gripper_angle[i]]*3
            force_signal = np.hstack([force_signal, u[6*i:6*(i+1)], griper_angle])
        self.interface.send_forces(force_signal)
        self.gripper_iter += 1

    def __controller_generate(self, fb):
        return self.ctr.generate(
            q=fb['q'],
            dq=fb['dq'],
            target=self.target_pos
        )

    def _reset(self):
        self.interface.sim.reset()
        self.picked = False
        self.picked_arr = [False] * self.n_robots
        self.loose_grap = [False] * self.n_robots
        self.r_reached = False
        self.num_episodes = 0
        self.gripper_iter = 0
        self.touch_index = 0
        self.action_in = False
        self.gripper_angle = np.ones(self.n_robots)*0.6
        self.interface.viewer._paused = False
        

        init_angle = self._create_init_angle()
        self.interface.set_joint_state(init_angle, [0]*6*self.n_robots)
        self.reaching_goal, self.obj_goal, self.dest_goal = self.__sample_goal()

        # Reaching Init
        if self.task == 'reaching':
            self.interface.set_mocap_xyz("target_reach", self.reaching_goal[0][:3])
            self.interface.set_mocap_orientation("target_reach", transformations.quaternion_from_euler(
                self.reaching_goal[0][3], self.reaching_goal[0][4], self.reaching_goal[0][5], axes="rxyz"))
            self.target_pos = np.reshape(np.hstack([self.reaching_goal]),(-1))
        # Object in the Gripper
        elif self.task in ['releasing', 'placing']:
            for _ in range(150):
                for i in range(self.n_robots):
                    pos = self.__get_property('EE_obj', 'pose')[i]
                    quat = transformations.quaternion_from_euler(pos[3], pos[4], pos[5], axes='rxyz')
                    del_pos = self._get_rotation(pos[3], pos[4], pos[5], [-0.04, 0, 0])
                    pos[:3] += del_pos
                    self.interface.set_obj_xyz(pos[:3], quat, idx=i)
                    self.__get_gripper_pose()
                    self.target_pos = self.gripper_pose.flatten()
                    self.gripper_angle_array[i] = np.zeros(self.skip_frames)
                    self._step_simulation()
                    self.interface.set_obj_xyz(pos[:3], quat, idx=i)
        else:
            for i in range(self.n_robots):
                self.interface.set_obj_xyz(self.obj_goal[i], [0,0,0,0], idx=i)

        # Grasping Init
        if self.task is 'grasping':
            self.__get_gripper_pose()
            self.target_pos = np.array([])
            for i in range(self.n_robots):
                x,y,z = self.obj_goal[i] - self.gripper_pose[i][:3]
                alpha = -np.arcsin(y / np.sqrt(y**2+z**2)) * np.sign(x)
                beta = np.arccos(x / np.linalg.norm([x,y,z])) * np.sign(x)
                gamma = uniform(-0.1, 0.1)
                self.target_pos = np.hstack([self.target_pos, self.obj_goal[i], [alpha, beta, gamma]])
            while True:
                self.interface.stop_obj()
                self._step_simulation()
                self.__get_gripper_pose()
                dist_diff = np.zeros(self.n_robots)
                ang_diff = np.zeros(self.n_robots)
                for i in range(self.n_robots):
                    dist_diff[i] = np.linalg.norm(self.gripper_pose[i][:3] - self.obj_goal[i])
                    grip = transformations.unit_vector(
                        transformations.quaternion_from_euler(
                            self.gripper_pose[i][3], self.gripper_pose[i][4], self.gripper_pose[i][5], axes="rxyz"))
                    tar = transformations.unit_vector(
                        transformations.quaternion_from_euler(
                            self.reaching_goal[i][3], self.reaching_goal[i][4], self.reaching_goal[i][5], axes="rxyz"))
                    angle_diff = grip-tar
                    ang_diff[i] = np.linalg.norm(angle_diff)
                if np.all(dist_diff[dist_diff<0.2]):
                    self.target_pos = np.array([])
                    for i in range(self.n_robots):
                        self.target_pos = np.hstack([self.target_pos, self.gripper_pose[i][:3], [alpha, beta, gamma]])
                    break
                if np.all(ang_diff[ang_diff>np.pi/6]):
                    break
            while True:
                self._step_simulation()
                self.__get_gripper_pose()
                # self.target_pos = np.reshape(np.hstack([self.obj_goal,np.repeat([ori_toward],self.n_robots,axis=0)]),(-1))
                self.target_pos = np.hstack([self.target_pos, self.obj_goal[i], [alpha, beta, gamma]])
                dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.obj_goal[0])
                if dist_diff < 0.15:
                    break
        elif self.task == "bimanipulation":
            self.__get_gripper_pose()
            x,y,z = self.obj_goal[1] - self.gripper_pose[1][:3]
            alpha = -np.arcsin(y / np.sqrt(y**2+z**2)) * np.sign(x)
            beta = np.arccos(x / np.linalg.norm([x,y,z])) * np.sign(x)
            gamma = uniform(-0.1, 0.1)
            reach_goal_ori = np.array([alpha, beta, gamma], dtype=np.float16)
            self.target_pos = np.hstack([self.gripper_pose[0], self.obj_goal[1], reach_goal_ori])
            while True:
                dist_reached = False
                self.interface.stop_obj()
                self._step_simulation()
                self.__get_gripper_pose()
                dist_diff = np.linalg.norm(self.gripper_pose[1][:3] - self.obj_goal[1])
                grip = transformations.unit_vector(
                    transformations.quaternion_from_euler(
                        self.gripper_pose[1][3], self.gripper_pose[1][4], self.gripper_pose[1][5], axes="rxyz"))
                tar = transformations.unit_vector(
                transformations.quaternion_from_euler(
                    self.reaching_goal[1][3], self.reaching_goal[1][4], self.reaching_goal[1][5], axes="rxyz"))
                angle_diff = grip-tar
                ang_diff = np.linalg.norm(angle_diff)
                if dist_diff < 0.2:
                    self.target_pos = np.hstack([self.gripper_pose[0], self.gripper_pose[1][:3], reach_goal_ori])
                    break
                if ang_diff < np.pi/6 and dist_reached:
                    break
            while True:
                self._step_simulation()
                self.__get_gripper_pose()
                self.target_pos = np.hstack([self.gripper_pose[0], self.obj_goal[1], reach_goal_ori])
                dist_diff = np.linalg.norm(self.gripper_pose[1][:3] - self.obj_goal[1])
                if dist_diff < 0.15:
                    break
        obs = self._get_observation()
        print(obs)
        return obs

    def _create_init_angle(self):
        if self.task in ['reaching', 'picking', 'pickAndplace']:
            random_init_angle = [uniform(0.7, 2.5), uniform(3.8,4), uniform(
                    1, 1.7), uniform(1.8, 2.5), uniform(1, 2.5), uniform(0.8, 2.3)] * self.n_robots
        elif self.task in ['carrying', 'grasping', 'placing']:
            angle0 = np.random.choice([uniform(3*pi/8,pi/2), uniform(pi/2,5*pi/8)])
            random_init_angle = [angle0, 3.85, uniform(
                    1, 1.1), uniform(2, 2.1), uniform(0.8, 2.3), uniform(-1.2, -1.1)] * self.n_robots
        elif self.task is 'releasing':
            random_init_angle = [uniform(1.9, 2), uniform(3.3,3.6), uniform(
                    0.5, 0.8), uniform(1.8, 2.5), uniform(1.3, 2), uniform(-0.4, -0.9), 0.6, 0.6, 0.6] * self.n_robots
        elif self.task is 'bimanipulation':
            grip_1 = [uniform(0.7, 2.5), uniform(3.8,4), uniform(
                    1, 1.7), uniform(1.8, 2.5), uniform(1, 2.5), uniform(0.8, 2.3)]
            angle0 = np.random.choice([uniform(3*pi/8,pi/2), uniform(pi/2,5*pi/8)])
            grip_2 = [angle0, 3.85, uniform(
                    1, 1.1), uniform(2, 2.1), uniform(0.8, 2.3), uniform(-1.2, -1.1)]
            random_init_angle = np.concatenate([grip_1, grip_2])
        return random_init_angle

    def __sample_goal(self):
        # Goal always in the global coordinate
        reach_goal = []
        obj_goal = []
        dest_goal = []
        for i in range(self.n_robots):
            # Reaching Goal
            if self.goal_buffer is None:
                reach_goal_pos = np.array([uniform(0.3, 0.42) * choice([-1, 1])
                            for _ in range(2)] + [uniform(0.3, 0.5)])
                xyz = reach_goal_pos - self.base_position[i]
                xyz /= np.linalg.norm(xyz)
                x,y,z = xyz
                alpha = -np.arcsin(y / np.sqrt(y**2+z**2)) * np.sign(x)
                beta = np.arccos(x / np.linalg.norm([x,y,z])) * np.sign(x)
                gamma = uniform(-0.1, 0.1)
                reach_goal_ori = np.array([alpha, beta, gamma], dtype=np.float16)
            else:
                random_idx = np.random.randint(0,len(self.goal_buffer)-1)
                reach_goal_pos = self.goal_buffer[random_idx][1:4]
                reach_goal_ori = self.goal_buffer[random_idx][4:7]
            reach_goal.append(np.hstack([reach_goal_pos, reach_goal_ori]))
            # Obj goal
            obj_pos_local = np.array([uniform(-0.1,0.1), 0.65+uniform(-0.08,0.02), self.object_z])
            obj_goal_pos = self.local_to_global(obj_pos_local,'coordinate',i)
            obj_goal.append(obj_goal_pos)
            # Dest goal
            dest_pos_local = np.array([0.4+uniform(-0.05,0.05), 0.3+uniform(-0.05,0.05), 0.3468])
            dest_goal_pos = self.local_to_global(dest_pos_local, 'coordinate', i)
            dest_goal.append(dest_goal_pos)
        return np.array(reach_goal), np.array(obj_goal), np.array(dest_goal)

    def _get_observation(self):
        self.__get_gripper_pose()
        self.touch_index = self._get_touch()
        observation = []
        for i in range(self.n_robots):
            if self.subgoal_obs:
                observation.append(np.hstack([
                    self.touch_index[i],
                    self.gripper_pose[i][:3] - self.base_position[i],
                    self.gripper_pose[i][3:]/np.pi,
                    # [(self.gripper_angle_1-0.65)/0.35],
                    [(self.gripper_angle[i]-0.8)/0.2],
                    self.__get_property('object_body','pose')[i][:3] - self.base_position[i],
                    # self.__get_property('object_body','pose')[0][3:]/np.pi,
                    [0,0,0],
                    self.dest_goal[i] - self.base_position[i],
                    self.gripper_pose[i][:3],
                    self.gripper_pose[i][3:]/np.pi,
                    [0,np.pi/2,0]
                ]))
            elif self.rulebased_subgoal:
                pos, ori = self._get_rulebased_subgoal()
                observation.append(np.hstack([
                    # Touch idx: 0 | 26
                    self.touch_index[i],
                    # Proprio (End-Effector): 1,2,3,4,5,6 | 27,28,29,30,31,32
                    self.global_to_local(self.gripper_pose[i][:3], 'coordinate', i),
                    self.global_to_local(self.gripper_pose[i][3:], 'orientation', i)/np.pi,
                    # Proprio (Gripper angle): 7 | 33
                    [(self.gripper_angle[i]-0.8)/0.2],
                    # Object pose: 8,9,10,11,12,13 | 34,35,36,37,38,39
                    self.global_to_local(self.__get_property('object_body','pose')[i][:3], 'coordinate', i),
                    self.global_to_local([0,0,0], 'orientation', i),
                    # Destination position: 14,15,16 | 40,41,42
                    self.global_to_local(self.dest_goal[i], 'coordinate', i),
                    # Reaching Target pose: 17,18,19,20,21,22 | 43,44,45,46,47,48
                    self.global_to_local(pos[i], 'coordinate', i),
                    self.global_to_local(ori[i], 'orientation', i)/np.pi,
                    # Destination Orientation: 23,24,25 | 49,50,51
                    self.global_to_local(np.array([0,np.pi/2,0]), 'orientation', i)
                ]))
            else:
                # Observation dimensions: 
                # touchidx     proprio          object       destination        reaching         destination_ori
                #    0,    1,2,3,4,5,6, 7,  8,9,10,11,12,13,  14,15,16,     17,18,19,20,21,22        23,24,25
                #   26,    1,2,3,4,5,6, 7,  8,9,10,11,12,13,  14,15,16,     17,18,19,20,21,22        23,24,25
                observation.append(np.hstack([
                    self.touch_index[i],
                    self.gripper_pose[i][:3] - self.base_position[i],
                    self.gripper_pose[i][3:]/np.pi,
                    # [(self.gripper_angle_1-0.65)/0.35],
                    [(self.gripper_angle[i]-0.8)/0.2],
                    self.__get_property('object_body','pose')[i][:3] - self.base_position[i],
                    # self.__get_property('object_body','pose')[0][3:]/np.pi,
                    [0,0,0],
                    self.dest_goal[i],
                    self.reaching_goal[i][:3] - self.base_position[i],
                    self.reaching_goal[i][3:]/np.pi,
                    [0,np.pi/2,0]
                ]))
        return np.array(observation, dtype=np.float32).flatten()
    
    def _get_rulebased_subgoal(self):
        # NOTE: rulebased subgoal
        # location: 0.15(m) away from the obj_goal with a direction of a vector from obj_goal to EE
        # orientation: direction of a vector from subgoal location to the obj_goal
        subgoal_pos_arr = []
        subgoal_ori_arr = []
        for i in range(self.n_robots):
            if i == 0:
                D = self.gripper_pose[i][:3] - self.obj_goal[i]
                subgoal_pos = D/np.linalg.norm(D) * 0.12 + self.obj_goal[i] + np.array([(np.random.uniform()-0.5)/25 for _ in range(3)])
                if subgoal_pos[2] < self.object_z+0.1:
                    subgoal_pos[2] = self.object_z+0.1
                if subgoal_pos[1] > self.__get_property('object_body','pose')[i][1]-0.15:
                    subgoal_pos[1] = self.__get_property('object_body','pose')[i][1]-0.15 
                xyz = np.copy(-D)
                obj_diff = np.linalg.norm(xyz)
                xyz /= obj_diff
                x,y,z = xyz
                quatpi = np.sqrt(2)/2
                quat_vec = np.array([quatpi, quatpi*x, quatpi*y, quatpi*z])
                quat_dummy = np.array([np.sqrt(3)/2,0.5*x,0.5*y,0.5*z])
                rpy_vec = np.array(transformations.euler_from_quaternion(quat_vec, 'rxyz'))
                rpy_dummy = np.array(transformations.euler_from_quaternion(quat_dummy, 'rxyz'))
                rpy_real = np.cross(rpy_vec, rpy_dummy)
                rpy_real[0] -= pi/2
                subgoal_ori = rpy_real + np.array([(np.random.uniform()-0.5)/10 for _ in range(3)])
            elif i == 1:
                subgoal_pos = self.dest_goal[0]
                subgoal_ori = np.array([0, -np.pi/2, 0])
            subgoal_pos_arr.append(subgoal_pos)
            subgoal_ori_arr.append(subgoal_ori)
        if self.task == 'placing':
            return np.array(self.dest_goal[0]), np.array([0,np.pi/2,0])
        else:
            return subgoal_pos_arr, subgoal_ori_arr

    def _get_camera(self):
        self.interface.offscreen.render(width=640, height=480, camera_id=0)
        image, depth = self.interface.offscreen.read_pixels(width=640, height=480, depth=True)
        image = np.flip(image, 0)
        depth = np.flip(depth, 0)
        #plt.imsave(self.package_path+"/test.jpg", image)
        #plt.imsave(self.package_path+"/test_depth.jpg", depth)
        return image, depth

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
                # angle_diff = grip-tar
                # ang_diff = np.linalg.norm(angle_diff)
                grip_euler = transformations.euler_from_quaternion(grip,'rxyz')
                tar_euler = transformations.euler_from_quaternion(tar,'rxyz')
                ang_diff = np.linalg.norm(np.array(grip_euler) - np.array(tar_euler))
                if ang_diff > np.pi:
                    ang_diff = 2*np.pi - ang_diff
                
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
                ee_vec = self._get_rotation(roll_e, pitch_e, yaw_e, [0,0,-1], False)
                xyz = self.interface.get_xyz('object_body' ) - self.gripper_pose[0][:3]
                obj_diff = np.linalg.norm(xyz)
                xyz /= obj_diff
                x,y,z = xyz
                pitch = -np.arccos(z)
                yaw = -np.arccos(-x/(np.sqrt(1-z**2)))
                roll = 0
                target_vec = self._get_rotation(roll, pitch, yaw, [0,0,1], True)
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
                if not self.binary:
                    dist_coef = 5
                    dist_th = 0.2
                    angle_coef = 2
                    angle_th = np.pi/6
                    grasp_coef = 5
                    grasp_value = 0.5
                    height_coef = 100
                    scale_coef = 0.01
                    
                    roll_e,pitch_e,yaw_e = self.__get_property('EE','pose')[0][3:]
                    ee_vec = self._get_rotation(roll_e, pitch_e, yaw_e, [0,0,-1], False)
                    xyz = self.interface.get_xyz('object_body' ) - self.gripper_pose[0][:3]
                    obj_diff = np.linalg.norm(xyz)
                    xyz /= obj_diff
                    x,y,z = xyz
                    pitch = -np.arccos(z)
                    yaw = -np.arccos(-x/(np.sqrt(1-z**2)))
                    roll = 0
                    target_vec = self._get_rotation(roll, pitch, yaw, [0,0,1], True)
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
                else:
                    return 0
            elif self.task == 'carrying':
                return 0
            elif self.task == 'releasing':
                return 0
            elif self.task == 'placing':
                return 0
            elif self.task == 'pushing':
                return 0
            elif self.task == 'pickAndplace':
                return 0
            elif self.task == 'bimanipulation':
                if np.all(self.picked_arr):
                    dist_coef = 5
                    dist_th = 0.5
                    obj_position = self.__get_property('object_body', 'position')
                    dist_diff = np.linalg.norm(obj_position[0] - obj_position[1])
                    reward = dist_coef*np.exp(-1/dist_th*dist_diff)
                    return reward * 0.05
                else:
                    return 0
        elif self.reward_method is not None:
            return self.reward_module(self.gripper_pose, self.reaching_goal)
        else:
            print("\033[31mConstant Reward. SHOULD BE FIXED\033[0m")
            return 30
    
    def _get_rotation(self, roll, pitch, yaw, vec, inv=False):
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
        # rot_ee = np.matmul(mat_yaw, np.matmul(mat_pit, mat_rol))
        rot_ee = np.matmul(mat_rol , np.matmul(mat_pit, mat_yaw))
        if not inv:
            return np.matmul(rot_ee, vec)
        else:
            return np.matmul(rot_ee.T, vec)

    def _get_touch(self):
        slicenum = 13
        touch_array = np.zeros((self.n_robots, 20))
        touch_idx = np.zeros(self.n_robots)
        for i in range(self.n_robots):
            for j in range(19):
                touch_sensor_name = str(j)+"_touch"
                palm_sensor_name = "EE_touch"
                if self.n_robots is not 1:
                    touch_sensor_name += "_" + str(i+1)
                    palm_sensor_name += "_" + str(i+1)
                touch_array[i][j] = self.interface.sim.data.get_sensor(touch_sensor_name)
            touch_array[i][-1] = self.interface.sim.data.get_sensor(palm_sensor_name)

            thumb = 1 if np.any(touch_array[i][1:5][touch_array[i][1:5]>0.001]) else 0
            index = 1 if np.any(touch_array[i][5:9][touch_array[i][5:9]>0.001]) else 0
            pinky = 1 if np.any(touch_array[i][9:13][touch_array[i][9:13]>0.001]) else 0
            # Grasped
            if (thumb and index) == 1 or (thumb and pinky) == 1:
                touch_idx[i] = 3
            # Inner Touch
            if np.any(touch_array[i][:slicenum][touch_array[i][:slicenum]>0.001]):
                touch_idx[i] = 1
            # Outer Touch
            elif np.any(touch_array[i][slicenum:][touch_array[i][slicenum:]>0.001]):
                touch_idx[i] = 2
            # Else
            else:
                touch_idx[i] = 0
        return touch_idx

    def _get_terminal_inspection(self):
        self.num_episodes += 1
        # obj_position = self.interface.get_xyz('object_body_1')
        obj_position = self.__get_property('object_body', 'position')
        obj_velocity = np.linalg.norm(self.interface.get_obj_vel())
        dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.reaching_goal[0][:3])
        obj_diff = np.linalg.norm(self.gripper_pose[0][:3] - obj_position[0])
        dest_diff = np.linalg.norm(self.dest_goal[0][:2] - obj_position[0][:2])
        relx, rely, relz = self.__get_property('EE', 'position')[0] - self.base_position[0]
        wb = np.linalg.norm(self.__get_property('EE', 'position')[0] - self.base_position[0])
        if len(self.target_pos) < 9:
            if pi - 0.1 < self.interface.get_feedback()['q'][2] < pi + 0.1:
                print("\033[91m \nUn wanted joint angle - possible singular state \033[0m")
                return True, -1, wb
        else:
            if pi - 0.1 < self.interface.get_feedback()['q'][2] < pi + 0.1 or pi - 0.1 < self.interface.get_feedback()['q'][8] < pi + 0.1:
                print("\033[91m \nUn wanted joint angle - possible singular state \033[0m")
                return True, -20, wb

        if self.task == 'reaching':
            grip = transformations.unit_vector(
                transformations.quaternion_from_euler(
                    self.gripper_pose[0][3], self.gripper_pose[0][4], self.gripper_pose[0][5], axes="rxyz"))
            tar = transformations.unit_vector(
                transformations.quaternion_from_euler(
                    self.reaching_goal[0][3], self.reaching_goal[0][4], self.reaching_goal[0][5], axes="rxyz"))
            # angle_diff = grip-tar
            # ang_diff = np.linalg.norm(angle_diff)
            grip_euler = transformations.euler_from_quaternion(grip,'rxyz')
            tar_euler = transformations.euler_from_quaternion(tar,'rxyz')
            ang_diff = np.linalg.norm(np.array(grip_euler) - np.array(tar_euler))
            if ang_diff > np.pi:
                ang_diff = 2*np.pi - ang_diff
            if dist_diff < 0.025 and ang_diff < np.pi/6: 
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
            if not self.binary:
                # Picked
                if (self.interface.get_xyz('object_body')[2] > self.object_z + 0.07) and self.touch_index in [1,3]:
                    print("\033[92m Picking Succeeded \033[0m")
                    return True, 200 - (self.num_episodes*0.1), wb
                # Dropped
                elif self.interface.get_xyz('object_body')[2] < 0.1:
                    print("\033[91m Dropped \033[0m")
                    return True, -20, wb
                # Else
                else:
                    return False, 0, wb
            else:
                # Picked
                if (self.interface.get_xyz('object_body')[2] > self.object_z + 0.07) and self.touch_index in [1,3]:
                    print("\033[92m Picking Succeeded \033[0m")
                    return True, 10, wb
                # Dropped
                elif self.interface.get_xyz('object_body')[2] < 0.1:
                    print("\033[91m Dropped \033[0m")
                    return True, -1, wb
                # Else
                else:
                    return False, 0, wb
        elif self.task == 'carrying':
            return True, 0, wb
        elif self.task == 'releasing':
            # Dropped
            if obj_position[2] < 0.1:
                print("\033[91m Dropped \033[0m")
                return True, -20, wb
            # Released Success
            elif dest_diff < 0.04 and self.touch_index == 0 and obj_position[2] < 0.35 and obj_velocity < 0.01:
                print("\033[92m Releasing Succeeded \033[0m")
                return True, 200 - (self.num_episodes*0.1), wb
            # Released Wrong
            elif dest_diff > 0.04 and self.touch_index == 0 and obj_position[2] < 0.20:
                print("\033[91m Released at the wrong position \033[0m")
                return True, -20, wb
            # Else
            else:
                return False, 0, wb
        elif self.task == 'placing':
            # Dropped
            if obj_position[2] < 0.1:
                print("\033[91m Dropped \033[0m")
                return True, -20, wb
            # Released Success
            elif dest_diff < 0.04 and self.touch_index == 0 and obj_position[2] < 0.35:
                print("\033[92m Placing Succeeded \033[0m")
                return True, 200 - (self.num_episodes*0.1), wb
            # Released Wrong
            elif dest_diff > 0.04 and self.touch_index == 0 and obj_position[2] < 0.20:
                print("\033[91m Released at the wrong position \033[0m")
                return True, -20, wb
            # Else
            else:
                return False, 0, wb
        elif self.task == 'pushing':
            return True, 0, wb
        elif self.task == 'pickAndplace':
            # Picked
            # if (self.__get_property('object_body', 'position')[0][2] > self.object_z + 0.07) and self.touch_index[0] in [1,3] and not self.picked:
            if obj_position[0][2] > self.object_z + 0.07 and self.touch_index[0] in [1,3] and not self.picked:
                print("\033[92m Picked \033[0m")
                self.picked = True
                return False, 20, wb
            # Released Success
            elif dest_diff < 0.04 and self.touch_index[0] == 0 and obj_position[0][2] < 0.35:
                print("\033[92m pickAndplace Succeeded \033[0m")
                return True, 180, wb
            # Dropped
            elif obj_position[0][2] < 0.1:
                print("\033[91m Dropped \033[0m")
                return True, -20, wb
            else:
                return False, 0, wb
        elif self.task == 'bimanipulation':
            for i in range(self.n_robots):
                if obj_position[i][2] > self.object_z + 0.07 and self.touch_index[i] in [1,3] and not self.picked_arr[i]:
                    print("\033[92m"+self.manipulator_name[i]+" Picked \033[0m")
                    self.picked_arr[i] = True
                    self.loose_grap[i] = False
                    return False, 10, wb
                if np.linalg.norm(obj_position[0] - obj_position[1]) < 0.03:
                    print("\033[92m bimanipulation Succeeded \033[0m")
                    return True, 200, wb
                if obj_position[i][2] < 0.1:
                    print("\033[91m Dropped \033[0m")
                    if self.loose_grap[i]:
                        return True, 0, wb
                    else:
                        return True, -20, wb
                if self.picked_arr[i] and (obj_position[i][2] < self.object_z + 0.01 and self.touch_index[i] in [0]):
                    self.picked_arr[i] = False
                    self.loose_grap[i] = True
                    print("\033[91m Loose Grap \033[0m")
                    return False, -10, wb
            if self.gripper_pose[1][0]-self.base_position[1][0] < -0.2 and not self.picked_arr[1]:
                print("\033[91m Gripper nothing at hand \033[0m")
                return True, -10, wb
            if self.gripper_pose[1][0]-self.base_position[1][0] > 0.2:
                print("\033[91m Gripper wrong way \033[0m")
                return True, -10, wb
            if not np.any(self.picked_arr) and np.linalg.norm(self.gripper_pose[0][:3]-self.gripper_pose[1][:3])<0.3:
                print("\033[91m Too close \033[0m")
                return True, -10, wb
            if self.picked_arr[1] and np.linalg.norm(obj_position[1]-self.dest_goal[0])<0.1 and not self.r_reached:
                self.r_reached = True
                print("\033[92m Right arm reached to the desired location with the object \033[0m")
                return False, 40, wb
            if self.r_reached and np.linalg.norm(obj_position[1]-self.dest_goal[0])>0.1:
                print("\033[91m Right arm out of the location \033[0m")
                return True, -40, wb
            return False, 0, wb

    def _take_action(self, a, weight=None, subgoal=None, id=None):
        self.action_in = True
        self.gripper_iter = 0
        self.__get_gripper_pose()

        if self.rulebased_subgoal:
            subpos, subori = self._get_rulebased_subgoal()
            subgoal_reach = np.append(subpos, subori)
        else:
            subgoal_reach = subgoal['level1_reaching/level0'][0] + self.target_pos
        
        # self.interface.set_mocap_xyz("subgoal_reach", subgoal_reach[:3])
        # self.interface.set_mocap_orientation("subgoal_reach", transformations.quaternion_from_euler(
        #     subgoal_reach[3], subgoal_reach[4], subgoal_reach[5], axes="rxyz"))

        if np.any(np.isnan(np.array(a))):
            print("WARNING, nan in action", a)
            a = np.zeros_like(a)
        # Action: Gripper Pose Increments (m,rad)
        # Action scaled to 0.01m, 0.05 rad
        if len(a) < 9:
            self.target_pos = self.gripper_pose[0] + np.hstack([a[:3]/100, a[3:6]/20])
        else:
            self.target_pos = self.gripper_pose.flatten() + np.hstack([a[:3]/25, a[3:6]/5, a[7:10]/25, a[10:13]/5])
        if len(self.target_pos) < 9:
            if abs(self.target_pos[5]) > pi:
                self.target_pos[5] += -np.sign(self.target_pos[5])*2*pi
        else:
            if abs(self.target_pos[11]) > pi:
                self.target_pos[11] += -np.sign(self.target_pos[11])*2*pi
        if len(a) == 7:
            prev1 = self.gripper_angle[0]
            self.gripper_angle[0] += a[6]/10
            self.gripper_angle[0] = max(min(self.gripper_angle[0],1),0.6)
            self.gripper_angle_array[0] = np.linspace(prev1, self.gripper_angle[0], self.skip_frames)
        elif len(a) == 14:
            prev1 = self.gripper_angle[0]
            self.gripper_angle[0] += a[6]/10
            self.gripper_angle[0] = max(min(self.gripper_angle[0],1),0.6)
            self.gripper_angle_array[0] = np.linspace(prev1, self.gripper_angle[0], self.skip_frames)
            prev1 = self.gripper_angle[1]
            self.gripper_angle[1] += a[13]/10
            self.gripper_angle[1] = max(min(self.gripper_angle[1],1),0.6)
            self.gripper_angle_array[1] = np.linspace(prev1, self.gripper_angle[1], self.skip_frames)
        
        if self.n_robots == 1:
            self.interface.set_mocap_xyz("hand", self.target_pos[:3])
            self.interface.set_mocap_orientation("hand", transformations.quaternion_from_euler(
                self.target_pos[3], self.target_pos[4], self.target_pos[5], axes="rxyz"))
        else:
            for i in range(self.n_robots):
                self.interface.set_mocap_xyz("hand_"+str(i+1), self.target_pos[:3])
                self.interface.set_mocap_orientation("hand_"+str(i+1), transformations.quaternion_from_euler(
                    self.target_pos[3], self.target_pos[4], self.target_pos[5], axes="rxyz"))
                
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
    
    def global_to_local(self, prop, type, robot_idx):
        translation = self.base_position[robot_idx]
        rol,pit,yaw = self.base_orientation[robot_idx]
        if type == 'coordinate':
            return self._get_rotation(rol,pit,yaw,prop-translation,inv=True)
        elif type == 'orientation':
            return self._get_rotation(rol,pit,yaw,prop-translation,inv=True)
    
    def local_to_global(self, prop, type, robot_idx):
        translation = self.base_position[robot_idx]
        rol,pit,yaw = self.base_orientation[robot_idx]
        if type == 'coordinate':
            return self._get_rotation(rol,pit,yaw,prop) + translation
        elif type == 'orientation':
            return self._get_rotation(rol,pit,yaw,prop) + translation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def set_capture_path(self, path):
        self.interface.viewer._image_path = path
    
    def capture(self):
        pass

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
        dual = True and torque
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
            if dual:
                interface.set_joint_state([1.5, 2, 1.1, 1, 1, 1, 1.5, 2, 1.1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                interface.viewer._paused = True
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
                orientation_quat1 = interface.get_orientation('EE_1')
                target_or1 = transformations.euler_from_quaternion(orientation_quat1, 'rxyz')
                orientation_quat2 = interface.get_orientation('EE_2')
                target_or2 = transformations.euler_from_quaternion(orientation_quat2, 'rxyz')
            else:
                #target_pos = interface.get_xyz('EE') + np.array([0.1, 0.1, 0.1])
                target_pos = np.array([0.4, 0.4, 0.4])
            rand_pos = np.array([uniform(0.3, 0.4) * choice([-1.2, 1.2]) for i in range(2)] + [uniform(0.3, 0.5)])
            target_or = [0,0,0]

            # x,y,z = rand_pos - interface.get_xyz('link1')
            # alpha = -np.arcsin(y / np.sqrt(y**2+z**2)) * np.sign(x)
            # beta = np.arccos(x / np.linalg.norm([x,y,z])) * np.sign(x)
            # target_or = np.array([alpha, beta, uniform(-0.1, 0.1)], dtype=np.float16)
            # print(target_or)
            # quat = transformations.quaternion_from_euler(target_or[0], target_or[1], target_or[2], axes='rxyz')
            # interface.set_mocap_xyz("hand", np.array(rand_pos))
            # interface.set_mocap_orientation("hand", quat)
            target_pos = rand_pos
            for _ in range(30):
                while True:
                    fb = interface.get_feedback()
                    if dual:
                        target = np.hstack([target_pos1, target_or1, target_pos2, target_or2])
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
