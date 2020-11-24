#!/usr/bin/env python

import os
import sys
import time
from pathlib import Path
from math import pi
from random import sample, randint, uniform

import scipy.interpolate

import numpy as np
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
        robot_file = kwargs.get('robot_file', None)
        if robot_file == None:
            n_robot_postfix = ['', '_dual', '_tri']    
            try:
                xml_name = 'jaco2'+n_robot_postfix[self.n_robots-1]
            except Exception:
                raise NotImplementedError("\n\t\033[91m[ERROR]: xml_file of the given number of robots doesn't exist\033[0m")
        else:
            xml_name = kwargs['robot_file']
        
        self.jaco = MujocoConfig(xml_name, n_robots=self.n_robots)
        self.interface = Mujoco(self.jaco, dt=0.005, visualize=False, create_offscreen_rendercontext=False)
        self.interface.connect()
        self.gripper_angle_1 = 0
        self.gripper_angle_2 = 0
        self.ctrl_type = self.jaco.ctrl_type
        self.task = kwargs.get('task', None)
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
            #self.ctr = OSC(self.jaco, kp=50, ko=180, kv=20, vmax=[0.2, 0.5236], ctrlr_dof=ctrl_dof)
            self.ctr = OSC(self.jaco, kp=50, ko=180, kv=20, vmax=[0.3, 0.7854], ctrlr_dof=ctrl_dof)
            self.target_pos = self._reset()
        else:
            _ = self._reset()

        ### ------------  REWARD  ------------ ###
        self.base_position = self.__get_property('link1', 'position')
        self.reward_method = kwargs.get('reward_method', None)
        self.reward_module = kwargs.get('reward_module', None)


    def _step_simulation(self):
        fb = self.interface.get_feedback()
        if self.controller:
            u = self.__controller_generate(fb)
            self.interface.send_forces(
                np.hstack([u, [self.gripper_angle_1, self.gripper_angle_1, self.gripper_angle_2]])
            )
        else:
            if self.ctrl_type == "torque":
                self.interface.send_signal(np.hstack([self.target_signal, [0, 0, 0]]))
            elif self.ctrl_type == "velocity":
                self.interface.send_signal(np.hstack([fb['q'], self.target_signal, [0, 0, 0]]))
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
        if target_angle == None:
            if self.task in ['carrying', 'placing', 'releasing']:
                random_init_angle = []
            elif self.task in ['grasping', 'pushing']:
                random_init_angle = [uniform_np(pi/4, 3*pi/4), 3.75, uniform_np(
                        2, 2.5), uniform_np(0, 0.1), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3)]
                random_init_angle *= self.n_robots
            else:
                random_init_angle = [uniform_np(-pi/2, pi/2), 3.75, uniform_np(
                        1.5, 2.5), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3), uniform_np(0.8, 2.3)]
                random_init_angle *= self.n_robots
        else:
            random_init_angle = target_angle
        self.interface.set_joint_state(random_init_angle, [0]*6*self.n_robots)
        for _ in range(3):
            fb = self.interface.get_feedback()
            self.target_pos = np.hstack([[-0.25, 0, -0.15, 0, 0, 0]*self.n_robots])
            _ = self.__controller_generate(fb)
            self.interface.send_forces([0]*9*self.n_robots)

        self.reaching_goal, self.obj_goal, self.dest_goal = self.__sample_goal()
        if self.task in ['grasping', 'pushing']:
            self.grasp_succeed_iter = 0
            self.target_pos = np.reshape(np.hstack([self.obj_goal,np.repeat([[0,0,0]],self.n_robots,axis=0)]),(-1))
            while True:
                self._step_simulation()
                self.__get_gripper_pose()
                if np.linalg.norm(self.gripper_pose[0][:3] - self.obj_goal[0]) < 0.2:
                    break
        if self.task is 'reaching':
            self.target_pos = np.hstack([self.dest_goal,np.repeat([[0,0,0]],self.n_robots,axis=0)])
            while True:
                self._step_simulation()
                self.__get_gripper_pose()
                if np.norm(self.gripper_pose[0] - self.obj_goal[0]) < 0.2:
                    break
        obs = self._get_observation()
        return obs[0]

    def _get_observation(self):
        if self.state_gen == None:
            self.__get_gripper_pose()
            observation = []
            for i in range(self.n_robots):
                # Observation dimensions: 6, 2, 3, 3, 3
                # [absolute gripper_pose, gripper angle, obj position, dest position, reaching target]
                observation.append(np.hstack([self.gripper_pose[i], [self.gripper_angle_1, self.gripper_angle_2], self.obj_goal[i], self.dest_goal[i], self.reaching_goal[i]]))
        else:
            image, depth = self._get_camera()
            data = [image, depth]
            observation = self.state_gen.generate(data)
        return np.array(observation)
    
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
        # TODO: Reward from IRL
        if self.reward_method is None:
            if self.task == 'reaching':
                dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.reaching_goal[0])
                if dist_diff > 0.5:
                    return 0
                else:
                    reward = ((3 - dist_diff * 1.3)) * 0.1
                return reward
            elif self.task == 'grasping':
                obj_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.obj_goal[0])
                reward = ((3 - obj_diff * 1.3)) * 0.01
                reward += (self.interface.get_xyz('object_body')[2] - 0.44) * 10
                return reward
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

    def __sample_goal(self):
        reach_goal = []
        obj_goal = []
        dest_goal = []
        for _ in range(self.n_robots):
            reach_goal_pos = [uniform(0.25, 0.35) * sample([-1, 1], 1)[0]
                          for i in range(2)] + [uniform(0.1, 0.4)]
            reach_goal.append(reach_goal_pos)
            obj_goal_pos = [0,0.65,0.5]
            obj_goal.append(obj_goal_pos)
            dest_goal_pos = [0.5,0.65,1]
            dest_goal.append(dest_goal_pos)
            self.interface.set_mocap_xyz("target", reach_goal_pos[:3])
            self.interface.set_mocap_xyz("object_body", obj_goal_pos[:3])
            # TODO: set_xyz object_holder, object_body, dest_holder
        return np.array(reach_goal), np.array(obj_goal), np.array(dest_goal)

    def _get_terminal_inspection(self):
        self.num_episodes += 1
        dist_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.reaching_goal[0])
        obj_diff = np.linalg.norm(self.gripper_pose[0][:3] - self.obj_goal[0])
        wb = np.linalg.norm(self.__get_property('EE', 'position')[0] - self.base_position[0])
        if pi - 0.1 < self.interface.get_feedback()['q'][2] < pi + 0.1:
            print("\033[91m \nUn wanted joint angle - possible singular state \033[0m")
            return True, -5, wb
        else:
            if wb > 0.9:
                print("\033[91m \nWorkspace out of bound \033[0m")
                return True, -5, wb
            else:
                if self.task == 'reaching':
                    if dist_diff < 0.15:  # TODO: Shape terminal inspection
                        print("\033[92m Target Reached \033[0m")
                        return True, 200 - (self.num_episodes*0.1), wb
                    else:
                        return False, 0, wb
                elif self.task == 'grasping':
                    if obj_diff > 0.3:
                        print("\033[91m \nGripper too far away from the object \033[0m")
                        return True, -5, wb
                    if self.interface.get_xyz('object_body')[2] > 0.46:
                        self.grasp_succeed_iter += 1
                    if self.grasp_succeed_iter > 200:
                        print("\033[92m Grasping Succeeded \033[0m")
                        return True, 200 - (self.num_episodes*0.1), wb
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
        _ = self.__get_gripper_pose()
        if self.controller:
            # Action: Gripper Pose Increments (m,rad)
            # Action scaled to 0.01m, 0.05 rad
            self.target_pos = self.gripper_pose[0] + np.hstack([a[:3]/100, a[3:6]/20])
            if len(a) > 6:
                self.gripper_angle_1 += a[6]
                self.gripper_angle_2 += a[7]
                self.gripper_angle_1 = max(min(self.gripper_angle_1,10),0)
                self.gripper_angle_2 = max(min(self.gripper_angle_2,10),0)
            else:
                self.gripper_angle_1 = 0
                self.gripper_angle_2 = 0
            self.interface.set_mocap_xyz("hand", self.target_pos[:3])
            self.interface.set_mocap_orientation("hand", transformations.quaternion_from_euler(
                self.target_pos[3], self.target_pos[4], self.target_pos[5], axes="rxyz"))
        else:
            # If Position: Joint Angle Increments (rad)
            # If Velocity: Joint Velocity (rad/s)
            # If Torque: Joint Torque (Nm)
            self.target_signal = a

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

    def _get_pressure(self):
        pass



if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../abr_control')))
    from assets.mujoco_config import MujocoConfig
    from mujoco import Mujoco
    from abr_control.controllers import OSC
    from abr_control.utils import transformations

    mobile = True
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
                target_pos = interface.get_xyz('EE')
            target_or = np.array([0, 0, 0], dtype=np.float16)
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
                            print("Reached")
                            break
                if dual:
                    target_pos1 += np.array([0.01, 0.01, 0.01])
                    target_pos2 += np.array([0.01, 0.01, 0.01])
                else:
                    target_pos += np.array([0.03, 0.03, 0.03])
                target_or += np.array([0.1, 0.1, 0.1])
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
