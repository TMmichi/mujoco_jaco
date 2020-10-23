import gym
import time
from gym import core, spaces
from gym.utils import seeding
import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import random

#from .ou_noise import OUNoise


class Transformation:
    """
    Transformation class for SE(2)
    """

    def __init__(self, matrix=None, translation=(0, 0), rotation=0):
        if isinstance(matrix, None.__class__):
            self._matrix = self.compute_matrix(translation, rotation)
        else:
            self._matrix = matrix.copy()
    
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            tmp = Transformation()
            tmp._matrix = np.matmul(self._matrix, other._matrix)
            return tmp
        elif isinstance(other, np.ndarray):
            if other.shape==(2,):
                return np.matmul(self._matrix, np.concatenate((other, [1])))[:2]
            else:
                return np.matmul(self._matrix, other)
        else:
            return self._matrix * other

    def __str__(self):
        return "Translation: %s\nRotation: %s\nTransfromation matrix:\n%s"%(
            self.get_translation(), self.get_rotation(), self._matrix
        )

    def transform(self, translation=(0, 0), rotation=0):
        self._matrix = np.matmul(self._matrix, self.compute_matrix(translation, rotation))

    def reset(self):
        self._matrix = self.compute_matrix((0, 0), 0)

    def get_translation(self):
        return self._matrix[0:2, 2]

    def get_rotation(self):
        return self._matrix[0:2, 0:2]

    def get_transformation(self):
        return self._matrix

    def get_direction(self):
        return self._matrix[0, 0:2]
    
    def x(self, x=None):
        if isinstance(x, None.__class__):
            return self._matrix[0, 2]
        else:
            self._matrix[0, 2] = x

    def y(self, y=None):
        if isinstance(y, None.__class__):
            return self._matrix[1, 2]
        else:
            self._matrix[1, 2] = y

    def euler_angle(self, angle=None):
        if isinstance(angle, None.__class__):
            return np.arctan2(self._matrix[1, 0], self._matrix[0, 0])
        elif isinstance(angle, float):
            self._matrix[0:2, 0:2] = self.compute_matrix((0, 0), angle)[0:2, 0:2]

    def inv(self, return_class=True):
        if return_class:
            tmp = Transformation()
            tmp._matrix = np.linalg.inv(self._matrix)
            return tmp
        else:
            return np.linalg.inv(self._matrix)

    def copy(self):
        tmp = Transformation(matrix=self._matrix)
        return tmp

    @staticmethod
    def compute_matrix(translation, rotation):
        c = np.cos(rotation)
        s = np.sin(rotation)
        return np.array(
            [
                [c, -s, translation[0]],
                [s,  c, translation[1]],
                [0,  0,              1]
            ]
        )



class Manipulator2D(gym.Env):
    def __init__(self, action=None, n_robots=1, n_target=1, arm1=1, arm2=1, dt=0.01, tol=0.1, 
                    episode_length=1500, reward_method=None, observation_method='relative', her=False, policy_name='', visualize=False):
        self.env_boundary = 5
        self.action_type = action
        self.observation_method = observation_method
        self.her = her
        self.policy_name = policy_name
        self.visualize = visualize
        self.vis_trigger = False
        self.vis_freq = 10
        
        if self.action_type == 'linear':
            if observation_method == 'absolute':
                self.obs_high = np.array([self.env_boundary, self.env_boundary, np.pi, self.env_boundary, self.env_boundary, np.pi])
                self.obs_low = -self.obs_high
            elif observation_method == 'relative':
                self.obs_high = np.array([float('inf'), np.pi])
                self.obs_low = -self.obs_high
            self.action_high = np.array([1])
            self.action_low = np.array([-1])
        elif self.action_type == 'angular':
            if observation_method == 'absolute':
                self.obs_high = np.array([self.env_boundary, self.env_boundary, np.pi, self.env_boundary, self.env_boundary, np.pi])
                self.obs_low = -self.obs_high
            elif observation_method == 'relative':
                self.obs_high = np.array([float('inf'), np.pi])
                self.obs_low = -self.obs_high
            self.action_high = np.array([np.pi])
            self.action_low = np.array([-np.pi])
        elif self.action_type == 'angular_adj':
            if observation_method == 'absolute':
                self.obs_high = np.array([self.env_boundary, self.env_boundary, np.pi, self.env_boundary, self.env_boundary, np.pi])
                self.obs_low = -self.obs_high
            elif observation_method == 'relative':
                self.obs_high = np.array([float('inf'), np.pi])
                self.obs_low = -self.obs_high
            self.action_high = np.array([np.pi])
            self.action_low = np.array([-np.pi])
        elif self.action_type == 'fused':
            if observation_method == 'absolute':
                self.obs_high = np.array([self.env_boundary, self.env_boundary, np.pi, self.env_boundary, self.env_boundary, np.pi])
                self.obs_low = -self.obs_high
            elif observation_method == 'relative':
                self.obs_high = np.array([float('inf'), np.pi, np.pi])
                self.obs_low = -self.obs_high
            self.action_high = np.array([1, np.pi])
            self.action_low = np.array([-1, -np.pi])
        elif self.action_type == 'pickAndplace':
            # pose of the agent + joint state
            self.obs_high = np.array([self.env_boundary, self.env_boundary, np.pi, np.pi/4, np.pi/4])
            # grasp index
            self.obs_high = np.append(self.obs_high, [n_target])
            # pose of targets
            for _ in range(n_target):
                self.obs_high = np.append(self.obs_high, [self.env_boundary, self.env_boundary, np.pi])
                self.obs_high = np.append(self.obs_high, [self.env_boundary, self.env_boundary, np.pi])
            self.obs_low = -self.obs_high
            self.action_high = np.array([1, np.pi, np.pi/4, np.pi/4])
            self.action_low = -self.action_high
            self.grasp = -1
            self.grasp_dist_threshold = tol
            self.grasp_ang_threshold = tol
        else:
            print(self.action_type)
            raise ValueError('action type not one of: linear, angular, fused, pickAndplace')

        if self.her:
            self.goal_high = np.array([float('inf')])
            self.goal_low = -self.goal_high
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32),
                'achieved_goal': spaces.Box(low=self.goal_low, high = self.goal_high, dtype = np.float32),
                'desired_goal': spaces.Box(low=self.goal_low, high = self.goal_high, dtype = np.float32)
            })
            self.action_space = spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)
        else:
            self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)
            self.action_space = spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)

        self.n_robots = n_robots
        self.n_target = n_target
        self.n_obstacle = 1
        self.link1_len = arm1
        self.link2_len = arm2
        self.dt = dt
        self.tol = tol
        self.target_speed = 0.25
        self.threshold = 0.3
        self.reward_method = reward_method
        self.reward_sign = 'positive'

        self.robot_geom = np.array(
            [
                [0.3, -0.2, -0.2, 0.3],
                [  0,  0.2, -0.2,   0],
                [  1,    1,    1,   1]
            ]
        )
        self.link2_geom = np.array(
            [
                [-self.link2_len, -0.1],
                [              0,    0],
                [              1,    1]
            ]
        )
        self.gripper_geom = np.array(
            [
                [0.1, -0.1, -0.1,  0.1],
                [0.1,  0.1, -0.1, -0.1],
                [   1,    1,   1,    1]
            ]
        )
        self.target_geom = np.array(
            [
                [0.15, -0.1, -0.1, 0.15],
                [  0,   0.1, -0.1,    0],
                [  1,     1,    1,    1]
            ]
        )

        self.top = top = 4.5
        self.bottom = bottom = top - 0.5
        self.right = right = 4.5
        self.left = left = right - 3.5
        self.table_start = np.array(
            [
                [  -right, -right, -left,   -left,  -right],
                [ -bottom,   -top,  -top, -bottom, -bottom],
                [       1,      1,     1,       1,       1]
            ]
        )
        self.table_target = np.array(
            [
                [  right, right, left,   left,  right],
                [ bottom,   top,  top, bottom, bottom],
                [      1,     1,    1,      1,      1]
            ]
        )

        self.seed()
        self.episode_length = episode_length

        self.reset()
        self.n_timesteps = 0
        self.n_episodes = 0
        self.episode_reward = 0
        self.accum_succ = 0
        if self.visualize:
            self.render_init()

        
    def step(self, action, weight=[0,0,0,0], test=False):
        self.n_timesteps += 1

        if True in np.isnan(action):
            print("ACTION NAN WARNING")
            #raise ValueError
        action = np.clip(action, self.action_low, self.action_high)
        
        if self.action_type == 'linear':
            self.robot_tf.transform(
                translation=(action[0]*self.dt, 0),
                rotation=(random.random()-0.5)*0.5*self.dt
            )
            self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
            self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf
            #self._move_object(self.target_tf[0], (random.random()-0.5), (random.random()-0.5)*2)
        elif self.action_type == 'angular':
            self.robot_tf.transform(
                translation=(0.1*self.dt, 0),
                rotation=action[0]*self.dt
            )
            self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
            self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf
            #self._move_object(self.target_tf[0], (random.random()-0.5)*0.1, (random.random()-0.5)*2)
        elif self.action_type == 'angular_adj':
            self.robot_tf.transform(
                rotation=action[0]*self.dt
            )
            self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
            self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf
            #self._move_object(self.target_tf[0], (random.random()-0.5)*0.1, (random.random()-0.5)*2)
        elif self.action_type == 'fused':
            self.robot_tf.transform(
                translation=(action[0]*self.dt, 0),
                rotation=action[1]*self.dt
            )
            self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
            self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf
            #self._move_object(self.target_tf[0], (random.random()-0.5)*2, (random.random()-0.5)*2)
        elif self.action_type == 'pickAndplace':
            self.robot_tf.transform(
                translation=(action[0]*self.dt, 0),
                rotation=action[1]*self.dt
            )

            if self.joint1_tf.euler_angle() < -np.pi/4 or self.joint1_tf.euler_angle() > np.pi/4:
                action[2] = 0
            if self.joint2_tf.euler_angle() < -np.pi/4 or self.joint2_tf.euler_angle() > np.pi/4:
                action[3] = 0 
            self.joint1_tf.transform(rotation=action[2] * self.dt)
            self.joint2_tf.transform(rotation=action[3] * self.dt)

            self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
            self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf

            if self.grasp > -1:
                (x,y) = self.link2_tf_global.get_translation()
                ang = self.link2_tf_global.euler_angle()
                self.target_tf[self.grasp].x(x)
                self.target_tf[self.grasp].y(y)
                self.target_tf[self.grasp].euler_angle(ang)

        self.t += self.dt

        reward, done = self._get_reward()
        self.episode_reward += reward

        info = {}

        obs = self._get_state()
        if self.her:
            obs['achieved_goal'] = [reward]
            if self.reward_method == 'sparse':
                obs['desired_goal'] = [0]
            else:
                obs['achieved_goal'] = [1]

        if test or self.visualize:
            if self.n_timesteps % self.vis_freq == 0:
                target_tf = []
                for i in range(self.n_target):
                    target_tf.append(self.target_tf[i].copy())
                if self.action_type == 'pickAndplace':
                    target_place_tf = []
                    for i in range(self.n_target):
                        target_place_tf.append(self.target_place_tf[i].copy())
                else:
                    target_place_tf = []
                self.buffer.append(
                    dict(
                        robot=self.robot_tf.copy(),
                        link1=self.link1_tf_global.copy(),
                        link2=self.link2_tf_global.copy(),
                        target=target_tf,
                        target_place_tf=target_place_tf,
                        time=self.t,
                        observations=obs,
                        actions=action,
                        reward=reward,
                        total_reward=self.episode_reward,
                        weight=weight
                    )
                )
                try:
                    self.realtime_render(self.buffer)
                    self.update_render()
                    self.buffer = []
                except Exception:
                    time.sleep(1)
                    self.render_init()
                    done = True
            
        return obs, reward, done, info

    def reset(self):
        print("  "+self.policy_name+" reset")
        self.n_timesteps = 0
        self.episode_reward = 0
        self.grasp = -1

        robot_rot = (random.random()-0.5)*2*3
        robot_x = (random.random()-0.5)*2*(self.env_boundary-2)
        robot_y = (random.random()-0.5)*2*(self.env_boundary-2)
        self.robot_tf = Transformation(
            translation=(robot_x, robot_y),
            rotation=robot_rot)
        self.joint1_tf = Transformation()
        self.link1_tf = Transformation(translation=(self.link1_len, 0))
        self.joint2_tf = Transformation()
        self.link2_tf = Transformation(translation=(self.link2_len, 0))
        self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
        self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf

        target_rot = (random.random()-0.5)*2*3
        if self.action_type == 'pickAndplace':
            self.target_tf = []
            self.target_place_tf = []
            for i in range(self.n_target):
                target_rot = (random.random()-0.5)*2*3
                x = -self.left-0.7*i-0.6
                y = -4.27
                target_tf = Transformation(translation=(x, y), rotation=target_rot)
                self.target_tf.append(target_tf)
                target_place_rot = (random.random()-0.5)*2*3
                x = self.left+0.7*i+0.7
                y = 4.23
                target_place_tf = Transformation(translation=(x, y), rotation=target_place_rot)
                self.target_place_tf.append(target_place_tf)
        elif self.action_type == 'angular_adj':
            self.target_tf = [0]*self.n_target
            for i in range(self.n_target):
                target_rot = (random.random()-0.5)*2*3
                target_tf = Transformation(translation=(
                    robot_x+(random.random()-0.5)*0.05, 
                    robot_y+(random.random()-0.5)*0.05),
                    rotation=target_rot)
                self.target_tf[i] = target_tf
        else:
            choice = random.randint(0,4)
            if choice >= 1:
                self.target_tf = [0]*self.n_target
                for i in range(self.n_target):
                    target_tf = Transformation(
                        translation=(
                            (random.random()-0.5)*2*(self.env_boundary-1.5),
                            (random.random()-0.5)*2*(self.env_boundary-1.5)
                        ),
                        rotation=target_rot
                    )
                    self.target_tf[i] = target_tf
            elif choice == 0:
                self.target_tf = [0]*self.n_target
                for i in range(self.n_target):
                    x = (random.random()-0.5)*2*(self.env_boundary-1.5)
                    y = np.clip(np.tan(robot_rot)*x,-self.env_boundary+1,self.env_boundary-1)
                    target_tf = Transformation(translation=(x, y), rotation=target_rot)
                    self.target_tf[i] = target_tf

        self.done = False
        self.t = 0
        self.buffer = []

        return self._get_state()

    def _move_object(self, object_tf, speed, rotation):
        object_tf.transform(
            translation = (speed * self.dt, 0),
            rotation = rotation * self.dt
        )
        if object_tf.x() > self.env_boundary:
            object_tf.x(self.env_boundary)
        if object_tf.x() < -self.env_boundary:
            object_tf.x(-self.env_boundary)
        if object_tf.y() > self.env_boundary:
            object_tf.y(self.env_boundary)
        if object_tf.y() < -self.env_boundary:
            object_tf.y(-self.env_boundary)

    def _get_reward(self):
        done = False
        
        if self.action_type == 'linear':
            mat_target_robot = self.robot_tf.inv()*self.target_tf[0]
            target_vector = np.dot(mat_target_robot.get_translation(), np.array([1,0])) * np.array([1,0])
            l = np.linalg.norm(target_vector)
            if self.reward_method == 'sparse':
                if l < self.tol:
                    reward = 1
                    done = True 
                    self.accum_succ += 1
                    print("\033[92m  SUCCEEDED\033[0m")
                else:
                    reward = 0
            else:
                if self.reward_sign == 'negative':
                    if l < self.tol: 
                        reward = 100
                        done = True 
                        self.accum_succ += 1
                        print("\033[92m  SUCCEEDED\033[0m")
                    else:
                        reward = -l * 0.5
                else:
                    if l < self.tol:
                        reward = 100
                        done = True 
                        self.accum_succ += 1
                        print("\033[92m  SUCCEEDED\033[0m")
                    else:
                        reward = np.exp(-(l-self.tol)**0.25)*0.1
        elif self.action_type == 'angular':
            mat_target_robot = self.robot_tf.inv()*self.target_tf[0]
            ang = -np.arctan2(mat_target_robot.y(), mat_target_robot.x())
            if self.reward_method == 'sparse':
                if abs(ang) < self.tol:
                    reward = 1
                    done = True
                    self.accum_succ += 1
                    print("\033[92m  SUCCEEDED\033[0m")
                else:
                    reward = 0
            else:
                if abs(ang) < self.tol:
                    reward = 100
                    done = True
                    self.accum_succ += 1
                    print("\033[92m  SUCCEEDED\033[0m")
                else:
                    reward = -abs(ang/np.pi)
        elif self.action_type == 'angular_adj':
            a_robot = self.robot_tf.euler_angle()
            a_target = self.target_tf[0].euler_angle()
            if a_robot * a_target > 0:
                angle_diff = abs(a_robot - a_target)
            else:
                angle_diff = abs(a_robot) + abs(a_target)
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
            if self.reward_method == 'sparse':
                if abs(angle_diff) < self.tol:
                    reward = 1
                    done = True
                    self.accum_succ += 1
                    print("\033[92m  SUCCEEDED\033[0m")
                else:
                    reward = 0
            else:
                pass
        elif self.action_type == 'fused':
            mat_target_robot = self.robot_tf.inv()*self.target_tf[0]
            l = np.linalg.norm(mat_target_robot.get_translation())
            a_robot = self.robot_tf.euler_angle()
            a_target = self.target_tf[0].euler_angle()
            if a_robot * a_target > 0:
                angle_diff = abs(a_robot - a_target)
            else:
                angle_diff = abs(a_robot) + abs(a_target)
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
            if self.reward_method == 'sparse':
                if l < self.tol and angle_diff < self.tol:
                    reward = 1
                    done = True
                    self.accum_succ += 1
                    print("\033[92m  SUCCEEDED\033[0m")
                else:
                    reward = 0
            else:
                if self.reward_sign == 'negative':
                    if l < self.tol and angle_diff < self.tol:
                        reward = 100
                        done = True
                        self.accum_succ += 1
                        print("\033[92m  SUCCEEDED\033[0m")
                    elif l >= 1:
                        reward = (-l - 2) * 0.125
                    else:
                        reward = (-l - 1 - angle_diff/np.pi ) * 0.125
                else:
                    if l < self.tol and angle_diff < self.tol:
                        reward = 100
                        done = True
                        self.accum_succ += 1
                        print("\033[92m  SUCCEEDED\033[0m")      
                    else:
                        #reward = np.exp(-(l-self.tol)**2*0.25)*0.01 + np.exp(-(angle_diff/np.pi/max(l,1))**2*0.25)*0.01
                        reward = 1/(l+0.5) * 0.002 + (np.pi+2)/(angle_diff+1) * 0.001
        elif self.action_type == 'pickAndplace':
            if self.crash_check():
                reward = -100
                done = True
            else:
                if self.grasp == -1:
                    if self.grasp_check():
                        reward = 100
                    else:
                        if self.reward_method == 'target':
                            min_l = 1000000
                            for index, target_tf in enumerate(self.target_tf):
                                mat_target_gripper = self.link2_tf_global.inv()*target_tf
                                l = np.linalg.norm(mat_target_gripper.get_translation())
                                min_l = min(l, min_l)
                                if min_l == l:
                                    min_l_index = index
                            a_gripper = self.link2_tf_global.euler_angle()
                            a_target = self.target_tf[min_l_index].euler_angle()
                            if a_gripper * a_target > 0:
                                angle_diff = abs(a_gripper - a_target)
                            else:
                                angle_diff = abs(a_gripper) + abs(a_target)
                                if angle_diff > np.pi:
                                    angle_diff = 2*np.pi - angle_diff
                            if l >= 1:
                                reward = -(l+2)
                            else:
                                reward = -(l+1+angle_diff/np.pi)
                else:
                    if self.place_check():
                        reward = 100
                        done = True
                    else:
                        if self.reward_method == 'target':
                            mat_target_place_gripper = self.link2_tf_global.inv()*self.target_place_tf[self.grasp]
                            l = np.linalg.norm(mat_target_place_gripper.get_translation())
                            a_gripper = self.link2_tf_global.euler_angle()
                            a_target = self.target_place_tf[self.grasp].euler_angle()
                            if a_gripper * a_target > 0:
                                angle_diff = abs(a_gripper - a_target)
                            else:
                                angle_diff = abs(a_gripper) + abs(a_target)
                                if angle_diff > np.pi:
                                    angle_diff = 2*np.pi - angle_diff
                            if l >= 1:
                                reward = -(l+2)
                            else:
                                reward = -(l+1+angle_diff/np.pi)
            if self.reward_method == 'time':
                reward -= 0.01
            
        x0, y0 = self.robot_tf.get_translation()
        if abs(x0) > self.env_boundary or abs(y0) > self.env_boundary:
            print("\033[91m  ROBOT Out of Boundary\033[0m")
            done = True
            if self.reward_method == 'sparse':
                reward = -1
            else:
                reward = -100

        if self.n_timesteps > self.episode_length:
            print("\033[91m  TIMES UP\033[0m")
            done = True

        if done:
            self.n_episodes += 1
            print("Num episodes: ",self.n_episodes,"\tSuccess rate: {0:3.2f}%".format(self.accum_succ/self.n_episodes*100))

        return reward, done
    
    def crash_check(self):
        x = self.robot_tf.get_translation()[0]
        y = self.robot_tf.get_translation()[1]
        if self.left - self.threshold < x < self.right + self.threshold:
            if self.bottom - self.threshold < y < self.top + self.threshold:
                print("\033[91m  CRASHED\033[0m")
                return True
        elif -self.right - self.threshold < x < -self.left + self.threshold:
            if -self.top - self.threshold < y < -self.bottom + self.threshold:
                print("\033[91m  CRASHED\033[0m")
                return True
        return False

    def grasp_check(self):
        pos = self.link2_tf_global.get_translation()
        ang = self.link2_tf_global.euler_angle()
        for index, target_tf in enumerate(self.target_tf):
            if np.linalg.norm(pos-target_tf.get_translation()) < self.grasp_dist_threshold:
                if abs(ang - target_tf.euler_angle()) < self.grasp_ang_threshold:
                    self.grasp = index
                    print("\033[92m  GRASPED\033[0m")
                    return True
        else:
            return False
    
    def place_check(self):
        pos = self.link2_tf_global.get_translation()
        ang = self.link2_tf_global.euler_angle()
        target_place_tf = self.target_place_tf[self.grasp]
        if np.linalg.norm(pos-target_place_tf.get_translation()) < self.grasp_dist_threshold:
            if abs(ang - target_place_tf.euler_angle()) < self.grasp_ang_threshold:
                self.grasp = -1
                print("\033[92m  PLACED\033[0m")
                return True
        return False

    def _get_state(self):
        # State(Observation)를 반환합니다.
        mat_target_robot = self.robot_tf.inv()*self.target_tf[0]
        dist = np.linalg.norm(mat_target_robot.get_translation())
        ang = -np.arctan2(mat_target_robot.y(), mat_target_robot.x())
        a_robot = self.robot_tf.euler_angle()
        a_target = self.target_tf[0].euler_angle()
        angle_diff = a_robot - a_target
        if angle_diff > np.pi:
            angle_diff = 2*np.pi - angle_diff
        elif angle_diff < -np.pi:
            angle_diff = 2*np.pi + angle_diff

        if self.her:
            if self.action_type in ['linear', 'angular', 'fused']:
                if self.observation_method == 'absolute':
                    state = self.robot_tf.get_translation()
                    state = np.append(state, self.robot_tf.euler_angle())
                    state = np.append(state, self.target_tf[0].get_translation())
                    state = np.append(state, self.target_tf[0].euler_angle())
                    return OrderedDict([
                                ('observation', state),
                                ('achieved_goal', [0]),
                                ('desired_goal', [1])])
                elif self.observation_method == 'relative':
                    if self.action_type in ['linear', 'angular']:
                        return OrderedDict([
                                ('observation', [dist, ang]),
                                ('achieved_goal', [0]),
                                ('desired_goal', [1])])
                    else:
                        return OrderedDict([
                                ('observation', [dist, ang]),
                                ('achieved_goal', [0]),
                                ('desired_goal', [1])])
            elif self.action_type == 'pickAndplace':
                state = self.robot_tf.get_translation()
                state = np.append(state, self.robot_tf.euler_angle())
                state = np.append(state, self.grasp)
                state = np.append(state, [self.joint1_tf.euler_angle()])
                state = np.append(state, [self.joint2_tf.euler_angle()])
                for target_tf in self.target_tf:
                    state = np.append(state, target_tf.get_translation())
                    state = np.append(state, target_tf.euler_angle())
                for target_place_tf in self.target_place_tf:
                    state = np.append(state, target_place_tf.get_translation())
                    state = np.append(state, target_place_tf.euler_angle())
                return OrderedDict([
                        ('observation', state),
                        ('achieved_goal', [0]),
                        ('desired_goal', [1])])
        else:
            if self.action_type in ['linear', 'angular', 'angular_adj', 'fused']:
                if self.observation_method == 'absolute':
                    state = self.robot_tf.get_translation()
                    state = np.append(state, self.robot_tf.euler_angle())
                    state = np.append(state, self.target_tf[0].get_translation())
                    state = np.append(state, self.target_tf[0].euler_angle())
                    return state
                elif self.observation_method == 'relative':
                    if self.action_type in ['linear', 'angular']:
                        return np.array([dist, ang])
                    else:
                        return np.array([dist, ang, angle_diff])
            elif self.action_type == 'pickAndplace':
                state = self.robot_tf.get_translation()
                state = np.append(state, self.robot_tf.euler_angle())
                state = np.append(state, self.grasp)
                state = np.append(state, [self.joint1_tf.euler_angle()])
                state = np.append(state, [self.joint2_tf.euler_angle()])
                for target_tf in self.target_tf:
                    state = np.append(state, target_tf.get_translation())
                    state = np.append(state, target_tf.euler_angle())
                for target_place_tf in self.target_place_tf:
                    state = np.append(state, target_place_tf.get_translation())
                    state = np.append(state, target_place_tf.euler_angle())
                return state

    def compute_reward(self):
        pass

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calculate_desired_action(self, obs):
        robot_tf = Transformation(translation=obs[:2],rotation=obs[2])
        target_tf = Transformation(translation=obs[3:5],rotation=obs[5])
        mat_target_robot = robot_tf.inv()*target_tf
        if self.action_type == 'linear':
            target_vector = np.dot(mat_target_robot.get_translation(), np.array([1,0])) * np.array([1,0])
            if target_vector[0] > 0:
                del robot_tf, target_tf
                return [np.clip(0.8 + np.random.normal()*0.05, -0.99, 0.99)]
            else:
                del robot_tf, target_tf
                return [np.clip(-0.8 + np.random.normal()*0.05, -0.99, 0.99)]
        elif self.action_type == 'angular':
            ang = -np.arctan2(mat_target_robot.y(), mat_target_robot.x())
            if ang > 0:
                return [np.clip(-2 + np.random.normal()*0.05, -np.pi, np.pi)]
            else:
                return [np.clip(2 + np.random.normal()*0.05, -np.pi, np.pi)]
        elif self.action_type == 'angular_adj':
            a_robot = self.robot_tf.euler_angle()
            a_target = self.target_tf[0].euler_angle()
            if a_robot * a_target > 0:
                angle_diff = a_robot - a_target
            else:
                if a_robot > a_target:
                    angle_diff = abs(a_robot) + abs(a_target)
                else:
                    angle_diff = -abs(a_robot) - abs(a_target)
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                elif angle_diff < -np.pi:
                    angle_diff = 2*np.pi + angle_diff
            if angle_diff > 0:
                return [np.clip(-2 + np.random.normal()*0.05, -np.pi, np.pi)]
            else:
                return [np.clip(2 + np.random.normal()*0.05, -np.pi, np.pi)]


    def render_init(self):        
        # set up figure and animation
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-self.env_boundary, self.env_boundary), ylim=(-self.env_boundary, self.env_boundary))

        self.graphic_robot, = self.ax.plot([], [], 'g', lw=1)
        self.graphic_robot_body, = self.ax.plot([], [], 'go-', fillstyle='none', ms=20)
        self.graphic_table_start, = self.ax.plot([], [], 'k-', lw=1)
        self.graphic_table_target, = self.ax.plot([], [], 'k-', lw=1)
        self.graphic_link1, = self.ax.plot([], [], 'ko-', lw=1, ms=2)
        self.graphic_link2, = self.ax.plot([], [], 'k', lw=1)
        self.graphic_gripper, = self.ax.plot([], [], 'k', lw=1)
        self.graphic_target_list = []
        for _ in range(self.n_target):
            target, = self.ax.plot([], [], 'b', lw=1)
            self.graphic_target_list.append(target)
        self.graphic_target_place_list = []
        for _ in range(self.n_target):
            target_place, = self.ax.plot([], [], 'g--', lw=1)
            self.graphic_target_place_list.append(target_place)
        self.graphic_time_text = self.ax.text(0.98, 0.21, '', transform=self.ax.transAxes, ha='right')
        self.graphic_reward_text = self.ax.text(0.98, 0.16, '', transform=self.ax.transAxes, ha='right')
        self.graphic_observation_text = self.ax.text(0.98, 0.11, '', transform=self.ax.transAxes, ha='right')
        self.graphic_action_text = self.ax.text(0.98, 0.06, '', transform=self.ax.transAxes, ha='right')
        self.graphic_weight_text = self.ax.text(0.98, 0.01, '', transform=self.ax.transAxes, ha='right')
        self.graphic_table_start_text = self.ax.text(0.03, 0.12, '', transform=self.ax.transAxes)
        self.graphic_table_target_text = self.ax.text(0.89, 0.80, '', transform=self.ax.transAxes)
        
        self.graphic_robot.set_data([], [])
        self.graphic_robot_body.set_data([], [])
        self.graphic_table_start.set_data([], [])
        self.graphic_table_target.set_data([], [])
        self.graphic_link1.set_data([], [])
        self.graphic_link2.set_data([], [])
        self.graphic_gripper.set_data([], [])
        for target in self.graphic_target_list:
            target.set_data([], [])
        for target_place in self.graphic_target_place_list:
            target_place.set_data([], [])
        self.graphic_time_text.set_text('')
        self.graphic_observation_text.set_text('')
        self.graphic_action_text.set_text('')
        self.graphic_reward_text.set_text('')
        self.graphic_weight_text.set_text('')
        self.graphic_table_start_text.set_text('table\ntarget')
        self.graphic_table_target_text.set_text('table\ngoal')

        self.ax.grid()

    def realtime_render(self, buffer):
        """perform animation step"""
        robot_points = buffer[0]['robot'] * self.robot_geom
        link2_points = buffer[0]['link2'] * self.link2_geom
        gripper_points = buffer[0]['link2'] * self.gripper_geom
        target_points_list = []
        for target_tf in buffer[0]['target']:
            target_points = target_tf * self.target_geom
            target_points_list.append(target_points)
        target_place_points_list = []
        for target_place_tf in buffer[0]['target_place_tf']:
            target_place_points = target_place_tf * self.target_geom
            target_place_points_list.append(target_place_points)

        self.graphic_robot.set_data((robot_points[0, :], robot_points[1, :]))
        fig_size = self.fig.get_size_inches()*self.fig.dpi
        self.graphic_robot_body._markersize = int(min(fig_size[0],fig_size[1])/30)
        self.graphic_robot_body.set_data((buffer[0]['robot'].get_translation()[0], buffer[0]['robot'].get_translation()[1]))
        self.graphic_table_start.set_data((self.table_start[0, :], self.table_start[1, :]))
        self.graphic_table_target.set_data((self.table_target[0, :], self.table_target[1, :]))
        self.graphic_link1.set_data((
            [buffer[0]['robot'].x(), buffer[0]['link1'].x()],
            [buffer[0]['robot'].y(), buffer[0]['link1'].y()]
        ))
        self.graphic_link2.set_data((link2_points[0, :], link2_points[1, :]))
        self.graphic_gripper.set_data((gripper_points[0, :], gripper_points[1, :]))
        for target, points in zip(self.graphic_target_list, target_points_list):
            target.set_data((points[0, :], points[1, :]))
        for target_place, points in zip(self.graphic_target_place_list, target_place_points_list):
            target_place.set_data((points[0, :], points[1, :]))
        self.graphic_time_text.set_text('time = %.1f' % buffer[0]['time'])
        self.graphic_reward_text.set_text('reward = {0: 1.3f}, {1: 1.3f}'.format(buffer[0]['reward'], buffer[0].get('total_reward',0)))
        weight = buffer[0]['weight']
        self.graphic_weight_text.set_text('weight: [{0: 2.2f}, {1: 2.2f}, {2: 2.2f}, {3: 2.2f}]'.format(weight[0], weight[1], weight[2], weight[3]))
        action = buffer[0]['actions']
        action_string = 'act: ['
        for index in range(len(action)-1):
            action_string += '{0: 1.2f}, '.format(action[index])
        action_string += '{0: 1.2f}]'.format(action[-1])
        self.graphic_action_text.set_text(action_string)
        obs = buffer[0]['observations']
        obs_string = 'obs: ['
        for index in range(len(obs)-1):
            obs_string += '{0: 2.2f}, '.format(obs[index])
        obs_string += '{0: 2.2f}]'.format(obs[-1])
        self.graphic_observation_text.set_text(obs_string)
        
        self.graphic_table_start_text.set_text('table\ntarget')
        self.graphic_table_target_text.set_text('table\ngoal')
    
    def update_render(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def test(env):
    '''
    Test script for the environment "Manipulator2D"
    '''
    for _ in range(10):
        env.reset()

        for _ in np.arange(0, 10, env.dt):

            link2_to_target = env.link2_tf_global.inv() * env.target_tf[0].get_translation()
            err1 = env.link2_tf * link2_to_target
            err2 = env.link1_tf * env.joint2_tf * err1
            err3 = env.joint1_tf * err2
            action = [
                np.linalg.norm(err3),
                np.arctan2(err3[1], err3[0]),
                np.arctan2(err2[1], err2[0]),
                np.arctan2(err1[1], err1[0])
            ]

            _, _, done, _ = env.step(action, test=True)

            if done:
                break

        env.render()


if __name__=='__main__':

    test(Manipulator2D(tol=0.01, action='pickAndplace', reward_method='target', n_target=3))
