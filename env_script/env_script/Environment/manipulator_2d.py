import gym
from gym import core, spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

from .ou_noise import OUNoise



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
    
    def __init__(self, action=None, arm1=1, arm2=1, dt=0.01, tol=0.1):
        self.action_type = action
        # Observation space를 구성하는 state의 최대, 최소를 지정한다.
        self.obs_high = np.array([float('inf'), np.pi]) # x1, y1, x2, y2, xd, yd
        self.obs_low = -self.obs_high

        # Action space를 구성하는 action의 최대, 최소를 지정한다.
        if self.action_type == 'linear':
            self.action_high = np.array([1])
            self.action_low = np.array([-1])
        elif self.action_type == 'angular':
            self.action_high = np.array([np.pi])
            self.action_low = np.array([-np.pi])
        elif self.action_type == 'fused':
            self.action_high = np.array([1, np.pi])
            self.action_low = np.array([-1, -np.pi])
        else:
            print(self.action_type)
            raise ValueError('action type not one of: linear, angular, fused')

        # GYM environment에서 요구하는 변수로, 실제 observation space와 action space를 여기에서 구성한다.
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)
        self.action_space = spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)

        # 로봇암의 요소를 결정하는 변수
        self.link1_len = arm1 # 로봇팔 길이
        self.link2_len = arm2
        self.dt = dt # Timestep
        self.tol = tol # 목표까지 거리

        # 학습 환경에서 사용할 난수 생성에 필요한 seed를 지정한다.
        self.seed()

        self.env_boundary = 5
        self.target_speed = 1.2
        self.episode_length = 1500

        # 변수를 초기화한다.
        self.reset()
        self.n_episodes = 0

        
    def step(self, action, weight=[0,0]):
        self.n_episodes += 1
        #self._move_target()

        if True in np.isnan(action):
            print("ACTION NAN WARNING")
            raise ValueError
        action = np.clip(action, self.action_low, self.action_high)
        

        if self.action_type == 'linear':
            self.robot_tf.transform(
                translation=(action[0]*self.dt, 0)
            )
        elif self.action_type == 'angular':
            self.robot_tf.transform(
                rotation=action[0]*self.dt
            )
        elif self.action_type == 'fused':
            self.robot_tf.transform(
                translation=(action[0]*self.dt, 0),
                rotation=action[1]*self.dt
            )

        #self.joint1_tf.transform(rotation=action[2] * self.dt)
        #self.joint2_tf.transform(rotation=action[3] * self.dt)

        self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
        self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf

        self.t += self.dt

        # Reward와 episode 종료 여부를 확인
        reward, done = self._get_reward()

        # 기타 목적으로 사용할 데이터를 담아둠
        info = {}

        # 시각화 목적으로 사용할 데이터를 self.buffer 에 저장
        self.buffer.append(
            dict(
                robot=self.robot_tf.copy(),
                link1=self.link1_tf_global.copy(),
                link2=self.link2_tf_global.copy(),
                target=self.target_tf.copy(),
                time=self.t,
                actions=action,
                reward=reward,
                weight=weight
            )
        )
        dist, ang = self._get_state()
        if self.n_episodes % 20:
            if len(action) == 1:
                print("dist:\t{0:2.3f}".format(dist),"\tang:\t{0: 2.3f}".format(ang),"\taction:\t[{0: 2.2f}]".format(action[0]),"\treward:\t{0: 2.3f}".format(reward))
            if len(action) == 2:
                print("dist:\t{0:2.3f}".format(dist),"\tang:\t{0: 2.3f}".format(ang),"\taction:\t[{0: 2.2f}\t{1: 2.2f}]".format(action[0],action[1]),"\treward:\t{0: 2.3f}".format(reward))

        # 일반적으로 Gym environment의 step function은 
        # State(observation), 현재 step에서의 reward, episode 종료 여부, 기타 정보로 구성되어있음
        return self._get_state(), reward, done, info


    def reset(self):
        print("\nreseted\n")
        self.n_episodes = 0
        # 매 episode가 시작될때 사용됨.
        # 사용 변수들 초기화
        robot_rot = (random.random()-0.5)*2*3
        self.robot_tf = Transformation(rotation=robot_rot)
        self.joint1_tf = Transformation()
        self.link1_tf = Transformation(translation=(self.link1_len, 0))
        self.joint2_tf = Transformation()
        self.link2_tf = Transformation(translation=(self.link2_len, 0))
        self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
        self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf

        # 목표 지점 생성
        if random.randint(0,1) == 0:
            self.target_tf = Transformation(
                translation=(
                    (random.random()-0.5)*2*(self.env_boundary-0.7),
                    (random.random()-0.5)*2*(self.env_boundary-0.7)
                )
            )
        else:
            x = (random.random()-0.5)*2*(self.env_boundary-0.7)
            y = np.clip(np.tan(robot_rot)*x,-self.env_boundary+0.7,self.env_boundary-0.7)
            self.target_tf = Transformation(translation=(x, y))

        self.ou = OUNoise(dt=self.dt, theta=0.1, sigma=0.2)

        self.done = False
        self.t = 0
        self.buffer = []    # 시각화를 위한 버퍼. episode가 리셋될 때마다 초기화.

        # Step 함수와 다르게 reset함수는 초기 state 값 만을 반환합니다.
        return self._get_state()


    def _move_target(self):
        self.target_tf.transform(
            translation = (self.target_speed * self.dt, 0),
            rotation = self.ou.evolve() * self.dt
        )
        if self.target_tf.x() > self.env_boundary:
            self.target_tf.x(self.env_boundary)
        if self.target_tf.x() < -self.env_boundary:
            self.target_tf.x(-self.env_boundary)
        if self.target_tf.y() > self.env_boundary:
            self.target_tf.y(self.env_boundary)
        if self.target_tf.y() < -self.env_boundary:
            self.target_tf.y(-self.env_boundary)


    def _get_reward(self):
        # 해당 step의 reward를 계산합니다.
        done = False

        mat_target_robot = self.robot_tf.inv()*self.target_tf
        l = np.linalg.norm(mat_target_robot.get_translation())
        
        if self.action_type in ['linear', 'fused']:
            if l < self.tol: 
                reward = 100.
                done = True 
            elif l > self.tol and l < 10:
                reward = -l
        elif self.action_type == 'angular':
            reward = -abs(self._get_state()[1])

        x0, y0 = self.robot_tf.get_translation()
        if abs(x0) > self.env_boundary:
            print("Robot이 Boundary를 벗어남.")
            done = True
            reward = -100
        elif abs(y0) > self.env_boundary:
            print("Robot이 Boundary를 벗어남.")
            done = True
            reward = -100

        if self.n_episodes > self.episode_length:
            done = True

        if done:
            pass
            #self.render()

        return reward, done

    
    def _get_state(self):
        # State(Observation)를 반환합니다.
        mat_target_robot = self.robot_tf.inv()*self.target_tf
        dist = np.linalg.norm(mat_target_robot.get_translation())
        ang = -np.arctan2(mat_target_robot.y(), mat_target_robot.x())
        return np.array([dist, ang])


    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def render(self):
        # Episode 동안의 로봇암 trajectory plot
        buffer = np.array(self.buffer)
        
        # set up figure and animation
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-self.env_boundary, self.env_boundary), ylim=(-self.env_boundary, self.env_boundary))
        ax.grid()

        robot, = ax.plot([], [], 'g', lw=2)
        link1, = ax.plot([], [], 'ko-', lw=2)
        link2, = ax.plot([], [], 'k', lw=2)
        gripper, = ax.plot([], [], 'k', lw=1)
        target, = ax.plot([], [], 'bo', ms=6)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        action_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        weight_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        reward_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)

        robot_geom = np.array(
            [
                [0.3, -0.2, -0.2, 0.3],
                [  0,  0.2, -0.2,   0],
                [  1,    1,    1,   1]
            ]
        )
        link2_geom = np.array(
            [
                [-self.link2_len, -0.1],
                [              0,    0],
                [              1,    1]
            ]
        )
        gripper_geom = np.array(
            [
                [0.1, -0.1, -0.1,  0.1],
                [0.1,  0.1, -0.1, -0.1],
                [   1,    1,   1,    1]
            ]
        )

        def init():
            """initialize animation"""
            robot.set_data([], [])
            link1.set_data([], [])
            link2.set_data([], [])
            gripper.set_data([], [])
            target.set_data([], [])
            time_text.set_text('')
            action_text.set_text('')
            reward_text.set_text('')
            weight_text.set_text('')
            return robot, link1, link2, gripper, target, time_text, reward_text, weight_text, action_text

        def animate(i):
            """perform animation step"""
            robot_points = buffer[i]['robot'] * robot_geom
            link2_points = buffer[i]['link2'] * link2_geom
            gripper_points = buffer[i]['link2'] * gripper_geom

            robot.set_data((robot_points[0, :], robot_points[1, :]))
            link1.set_data((
                [buffer[i]['robot'].x(), buffer[i]['link1'].x()],
                [buffer[i]['robot'].y(), buffer[i]['link1'].y()]
            ))
            link2.set_data((link2_points[0, :], link2_points[1, :]))
            gripper.set_data((gripper_points[0, :], gripper_points[1, :]))
            target.set_data([buffer[i]['target'].x(), buffer[i]['target'].y()])
            time_text.set_text('time = %.1f' % buffer[i]['time'])
            reward_text.set_text('reward = %.3f' % buffer[i]['reward'])
            weight = buffer[i]['weight']
            weight_text.set_text('weight: linear = {0:2.3f}, angular = {1:2.3f}'.format(weight[0], weight[1]))
            action = buffer[i]['actions']
            if len(action) == 1:
                action_text.set_text('action = {0:2.3f}'.format(action[0]))
            elif len(action) == 2:
                action_text.set_text('action = {0: 2.3f}, {1: 2.3f}'.format(action[0],action[1]))
            return robot, link1, link2, gripper, target, time_text, reward_text, weight_text

        interval = self.dt * 1000
        ani = animation.FuncAnimation(fig, animate, frames=len(self.buffer),
                                        interval=interval, blit=True, init_func=init)

        plt.show()



def test(env):
    '''
    Test script for the environment "Manipulator2D"
    '''

    # 환경 초기화
    env.reset()

    # 10초 동안의 움직임을 관찰
    for t in np.arange(0, 10, env.dt):
        # 강화학습이 아닌 위에서 계산한 값을 이용하여 목표 각도에 가까워지도록 피드백 제어

        # position error를 이용해 control input 계산
        link2_to_target = env.link2_tf_global.inv() * env.target_tf.get_translation()
        err1 = env.link2_tf * link2_to_target
        err2 = env.link1_tf * env.joint2_tf * err1
        err3 = env.joint1_tf * err2
        action = [
            np.linalg.norm(err3),
            np.arctan2(err3[1], err3[0]),
            np.arctan2(err2[1], err2[0]),
            np.arctan2(err1[1], err1[0])
        ]

        # Environment의 step 함수를 호출하고, 
        # 변화된 state(observation)과 reward, episode 종료여부, 기타 정보를 가져옴
        next_state, reward, done, info = env.step(action)

        # episode 종료
        if done:
            break

    # Episode 동안의 로봇암 trajectory plot
    env.render()


if __name__=='__main__':

    test(Manipulator2D(tol=0.01))
