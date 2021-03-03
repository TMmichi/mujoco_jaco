#!/usr/bin/env python

import os, sys
import time
import path_config
from pathlib import Path
from configuration import model_configuration, pretrain_configuration, info, total_time_step
try:
    import spacenav, atexit
except Exception:
    pass

import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac import SAC
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_hpcsac
from stable_baselines.hpcppo import HPCPPO
from stable_baselines.hpcppo.policies import MlpPolicy as MlpPolicy_hpcppo
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from env_script.env_mujoco import JacoMujocoEnv
from state_gen.state_generator import State_generator

from argparser import ArgParser

default_lr = model_configuration['learning_rate']
def _lr_scheduler(frac):
    return default_lr * frac

class RL_controller:
    def __init__(self):
        # Arguments
        parser = ArgParser(isbaseline=True)
        args = parser.parse_args()

        # Debug
        args.debug = False
        print("DEBUG = ", args.debug)

        self.sess_SRL = tf_util.single_threaded_session()
        args.sess = self.sess_SRL
        args.visualize = True

        # State Generation Module defined here
        # self.stateGen = State_generator(**vars(args))
        # args.stateGen = self.stateGen

        # Reward Generation
        self.reward_method = None
        self.reward_module = None
        args.reward_method = self.reward_method
        args.reward_module = self.reward_module

        # Action
        self.g_angle = 0
        self.g_changed = None
        self.pressed = {0:False, 1:False}        # 0:Left - Open, 1:Right - Close

        # If resume training on pre-trained models with episodes, else None
        package_path = str(Path(__file__).resolve().parent.parent)
        self.model_path = package_path+"/models_baseline/"
        os.makedirs(self.model_path, exist_ok=True)
        
        self.steps_per_batch = 100
        self.batches_per_episodes = 5
        args.steps_per_batch = self.steps_per_batch
        args.batches_per_episodes = self.batches_per_episodes
        self.num_episodes = 20000
        self.args = args
        self.trial = 42


    def train_from_scratch_PPO1(self):
        print("Training from scratch called")
        self.args.train_log = False
        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing']
        self.args.task = task_list[1]
        prefix = self.args.task+"_trained_at_" + str(time.localtime().tm_mon) + "_" + str(time.localtime().tm_mday)\
            + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min) + ":" + str(time.localtime().tm_sec)
        model_dir = self.model_path + prefix + "_" + str(self.trial)
        os.makedirs(model_dir, exist_ok=True)
        print("\033[92m"+model_dir+"\033[0m")

        self.args.log_dir = model_dir
        self.args.robot_file = "jaco2_curtain_torque"
        # self.args.robot_file = "jaco2_curtain_velocity"
        self.args.controller = True
        self.args.n_robots = 1
        self.args.prev_action = False
        env = JacoMujocoEnv(**vars(self.args))

        net_arch = {'pi': model_configuration['layers']['policy'], 'vf': model_configuration['layers']['value']}
        if self.args.task is 'reaching':
            if self.args.controller:
                obs_relativity = {'subtract':{'ref':[18,19,20],'tar':[1,2,3]}}
                obs_index = [1,2,3,4,5,6, 18,19,20]
            else:
                obs_relativity = {'subtract':{'ref':[30,31,32,33,34,35],'tar':[13,14,15,16,17,18]}}
                obs_index = [1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,15,16,17,18, 30,31,32,33,34,35]
        elif self.args.task in ['grasping','carrying']:
            # obs_relativity = {'subtract':{'ref':[9,10,11],'tar':[1,2,3]}, 'leave':[2]}
            # obs_relativity = {'subtract':{'ref':[9,10,11],'tar':[1,2]}, 'leave':[0,1,2]}
            if self.args.controller:
                obs_relativity = {}
                # obs_index = [0, 1,2,3,4,5,6, 7,8,  9,10,11]
                obs_index = [0, 1,2,3,4,5,6, 7, 8,9,10]
            else:
                obs_relativity = {}
                obs_index = [0, 1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,15,16,17,18, 19, 20,21,22, 26]
        policy_kwargs = {'net_arch': [net_arch], 'obs_relativity':obs_relativity, 'obs_index':obs_index, 'squash':False}
        policy_kwargs.update(model_configuration['policy_kwargs'])
        model_dict = {'gamma': 0.99, 'clip_param': 0.02,
                      'tensorboard_log': model_dir, 'policy_kwargs': policy_kwargs, 'verbose':1}
        self.trainer = PPO1(MlpPolicy, env, **model_dict)
        
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes
        self.trainer.learn(total_timesteps=self.num_timesteps, save_interval=50, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")

    def train_from_scratch_PPO2(self):
        print("Training from scratch called")
        self.args.train_log = False
        self.args.visualize = False
        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing']
        self.args.task = task_list[1]
        prefix = self.args.task+"_trained_at_" + str(time.localtime().tm_mon) + "_" + str(time.localtime().tm_mday)\
            + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min) + ":" + str(time.localtime().tm_sec)
        prefix = 'test'
        model_dir = self.model_path + prefix + "_" + str(self.trial)
        os.makedirs(model_dir, exist_ok=True)
        print("\033[92m"+model_dir+"\033[0m")

        self.args.log_dir = model_dir
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.n_robots = 1
        
        env_list = []
        for i in range(4):
            env_list.append(JacoMujocoEnv)
        env = DummyVecEnv(env_list, dict(**vars(self.args)))
        env = VecNormalize(env)
        
        net_arch = {'pi': model_configuration['layers']['policy'], 'vf': model_configuration['layers']['value']}
        if self.args.task is 'reaching':
            obs_relativity = {'subtract':{'ref':[23,24,25,26,27,28],'tar':[0,1,2,3,4,5]}}
            obs_index = [0,1,2,3,4,5, 8,9,10,11,12,13, 23,24,25,26,27,28]
        elif self.args.task in ['grasping','carrying']:
            obs_relativity = {'subtract':{'ref':[14,15,16],'tar':[0,1,2]}, 'leave':[2]}
            obs_index = [0,1,2,3,4,5, 6,7, 14,15,16]
        policy_kwargs = {'net_arch': [net_arch], 'obs_relativity':obs_relativity, 'obs_index':obs_index}
        policy_kwargs.update(model_configuration['policy_kwargs'])
        model_dict = {'gamma': 0.99, 'tensorboard_log': model_dir, 'policy_kwargs': policy_kwargs}
        self.trainer = PPO2(MlpPolicy, env, **model_dict)
        
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes
        self.trainer.learn(total_timesteps=self.num_timesteps, save_interval=50, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")
    
    def train_from_scratch_SAC(self):
        print("Training from scratch called")
        self.args.train_log = False
        self.args.visualize = False
        task_list = ['reaching', 'grasping', 'carrying', 'releasing', 'pushing']
        self.args.task = task_list[1]
        prefix = self.args.task+"_trained_at_" + str(time.localtime().tm_mon) + "_" + str(time.localtime().tm_mday)\
            + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min) + ":" + str(time.localtime().tm_sec)
        # prefix="comparison_observation_range_sym_nobuffer"
        model_dir = self.model_path + prefix + "_" + str(self.trial)
        os.makedirs(model_dir, exist_ok=True)
        print("\033[92m"+model_dir+"\033[0m")

        self.args.log_dir = model_dir
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.controller = True
        self.args.n_robots = 1
        self.args.prev_action = False
        buffer = None

        net_arch = {'pi': model_configuration['layers']['policy'], 'vf': model_configuration['layers']['value']}
        if self.args.task is 'reaching':
            if self.args.controller:
                traj_dict = np.load(self.model_path+'trajectories/'+self.args.task+".npz", allow_pickle=True)
                self.args.init_buffer = np.array(traj_dict['obs'])
                # obs_relativity = {'subtract':{'ref':[17,18,19,20,21,22],'tar':[1,2,3,4,5,6]}, 'leave':[1,2,3]}
                obs_relativity = {'subtract':{'ref':[17,18,19,20,21,22],'tar':[1,2,3,4,5,6]}}
                # obs_relativity = {}
                obs_index = [1,2,3,4,5,6, 17,18,19,20,21,22]
            else:
                obs_relativity = {'subtract':{'ref':[30,31,32,33,34,35],'tar':[13,14,15,16,17,18]}}
                obs_index = [1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,15,16,17,18, 30,31,32,33,34,35]
        elif self.args.task in ['grasping','carrying']:
            buffer = self.create_buffer('trajectory_expert5_mod')
            # obs_relativity = {'subtract':{'ref':[9,10,11],'tar':[1,2,3]}, 'leave':[2]}
            # obs_relativity = {'subtract':{'ref':[9,10,11],'tar':[1,2]}, 'leave':[0,1,2]}
            if self.args.controller:
                obs_relativity = {}
                # obs_index = [0, 1,2,3,4,5,6, 7,8,  9,10,11]
                obs_index = [0, 1,2,3,4,5,6, 7, 8,9,10]
            else:
                obs_relativity = {}
                obs_index = [0, 1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,15,16,17,18, 19, 20,21,22, 26]
        
        env = JacoMujocoEnv(**vars(self.args))

        policy_kwargs = {'net_arch': [net_arch], 'obs_relativity':obs_relativity, 'obs_index':obs_index}
        policy_kwargs.update(model_configuration['policy_kwargs'])
        model_dict = {'gamma': 0.99, 'tensorboard_log': model_dir, 'policy_kwargs': policy_kwargs, 'verbose': 1, \
                      'replay_buffer': buffer, 'learning_rate':_lr_scheduler}
        self.trainer = SAC_MULTI(MlpPolicy_hpcsac, env, **model_dict)
        
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes
        self.trainer.learn(total_timesteps=self.num_timesteps, save_interval=10000, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")

    def train_continue(self):
        self.args.train_log = False
        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing']
        self.args.task = task_list[0]
        self.args.visualize = False
        self.args.prev_action = False
        model_dir = self.model_path + 'reaching_trained_at_1_13_17:47:15_31/continue1'
        policy_dir = model_dir + '/policy_3860000.zip'
        sub_dir = '/continue4'
        print("\033[92m"+model_dir + sub_dir+"\033[0m")
        
        # buffer = self.create_buffer('trajectory_expert5')
        buffer = None

        self.args.log_dir = model_dir
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.n_robots = 1
        traj_dict = np.load(self.model_path+'trajectories/'+self.args.task+"2.npz", allow_pickle=True)
        self.args.init_buffer = np.array(traj_dict['obs'])
        env = JacoMujocoEnv(**vars(self.args))
        
        os.makedirs(model_dir+sub_dir, exist_ok=True)
        # net_arch = {'pi': [128,128], 'vf': [64, 64]}
        # policy_kwargs = {'net_arch': [net_arch]}
        self.trainer = SAC_MULTI.load(policy_dir, policy=MlpPolicy_hpcsac, env=env, replay_buffer=buffer, tensorboard_log=model_dir+sub_dir, \
                                    learning_rate=_lr_scheduler, learning_starts=0)
        # self.trainer = PPO1.load(policy_dir, env=env, tensorboard_log=model_dir+sub_dir, policy_kwargs=policy_kwargs, exact_match=True, only={'value':True})
        # self.trainer = PPO1.load(policy_dir, env=env, tensorboard_log=model_dir+sub_dir)
        self.trainer.learn(total_time_step, save_interval=10000, save_path=model_dir+sub_dir)
        print("Train Finished")
        self.trainer.save(model_dir+sub_dir)

    def train_from_expert(self, n_episodes=100):
        print("Training from expert called")
        self.args.train_log = False
        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing']
        self.args.task = task_list[1]
        prefix = self.args.task+"_trained_from_expert_at_" + str(time.localtime().tm_mon) + "_" + str(time.localtime().tm_mday)\
            + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min) + ":" + str(time.localtime().tm_sec)
        model_dir = self.model_path + prefix + "_" + str(self.trial)
        print("\033[92m"+model_dir+"\033[0m")

        # self._open_connection()
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.prev_action = False
        # env = JacoMujocoEnv(**vars(self.args))        
        traj_dict = np.load(self.model_path+'trajectories/'+self.args.task+"_trajectory_expert4.npz", allow_pickle=True)
        dataset = ExpertDataset(traj_data=traj_dict, batch_size=16384)
        buffer = ReplayBuffer(64)

        quit()

        net_arch = {'pi': model_configuration['layers']['policy'], 'vf': model_configuration['layers']['value']}
        if self.args.task is 'reaching':
            obs_relativity = {'subtract':{'ref':[18,19,20],'tar':[1,2,3]}}
            obs_index = [1,2,3,4,5,6, 18,19,20]
        elif self.args.task in ['grasping','carrying']:
            # obs_relativity = {'subtract':{'ref':[9,10,11],'tar':[1,2,3]}, 'leave':[2]}
            # obs_relativity = {'subtract':{'ref':[9,10,11],'tar':[1,2]}, 'leave':[0,1,2]}
            obs_relativity = {}
            # obs_index = [0, 1,2,3,4,5,6, 7,8,  9,10,11]
            obs_index = [0, 1,2,3,4,5,6, 7, 8,9,10]
        policy_kwargs = {'net_arch': [net_arch], 'obs_relativity':obs_relativity, 'obs_index':obs_index}
        policy_kwargs.update(model_configuration['policy_kwargs'])
        model_dict = {'gamma': 0.99, 'clip_param': 0.02,
                      'tensorboard_log': model_dir, 'policy_kwargs': policy_kwargs, 'verbose':1}
        self.trainer = PPO1(MlpPolicy, env, **model_dict)

        # model_dir = self.model_path + 'grasping_trained_from_expert_at_12_8_12:15:5'
        # policy_dir = model_dir + "/policy_19500.zip"
        # sub_dir = '/continue_from_expert'
        # model_dir += sub_dir
        # self.trainer = PPO1.load(policy_dir, env=env)

        print("\033[91mPretraining Starts\033[0m")
        self.trainer.pretrain(dataset, **pretrain_configuration)
        print("\033[91mPretraining Finished\033[0m")
        del dataset
        os.makedirs(model_dir, exist_ok=True)
        
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes
        self.trainer.learn(total_time_step, save_interval=50, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")

    def _open_connection(self):
        try:            
            print("Opening connection to SpaceNav driver ...")
            spacenav.open()
            print("... connection established.")
        except Exception:
            print("No connection to the SpaceNav driver. Is spacenavd running?")
    
    def _close_connection(self):
        atexit.register(spacenav.close)

    def _expert_3d(self, _obs):
        if sys.platform in ['linux', 'linux2']:
            event = spacenav.poll()
            if type(event) is spacenav.MotionEvent:
                # action = np.array([event.x, event.z, event.y, event.rx, -event.ry, event.rz, self.g_angle])/350*1.5
                action = np.array([event.x, event.z, event.y, event.rx, -event.ry, event.rz])/350*1.5
            elif type(event) is spacenav.ButtonEvent:
                if self.g_changed is not None:
                    self.g_changed = not self.g_changed
                else:
                    self.g_changed = True
                try:
                    action = np.array([event.x, event.z, event.y, event.rx, -event.ry, event.rz, 0])/350*1.5
                except Exception:
                    action = [0,0,0,0,0,0,0]
                self.pressed[event.button] = event.pressed
            else:
                # action = [0,0,0,0,0,0,0]
                action = [0,0,0,0,0,0]
            if self.pressed[0]:
                self.g_angle = 0.5
            elif self.pressed[1]:
                self.g_angle = -0.5
            else:
                self.g_angle = 0
            
            # action[6] = self.g_angle
            spacenav.remove_events(1)
            if self.g_changed is not None and not self.g_changed:
                # print("Removed")
                spacenav.remove_events(2)
                self.g_changed = None
            return action
        else:
            action = [0,0,0,0,0,0,0]
            return action
    
    def generate_traj(self):
        print("Trajectory Generating")
        self.args.train_log = False
        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing']
        self.args.task = task_list[0]

        self._open_connection()
        self.args.robot_file = "jaco2_curtain_torque"
        env = JacoMujocoEnv(**vars(self.args))
        # traj_dict = generate_expert_traj(self._expert_3d, self.model_path+'/trajectories/'+self.args.task+'_trajectory_expert1', env, n_episodes=100)
        traj_dict = generate_expert_traj(self._expert_3d, self.model_path+'/trajectories/'+self.args.task+"2", env, n_episodes=1)
        self._close_connection()

    def create_buffer(self, name):
        traj_dict = np.load(self.model_path+'trajectories/'+self.args.task+"_"+name+".npz", allow_pickle=True)
        buffer = ReplayBuffer(50000, discard=True)
        buff_arry = []
        print("episodes: ", len(traj_dict['actions']))
        for i in range(len(traj_dict['actions'])):
            obs = np.array(traj_dict['obs'][i])
            action = traj_dict['actions'][i]
            reward = traj_dict['rewards'][i]
            new_obs = traj_dict['next_obs'][i]
            done = float(traj_dict['episode_starts'][i])
            buff_arry.append((np.copy(obs), np.copy(action), np.copy(reward), np.copy(new_obs), np.copy(done)))
            del obs, action, reward, new_obs, done
        # print(buff_arry)
        buffer.set_storate(buff_arry)
        del traj_dict
        return buffer

    def train_HPC(self):
        task_list = ['picking', 'placing', 'pickAndplace']
        composite_primitive_name = self.args.task = task_list[0]
        algo_list = ['sac','ppo']
        algo = algo_list[1]

        self.args.train_log = False
        self.args.visualize = False
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.controller = True
        self.args.n_robots = 1
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        self.args.auxiliary = False

        if algo == 'sac':
            env = JacoMujocoEnv(**vars(self.args))
            policy = MlpPolicy_hpcsac
            self.model = SAC_MULTI(policy=policy, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)
        elif algo == 'ppo':
            env_list = []
            for i in range(1):
                env_list.append(JacoMujocoEnv)
            env = DummyVecEnv(env_list, dict(**vars(self.args)))
            env = VecNormalize(env)
            policy = MlpPolicy_hpcppo
            self.model = HPCPPO(policy=policy, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)


        if self.args.auxiliary:
            prefix = composite_primitive_name + "_trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(
                    time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)
        else:
            prefix = composite_primitive_name +'_'+algo+"_noaux_trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(
                    time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)
        # prefix = 'HPCtest'
        model_dir = self.model_path + prefix + "_" + str(self.trial)
        self.args.log_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        print("\033[92m"+model_dir+"\033[0m")

        obs_min = [-3, -1,-1,-1,-1,-1,-1, -1, -1,-1,-1, -1,-1,-1,-1,-1,-1]
        obs_max = [ 3,  1, 1, 1, 1, 1, 1,  1,  1, 1, 1,  1, 1, 1, 1, 1, 1]
        act_min = [-1,-1,-1,-1,-1,-1, -1]
        act_max = [ 1, 1, 1, 1, 1, 1,  1]
        obs_idx = [0, 1,2,3,4,5,6, 7, 8,9,10, 17,18,19,20,21,22]
        act_idx = [0,1,2,3,4,5, 6]
        if self.args.auxiliary:
            self.model.construct_primitive_info(name='aux1', freeze=False, level=0,
                                            obs_range={'min': obs_min, 'max': obs_max}, obs_index=obs_idx, 
                                            act_range={'min': act_min, 'max': act_max}, act_index=act_idx, act_scale=0.1,
                                            obs_relativity={},
                                            layer_structure={'policy':[128, 128, 128]})

        # Pretrained primitives
        prim_name = 'reaching'
        # policy_zip_path = self.model_path+prim_name+'_trained_at_1_13_17:47:15_31/continue1/continue4/policy_3580000.zip'
        policy_zip_path = self.model_path+prim_name+'_trained_at_1_13_17:47:15_31/continue1/policy_3860000.zip'
        self.model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                        obs_range=None, obs_index=[1,2,3,4,5,6, 17,18,19,20,21,22],
                                        act_range=None, act_index=[0,1,2,3,4,5], act_scale=1,
                                        obs_relativity={'subtract':{'ref':[17,18,19,20,21,22],'tar':[1,2,3,4,5,6]}},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
                                        load_value=False)

        prim_name = 'grasping'
        # policy_zip_path = self.model_path+prim_name+"_trained_at_12_28_17:26:27_15/continue1/policy_2330000.zip"
        policy_zip_path = self.model_path+'comparison_observation_range_sym_discard_0/policy_8070000.zip'
        self.model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                        obs_range=None, obs_index=[0, 1,2,3,4,5,6, 7, 8,9,10], 
                                        act_range=None, act_index=[0,1,2,3,4,5, 6], act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                        load_value=False)
        
        # Weight definition
        number_of_primitives = 3 if self.args.auxiliary else 2
        if self.args.rulebased_subgoal:
            subgoal_dict = None
        else:
            subgoal_dict = {'level1_reaching/level0':[17,18,19,20,21,22]}
        self.model.construct_primitive_info(name='weight', freeze=False, level=1,
                                        obs_range=0, obs_index=obs_idx,
                                        act_range=0, act_index=list(range(number_of_primitives)), act_scale=None,
                                        obs_relativity={},
                                        layer_structure={'policy':[256, 256, 256],'value':[256, 256, 256]},
                                        subgoal=subgoal_dict)

        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes 
        model_dict = {'gamma': 0.99, 'tensorboard_log': model_dir,'verbose': 1, \
            'learning_rate':_lr_scheduler, 'learning_starts':100, 'ent_coef': 0, 'batch_size': 1} #
        if algo == 'sac':
            self.model.pretrainer_load(model=self.model, policy=policy, env=env, **model_dict)
        elif algo == 'ppo':
            self.model.pretrainer_load(model=self.model, policy=policy, env=env, **model_dict)
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.model.learn(total_timesteps=self.num_timesteps, save_interval=100, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.model.save(model_dir+"/policy")

    def _write_log(self, model_dir, info):
        model_log = open(model_dir+"/model_log.txt", 'w')
        if info['layers'] != None:
            model_log.writelines("Layers:\n")
            model_log.write("\tpolicy:\t[")
            for i in range(len(info['layers']['policy'])):
                model_log.write(str(info['layers']['policy'][i]))
                if i != len(info['layers']['policy'])-1:
                    model_log.write(", ")
                else:
                    model_log.writelines("]\n")
            model_log.write("\tvalue:\t[")
            for i in range(len(info['layers']['value'])):
                model_log.write(str(info['layers']['value'][i]))
                if i != len(info['layers']['value'])-1:
                    model_log.write(", ")
                else:
                    model_log.writelines("]\n\n")
            info.pop('layers')
        
        for name, item in info.items():
            model_log.writelines(name+":\t\t{0}\n".format(item))
        model_log.close()

    def test(self):
        print("Testing called")
        self.args.train_log = False
        self.args.visualize = True
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.n_robots = 1

        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing']
        self.args.task = task_list[0]
        algo_list = ['sac','ppo']
        algo = algo_list[0]
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        
        if self.args.task == 'reaching':
            traj_dict = np.load(self.model_path+'trajectories/'+self.args.task+"2.npz", allow_pickle=True)
            self.args.init_buffer = np.array(traj_dict['obs'])
        
        if algo == 'sac':
            env = JacoMujocoEnv(**vars(self.args))
        elif algo == 'ppo':
            env_list = []
            for i in range(2):
                env_list.append(JacoMujocoEnv)
            env = DummyVecEnv(env_list, dict(**vars(self.args)))
            env = VecNormalize(env)

        ##### Grasping
        # Upper grasp
        # prefix = self.args.task + '_trained_at_12_28_17:26:27_15/continue1/policy_2330000.zip'
        # Side grasp (better)
        # prefix = 'comparison_observation_range_sym_discard_0/policy_8070000.zip'
        # Side grasp
        # prefix = 'comparison_observation_range_sym_nobuffer_2/policy_4330000.zip'

        ##### Reaching
        # prefix = self.args.task + '_trained_at_11_27_18:25:54/policy_9999105.zip'
        # prefix = self.args.task + '_trained_at_1_2_20:34:52_22/policy.zip'
        # prefix = self.args.task + '_trained_at_1_3_17:15:43_23/policy.zip'
        # prefix = self.args.task + '_trained_at_1_4_23:29:45_24/policy.zip'
        # prefix = self.args.task + '_trained_at_1_4_23:30:10_25/policy.zip'
        # prefix = self.args.task + '_trained_at_1_8_16:1:46_26/policy_7440000.zip'
        # prefix = self.args.task + '_trained_at_1_8_16:1:46_26/policy.zip'
        # prefix = self.args.task + '_trained_at_1_8_16:2:2_27/policy_7420000.zip'
        # prefix = self.args.task + '_trained_at_1_13_17:47:41_32/policy_6750000.zip'
        prefix = self.args.task + '_trained_at_1_13_17:47:15_31/continue1/policy_3860000.zip'
        # prefix = self.args.task + '_trained_at_1_13_17:47:15_31/continue1/policy_4300000.zip'
        # prefix = self.args.task + '_trained_at_1_13_17:47:15_31/continue1/continue4/policy_3580000.zip'

        ##### Picking
        # prefix = self.args.task + '_trained_at_2021_1_20_12:5_39/policy_8990000.zip'
        # prefix = 'HPCtest_0/policy_2710000.zip'
        # prefix = self.args.task + '_ppo_noaux_trained_at_2021_2_25_15:29_42/policy_50689.zip'
        # prefix = self.args.task + '_ppo_noaux_trained_at_2021_2_26_14:16_42/policy.zip'


        model_dir = self.model_path + prefix
        test_iter = 100
        if self.args.task in ['picking','placing','pickAndplace']:
            if algo == 'sac':
                self.model = SAC_MULTI(policy=MlpPolicy_hpcsac, env=None, _init_setup_model=False, composite_primitive_name='picking')
            elif algo == 'ppo':
                self.model = HPCPPO(policy=MlpPolicy_hpcppo, env=None, _init_setup_model=False, composite_primitive_name='picking')
            obs_idx = [0, 1,2,3,4,5,6, 7, 8,9,10, 17,18,19,20,21,22]
            act_idx = [0,1,2,3,4,5, 6]
            self.model.construct_primitive_info(name=None, freeze=True, level=1,
                                                obs_range=None, obs_index=obs_idx,
                                                act_range=None, act_index=act_idx, act_scale=1,
                                                obs_relativity={},
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(model_dir), 
                                                load_value=True)
            if algo == 'sac':
                SAC_MULTI.pretrainer_load(self.model, MlpPolicy_hpcsac, env)
            elif algo == 'ppo':
                HPCPPO.pretrainer_load(self.model, MlpPolicy_hpcsac, env)
        else:
            if algo == 'sac':
                print("SAC model LOADING in MAIN")
                self.model = SAC_MULTI.load(model_dir, MlpPolicy_hpcsac, env)
            else:
                pass
        for _ in range(test_iter):
            iter = 0
            obs = env.reset()
            done = False
            while not done:
                iter += 1
                if self.args.task in ['picking','placing','pickAndplace']:
                    action, subgoal, weight = self.model.predict_subgoal(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action, log=False, weight=weight, subgoal=subgoal)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action, log=False)
                if algo == 'ppo':
                    done = True if done.any() == True else False
                print('gripper action: ', action)
                # print('reward: {0:2.3f}'.format(reward), end='\n')

    def generate(self):
        pass


if __name__ == "__main__":
    controller = RL_controller()
    # controller.train_from_scratch_PPO1()  
    # controller.train_from_scratch_PPO2()
    # controller.train_from_scratch_SAC()
    # controller.train_continue()
    # controller.train_from_expert()
    controller.train_HPC()
    # controller.generate_traj()
    # controller.test()