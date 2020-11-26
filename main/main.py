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
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.ppo2 import PPO2
from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_sac
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common import set_global_seeds

from env_script.env_mujoco import JacoMujocoEnv
from state_gen.state_generator import State_generator

from argparser import ArgParser


class RL_controller:
    def __init__(self):
        # Arguments
        parser = ArgParser(isbaseline=True)
        args = parser.parse_args()

        # Debug
        args.debug = True
        print("DEBUG = ", args.debug)

        self.sess_SRL = tf_util.single_threaded_session()
        args.sess = self.sess_SRL

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
        
        #self.steps_per_batch = 100
        self.steps_per_batch = 10000
        self.batches_per_episodes = 5
        args.steps_per_batch = self.steps_per_batch
        args.batches_per_episodes = self.batches_per_episodes
        self.num_episodes = 10000
        self.args = args


    def train_from_scratch(self):
        print("Training from scratch called")
        self.args.train_log = False
        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing']
        self.args.task = task_list[1]
        prefix = self.args.task+"_trained_at_" + str(time.localtime().tm_mon) + "_" + str(time.localtime().tm_mday)\
            + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min) + ":" + str(time.localtime().tm_sec)
        model_dir = self.model_path + prefix
        os.makedirs(model_dir, exist_ok=True)

        self.args.log_dir = model_dir
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.n_robots = 1
        env = JacoMujocoEnv(**vars(self.args))

        net_arch = {'pi': model_configuration['layers']['policy'], 'vf': model_configuration['layers']['value']}
        if self.args.task is 'reaching':
            obs_relativity = {'subtract':{'ref':[14,15,16,17,18,19],'tar':[0,1,2,3,4,5]}}
            obs_index = [0,1,2,3,4,5, 14,15,16,17,18,19]
        elif self.args.task is 'grasping':
            obs_relativity = {'subtract':{'ref':[8,9,10],'tar':[0,1,2]}}
            obs_index = [0,1,2,3,4,5, 6,7, 8,9,10]
        policy_kwargs = {'net_arch': [net_arch], 'obs_relativity':obs_relativity, 'obs_index':obs_index}
        policy_kwargs.update(model_configuration['policy_kwargs'])
        model_dict = {'gamma': 0.99, 'clip_param': 0.02,
                      'tensorboard_log': model_dir, 'policy_kwargs': policy_kwargs}
        self.trainer = PPO1(MlpPolicy, env, **model_dict)
        #self.trainer = SAC_MULTI(MlpPolicy_sac, env, **model_configuration)
        
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes
        self.trainer.learn(total_timesteps=self.num_timesteps, save_interval=10, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")


    def train_continue(self, model_dir):
        self.args.train_log = False
        env = JacoMujocoEnv(**vars(self.args))
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes 
        try:
            # self.trainer = SAC.load(self.model_path + model_dir + "/policy.zip", env=env)
            self.trainer = SAC_MULTI.load(self.model_path+model_dir+"/policy.zip", env=env)
            print("layer_norm: ",self.trainer.policy_tf.layer_norm)
            '''
            param_dict = SAC_MULTI._load_from_file(self.model_path+model_dir+"/policy.zip")[1]
            for name, value in param_dict.items():
                print(name, value.shape)
            '''
            quit()
            self.trainer.learn(total_timesteps=self.num_timesteps)
            print("Train Finished")
            self.trainer.save(model_dir)
        except Exception as e:
            print(e)


    def train_from_expert(self, n_episodes=100):
        print("Training from expert called")
        self.args.train_log = False
        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing']
        self.args.task = task_list[1]
        prefix = self.args.task+"_trained_from_expert_at_" + str(time.localtime().tm_mon) + "_" + str(time.localtime().tm_mday)\
            + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min) + ":" + str(time.localtime().tm_sec)
        model_dir = self.model_path + prefix
        os.makedirs(model_dir, exist_ok=True)

        self._open_connection()
        self.args.robot_file = "jaco2_curtain_torque"
        env = JacoMujocoEnv(**vars(self.args))
        traj_dict = generate_expert_traj(self._expert_3d, self.args.task+'_trajectory_expert', env, n_episodes=n_episodes)
        self._close_connection()
        traj_dict = np.load(model_dir+"/trajectory_expert.npz", allow_pickle=True)
        dataset = ExpertDataset(traj_data=traj_dict, batch_size=1024)
        
        net_arch = {'pi': model_configuration['layers']['policy'], 'vf': model_configuration['layers']['value']}
        if self.args.task is 'reaching':
            obs_relativity = {'subtract':{'ref':[14,15,16,17,18,19],'tar':[0,1,2,3,4,5]}}
            obs_index = [0,1,2,3,4,5, 14,15,16,17,18,19]
        elif self.args.task is 'grasping':
            obs_relativity = {'subtract':{'ref':[8,9,10],'tar':[0,1,2]}}
            obs_index = [0,1,2,3,4,5, 6,7, 8,9,10]
        policy_kwargs = {'net_arch': [net_arch], 'obs_relativity':obs_relativity, 'obs_index':obs_index}
        policy_kwargs.update(model_configuration['policy_kwargs'])
        model_dict = {'gamma': 0.99, 'clip_param': 0.02,
                      'tensorboard_log': model_dir, 'policy_kwargs': policy_kwargs}
        self.trainer = PPO1(MlpPolicy, env, **model_dict)
        self.trainer.pretrain(dataset, **pretrain_configuration)
        del dataset
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes
        self.trainer.learn(total_time_step, save_interval=int(total_time_step*0.05), save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")

    def _open_connection(self):
        try:            
            print("Opening connection to SpaceNav driver ...")
            spacenav.open()
            print("... connection established.")
        except spacenav.ConnectionError:
            print("No connection to the SpaceNav driver. Is spacenavd running?")
    
    def _close_connection(self):
        atexit.register(spacenav.close)

    def _expert_3d(self, _obs):
        if sys.platform in ['linux', 'linux2']:
            event = spacenav.poll()
            if type(event) is spacenav.MotionEvent:
                action = np.array([event.x, event.z, event.y, event.rx, -event.ry, event.rz, self.g_angle, self.g_angle])/350*1.5
            elif type(event) is spacenav.ButtonEvent:
                # print("button: ",event.button)
                # print("pressed: ",event.pressed)
                if self.g_changed is not None:
                    self.g_changed = not self.g_changed
                else:
                    self.g_changed = True
                try:
                    action = np.array([event.x, event.z, event.y, event.rx, -event.ry, event.rz, 0, 0])/350*1.5
                except Exception:
                    action = [0,0,0,0,0,0,0,0]
                self.pressed[event.button] = event.pressed
            else:
                action = [0,0,0,0,0,0,0,0]
            if self.pressed[0]:
                self.g_angle = 0.5
            elif self.pressed[1]:
                self.g_angle = -0.5
            else:
                self.g_angle = 0
            #print("self.prssed: ",self.pressed, self.g_angle)
            
            action[6] = action[7] = self.g_angle
            spacenav.remove_events(1)
            if self.g_changed is not None and not self.g_changed:
                # print("Removed")
                spacenav.remove_events(2)
                self.g_changed = None

            return action

        else:
            action = [0,0,0,0,0,0,0,0]
            return action
    

    def train_with_additional_layer(self):
        self.args.train_log = False
        env = JacoMujocoEnv(**vars(self.args))
        prefix = "trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(
                time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)

        model_dir = self.model_path + prefix
        self.args.log_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        composite_primitive_name='pick'
        model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)
        
        model.construct_primitive_info(name='aux1', freeze=False, level=0,
                                        obs_range=[-2, 2], obs_index=[0, 1, 2, 3, 4, 5], 
                                        act_range=[-1.4, 1.4], act_index=[0, 1, 2, 3], act_scale=0.1,
                                        obs_relativity=None,
                                        layer_structure={'policy':[64, 64]})

        # Pretrained primitives
        prim_name = 'reaching'
        policy_zip_path = self.model_path+"test"+"/policy.zip"
        model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                        obs_range=None, obs_index=[0, 1, 2, 3, 4, 5],
                                        act_range=None, act_index=[0, 1, 2, 3, 4, 5], act_scale=1,
                                        obs_relativity={'substract':{'ref':[6,7,8], 'tar':[0,1,2]}},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
                                        load_value=True)

        prim_name = 'grasping'
        policy_zip_path = self.model_path+"test2"+"/policy.zip"
        model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                        obs_range=None, obs_index=[0, 1, 2, 3, 4, 5], 
                                        act_range=None, act_index=[0, 1, 2, 3, 4, 5], act_scale=1,
                                        obs_relativity={'substract':{'ref':[6,7,8], 'tar':[0,1,2]}},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                        load_value=True)
        
        # Weight definition  
        number_of_primitives = 5
        total_obs_dim = env.get_num_observation()
        model.construct_primitive_info(name='weight', freeze=False, level=1,
                                        obs_range=0, obs_index=list(range(total_obs_dim)),
                                        act_range=0, act_index=list(range(number_of_primitives)), act_scale=None,
                                        obs_relativity=None,
                                        layer_structure={'policy':[512, 256, 256],'value':[512, 256, 256]})

        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes 
        model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env, **model_configuration)
        print("\033[91mTraining Starts\033[0m")
        self.trainer.learn(total_timesteps=self.num_timesteps)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")


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
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.n_robots = 1

        self.args.task = 'reaching'
        #self.args.task = 'grasping'
        env = JacoMujocoEnv(**vars(self.args))
        #prefix = "reaching_trained_at_11_25_17:9:27/policy_4999425.zip"
        prefix = "reaching_trained_at_11_25_17:9:35/policy_4999425.zip"
        model_dir = self.model_path + prefix
        test_iter = 100
        # self.model = SAC_MULTI.pretrainer_load(model_dir)
        self.model = PPO1.load(model_dir)
        for _ in range(test_iter):
            obs = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, _, done, _ = env.step(action, log=False)
                print(action, end='\r')
    
    def generate(self):
        pass


if __name__ == "__main__":
    controller = RL_controller()
    #controller.train_from_scratch()
    #controller.test()
    controller.train_from_expert()