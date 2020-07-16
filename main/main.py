#!/usr/bin/env python

import os
import sys
import path_config
import time
from collections import OrderedDict

import stable_baselines.common.tf_util as tf_util
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac import SAC
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_sac, LnMlpPolicy as LnMlpPolicy_sac
from stable_baselines.gail import generate_expert_traj
from env_script.env_mujoco import JacoMujocoEnv
from state_gen.state_generator import State_generator
#from reward_module.reward_module import *

from argparser import ArgParser


class RL_controller:
    def __init__(self):
        # Arguments
        parser = ArgParser(isbaseline=True)
        args = parser.parse_args()

        # Debug
        args.debug = True
        print("DEBUG = ", args.debug)

        # TensorFlow Setting for State Representation Module
        # NOTE: RL trainer will use its own tf.Session
        self.sess_SRL = tf_util.single_threaded_session()
        args.sess = self.sess_SRL

        # State Generation Module defined here
        #self.stateGen = State_generator(**vars(args))
        #args.stateGen = self.stateGen

        # Reward Generation
        self.reward_method = "l2"
        self.reward_module = ""
        args.reward_method = self.reward_method
        args.reward_module = self.reward_module

        # If resume training on pre-trained models with episodes, else None
        if sys.platform in ["linux", "linux2"]:
            self.model_path = "/home/ljh/Project/mujoco_jaco/models_baseline/"
            self.tb_dir = "/home/ljh/Project/mujoco_jaco/tensorboard_log/"
        elif sys.platform == "darwin":
            self.model_path = "/Users/jeonghoon/Google_drive/Workspace/MLCS/mujoco_jaco/models_baseline/"
            self.tb_dir = "/Users/jeonghoon/Google_drive/Workspace/MLCS/mujoco_jaco/tensorboard_log/"
        else:
            raise NotImplementedError
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.steps_per_batch = 100
        self.batches_per_episodes = 5
        args.steps_per_batch = self.steps_per_batch
        args.batches_per_episodes = self.batches_per_episodes
        self.num_episodes = 1
        self.args = args


    def train_from_scratch(self):
        print("Training from scratch called")
        self.args.train_log = True
        prefix = "trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(
                time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)
        model_dir = self.model_path + prefix
        self.args.log_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        tb_path = self.tb_dir + prefix
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes
        # layers = {"policy": [64, 64], "value": [256, 256, 128]}
        layers = None
        env = JacoMujocoEnv(**vars(self.args))

        # NOTE: Layer Normalization for RNNs, but for just fc..?
        #self.trainer = TRPO(MlpPolicy, self.env, cg_damping=0.1, vf_iters=5, vf_stepsize=1e-3, timesteps_per_batch=self.steps_per_batch,
        #                   tensorboard_log=tb_path, full_tensorboard_log=True)
        #self.trainer = SAC(LnMlpPolicy_sac, env, layers=layers,
        #                   tensorboard_log=tb_path, full_tensorboard_log=True)
        self.trainer = SAC(MlpPolicy_sac, env, layers=layers, 
                            tensorboard_log=tb_path, full_tensorboard_log=True)
        model_log = open(model_dir+"/model_log.txt", 'w')
        self._write_log(model_log, layers)
        model_log.close()
        print("\033[91mTraining Starts\033[0m")
        self.trainer.learn(total_timesteps=self.num_timesteps)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")
    
    def _write_log(self, model_log, layers):
        if layers != None:
            model_log.writelines("Layers:\n")
            model_log.write("\tpolicy:\t[")
            for i in range(len(layers['policy'])):
                model_log.write(str(layers['policy'][i]))
                if i != len(layers['policy'])-1:
                    model_log.write(", ")
                else:
                    model_log.writelines("]\n")
            model_log.write("\tvalue:\t[")
            for i in range(len(layers['value'])):
                model_log.write(str(layers['value'][i]))
                if i != len(layers['value'])-1:
                    model_log.write(", ")
                else:
                    model_log.writelines("]\n")
        model_log.writelines("Reward Method:\t\t\t\t{0}\n".format(self.reward_method))
        model_log.writelines("Steps per batch:\t\t\t{0}\n".format(self.steps_per_batch))
        model_log.writelines("Batches per episodes:\t\t{0}\n".format(self.batches_per_episodes))
        model_log.writelines("Numbers of episodes:\t\t{0}\n".format(self.num_episodes))
        model_log.writelines("Total number of episodes:\t{0}\n".format(self.steps_per_batch * self.batches_per_episodes * self.num_episodes))
    
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

    def train_from_expert(self, n_episodes=10):
        print("Training from expert called")
        self.args.train_log = False

        env = JacoMujocoEnv(**vars(self.args))
        generate_expert_traj(self._expert, 'expert_traj',
                             env, n_episodes=n_episodes)

    def _expert(self, _obs):
        action = []
        return action

    def train_with_additional_layer(self):
        self.args.train_log = False
        env = JacoMujocoEnv(**vars(self.args))

        primitives = OrderedDict()
        # Newly appointed primitives
        SAC_MULTI.construct_primitive_info(name='train/aux1', primitive_dict=primitives, 
                                            obs_dimension=6, obs_range=[-2, 2], obs_index=[0, 1, 2, 3, 4, 5], 
                                            act_dimension=6, act_range=[-1.4, 1.4], act_index=[0, 1, 2, 3, 4, 5], 
                                            layer_structure=[256, 256])
        SAC_MULTI.construct_primitive_info('train/aux2', primitives, 
                                            6, [-2, 2], [0, 1, 2, 3, 4, 5], 
                                            6, [-1.4, 1.4], [0, 1, 2, 3, 4, 5], 
                                            [256, 128])
        # Pretrained primitives
        policy_zip_path = self.model_path+"test"+"/policy.zip"
        SAC_MULTI.construct_primitive_info('freeze/reaching', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1, 2, 3, 4, 5], 
                                            act_dimension=None, act_range=None, act_index=[0, 1, 2, 3, 4, 5], 
                                            layer_structure=None, loaded_policy=SAC_MULTI._load_from_file(policy_zip_path))
        # Weight definition  
        number_of_primitives = 3
        total_obs_dim = env.get_num_observation()
        SAC_MULTI.construct_primitive_info('train/weight', primitives, 
                                            total_obs_dim, 0, list(range(total_obs_dim)), 
                                            number_of_primitives, [0,1], number_of_primitives, 
                                            [512, 512, 512])

        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes 
        self.trainer = SAC_MULTI.pretrainer_load(policy=MlpPolicy_sac, primitives=primitives, env=env)
        self.trainer.learn(total_timesteps=self.num_timesteps)


    def test(self, policy):
        print("Testing called")
        self.args.train_log = False
        self.env = JacoMujocoEnv(**vars(self.args))
        model_dir = self.model_path+policy+"/policy.zip"
        test_iter = 100
        self.model = SAC.load(model_dir)
        for _ in range(test_iter):
            obs = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, rewards, done, _ = self.env.step(action, log=False)
                print("{0:2.3f}".format(rewards), end='\r')
    
    def generate(self):
        pass


if __name__ == "__main__":
    controller = RL_controller()
    iter = 0
    while True:
        opt = input("Train / Test / Generate (1/2/3): ")
        if opt == "1":
            iter_train = 0
            while True:
                iter_train += 1
                opt2 = input("Train_from_scratch / Train_from_pre_model / Train_from_expert / Train with additional layer (1/2/3/4): ")
                if opt2 == "1":
                    controller.train_from_scratch()
                    break
                elif opt2 == "2":
                    model_dir = input("Enter model name: ")
                    controller.train_continue(model_dir)
                    break
                elif opt2 == "3":
                    n_episodes = int(
                        input("How many trials do you want to record?"))
                    controller.train_from_expert(n_episodes)
                    break
                elif opt2 == "4":
                    controller.train_with_additional_layer()
                    break
                else:
                    if iter_train <= 5:
                        print(
                            "Wront input, press 1 or 2 (Wrong trials: {0})".format(iter_train))
                    else:
                        print("Wront input, Abort")
                        break
            break
        elif opt == "2":
            policy = input("Enter trained policy: ")
            controller.test(policy)
            break
        elif opt == "3":
            controller.generate()
            break
        else:
            iter += 1
            if iter <= 5:
                print(
                    "Wront input, press 1 or 2 (Wrong trials: {0})".format(iter))
            else:
                print("Wront input, Abort")
                break
