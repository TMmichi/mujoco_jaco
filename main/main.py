#!/usr/bin/env python

import os, sys
import path_config
import math

import stable_baselines.common.tf_util as tf_util
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac import SAC
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_sac, LnMlpPolicy as LnMlpPolicy_sac
from stable_baselines.gail import generate_expert_traj
from env_script.env_mujoco import JacoMujocoEnv
from state_gen.state_generator import State_generator

#from env.env_real import Real
from argparser import ArgParser

class RL_controller:
    def __init__(self):
        # Arguments
        parser = ArgParser(isbaseline=True)
        args = parser.parse_args()

        # Debug
        args.debug = True
        print("DEBUG = ", args.debug)

        # TensorFlow Setting
        self.sess = tf_util.single_threaded_session()
        args.sess = self.sess

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
            self.model_path = "/home/ljh/Project/mujoco_jaco/src/models_baseline/"
            self.tb_dir = "/home/ljh/Project/mujoco_jaco/src/tensorboard_log"
        elif sys.platform == "darwin":
            self.model_path = "/Users/jeonghoon/Google_drive/Workspace/MLCS/mujoco_jaco/src/models_baseline/"
            self.tb_dir = "/Users/jeonghoon/Google_drive/Workspace/MLCS/mujoco_jaco/src/tensorboard_log"
        else:
            raise NotImplementedError
        args.model_path = self.model_path
        args.tb_dir = self.tb_dir
        self.steps_per_batch = 100
        self.batches_per_episodes = 5
        args.steps_per_batch = self.steps_per_batch
        args.batches_per_episodes = self.batches_per_episodes
        self.num_episodes = 10000
        self.train_num = 1

        self.args = args


    def train_from_scratch(self):
        print("Training from scratch called")
        self.env = JacoMujocoEnv(**vars(self.args))
        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * \
            math.ceil(self.num_episodes / self.train_num)
        # self.trainer = TRPO(MlpPolicy, self.env, cg_damping=0.1, vf_iters=5, vf_stepsize=1e-3, timesteps_per_batch=self.steps_per_batch,
        #                    tensorboard_log=args.tb_dir, full_tensorboard_log=True)
        self.trainer = SAC(
            LnMlpPolicy_sac, self.env, tensorboard_log=self.tb_dir, full_tensorboard_log=True)
        with self.sess:
            for train_iter in range(self.train_num):
                print("\033[91mTraining Iter: ", train_iter,"\033[0m")
                model_dir=self.model_path + str(self.train_num)
                os.makedirs(model_dir, exist_ok = True)
                self.trainer.learn(total_timesteps = self.num_timesteps)
                print("Train Finished")
                self.trainer.save(model_dir)
        
    def train_from_expert(self,n_episodes=10):
        print("Training from expert called")
        self.env = JacoMujocoEnv(**vars(self.args))
        generate_expert_traj(self._expert, 'expert_traj', self.env, n_episodes=n_episodes)
        
    def _expert(self,_obs):
        action = []
        return action
    
    def test(self):
        print("Testing called")
        self.env = JacoMujocoEnv(**vars(self.args))
        model_name = str(1) + ".zip"
        model_dir = self.model_path + model_name
        test_iter = 100
        with self.sess:
            self.model = SAC.load(model_dir)
            for _ in range(test_iter):
                obs = self.env.reset()
                done = False
                while not done:
                    action, state = self.model.predict(obs)
                    obs,rewards,done,_ = self.env.step(action,log=False)
                    print(rewards,end='\r')


if __name__ == "__main__":
    controller = RL_controller()
    iter = 0
    while True:
        opt = input("Train/Test (1/2): ")
        if opt == "1":
            iter_train = 0
            while True:
                iter_train += 1
                opt2 = input("Train_from_scratch/Train_from_expert (1/2): ")
                if opt2 == "1":
                    controller.train_from_scratch()
                    break
                elif opt2 == "2":
                    n_episodes = int(input("How many trials do you want to record?"))
                    controller.train_from_expert(n_episodes)
                    break
                else:
                    if iter_train <= 5:
                        print("Wront input, press 1 or 2 (Wrong trials: {0})".format(iter_train))
                    else:
                        print("Wront input, Abort")
                        break
            break
        elif opt == "2":
            controller.test()
            break
        else:
            iter += 1
            if iter <= 5:
                print("Wront input, press 1 or 2 (Wrong trials: {0})".format(iter))
            else:
                print("Wront input, Abort")
                break

