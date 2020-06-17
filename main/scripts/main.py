6#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import stable_baselines.common.tf_util as tf_util
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac import SAC
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_sac, LnMlpPolicy as LnMlpPolicy_sac
from state_gen.state_generator import State_generator
from env_script.env_mujoco import JacoMujocoEnv

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
        #self.model_path = "/home/ljh/Project/mujoco_jaco/src/models_baseline/"
        self.model_path = "/Users/jeonghoon/Google_drive/Workspace/MLCS/mujoco_jaco/src/models_baseline/"
        args.model_path = self.model_path
        #self.tb_dir = "/home/ljh/Project/mujoco_jaco/src/tensorboard_log"
        self.tb_dir = "/Users/jeonghoon/Google_drive/Workspace/MLCS/mujoco_jaco/src/tensorboard_log"
        args.tb_dir = self.tb_dir
        self.steps_per_batch = 100
        self.batches_per_episodes = 5
        args.steps_per_batch = self.steps_per_batch
        args.batches_per_episodes = self.batches_per_episodes
        self.num_episodes = 10000
        self.train_num = 1

        self.args = args


    def _train(self):
        print("Training service init")
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
    
    def _test(self):
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
            break
        elif opt == "2":
            controller._test()
            break
        else:
            iter += 1
            if iter <= 5:
                print("Wront input, press 1 or 2 (Wrong trials: {0})".format(iter))
            else:
                print("Wront input, Exit")
                break

