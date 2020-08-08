#!/usr/bin/env python

import os
import sys
import time
import path_config
from pathlib import Path
from tkinter import *
from tkinter import simpledialog
from collections import OrderedDict
try:
    import spacenav, atexit
except Exception:
    pass

import numpy as np

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

        # Action
        self.g_angle = 0
        self.g_changed = None
        self.pressed = {0:False, 1:False}        # 0:Left - Open, 1:Right - Close

        # If resume training on pre-trained models with episodes, else None
        package_path = str(Path(__file__).resolve().parent.parent)
        self.model_path = package_path+"/models_baseline/"
        self.tb_dir = package_path+"/tensorboard_log/"
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.steps_per_batch = 100
        self.batches_per_episodes = 5
        args.steps_per_batch = self.steps_per_batch
        args.batches_per_episodes = self.batches_per_episodes
        self.num_episodes = 2
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
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.n_robots = 1
        env = JacoMujocoEnv(**vars(self.args))

        layers = {"policy": [64, 64], "value": [256, 256, 128]}
        #layers = None
        #self.trainer = TRPO(MlpPolicy, self.env, cg_damping=0.1, vf_iters=5, vf_stepsize=1e-3, timesteps_per_batch=self.steps_per_batch,
        #                   tensorboard_log=tb_path, full_tensorboard_log=True)
        #self.trainer = SAC(LnMlpPolicy_sac, env, layers=layers,
        #                   tensorboard_log=tb_path, full_tensorboard_log=True)
        self.trainer = SAC_MULTI(MlpPolicy_sac, env, layers=layers, 
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

    def train_from_expert(self, n_episodes=10, con_method=2):
        print("Training from expert called")
        self.args.train_log = False
        if con_method == 2:
            if sys.platform in ['linux', 'linux2']:
                try:            
                    print("Opening connection to SpaceNav driver ...")
                    spacenav.open()
                    print("... connection established.")
                    atexit.register(spacenav.close)
                    self.args.robot_file = "jaco2_curtain_torque"
                    env = JacoMujocoEnv(**vars(self.args))
                    generate_expert_traj(self._expert_3d, 'expert_traj',
                                        env, n_episodes=n_episodes)
                except spacenav.ConnectionError:
                    print("No connection to the SpaceNav driver. Is spacenavd running?")
            else:
                pass
        elif con_method == 1:
            self.args.robot_file = "jaco2_curtain_torque"
            env = JacoMujocoEnv(**vars(self.args))
            generate_expert_traj(self._expert_keyboard, 'expert_traj',
                                env, n_episodes=n_episodes)
        else:
            pass

    def _expert_3d(self, _obs):
        if sys.platform in ['linux', 'linux2']:
            event = spacenav.poll()
            if type(event) is spacenav.MotionEvent:
                action = np.array([event.x, event.z, event.y, event.rx, -event.ry, event.rz, self.g_angle, self.g_angle])/350*1.5
            elif type(event) is spacenav.ButtonEvent:
                print("button: ",event.button)
                print("pressed: ",event.pressed)
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
                self.g_angle += 0.25
            elif self.pressed[1]:
                self.g_angle -= 0.25
            self.g_angle = np.clip(self.g_angle, 0, 10)
            
            action[6] = action[7] = self.g_angle
            spacenav.remove_events(1)
            if self.g_changed is not None and not self.g_changed:
                print("Removed")
                spacenav.remove_events(2)
                self.g_changed = None

            return action

        else:
            action = [0,0,0,0,0,0,0,0]
            return action
    
    def _expert_keyboard(self, _obs):
        self.action = [0,0,0,0,0,0]
        return self.action
    
    def _build_popup(self):
        self.root = Tk()
        self.root.title("Keyboard input")
        self.root.geometry("400x400")
        button_a = Button(self.root, text="a")
        button_a.bind("<a>", self._clicker)

    def _clicker(self, event):
        self.action
        pass

    def train_with_additional_layer(self):
        self.args.train_log = False
        env = JacoMujocoEnv(**vars(self.args))
        prefix = "trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(
                time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)
        model_dir = self.model_path + prefix
        os.makedirs(model_dir, exist_ok=True)
        tb_path = self.tb_dir + prefix

        primitives = OrderedDict()
        separate_value = True
        
        # Newly appointed primitives
        # name = "{'train', 'freeze'}/{'loaded', ''}/'primitive_name'"
        SAC_MULTI.construct_primitive_info(name='train/aux1', primitive_dict=primitives, 
                                            obs_dimension=6, obs_range=[-2, 2], obs_index=[0, 1, 2, 3, 4, 5], 
                                            act_dimension=4, act_range=[-1.4, 1.4], act_index=[0, 1, 2, 3], 
                                            policy_layer_structure=[256, 256])
        SAC_MULTI.construct_primitive_info('train/aux2', primitives, 
                                            6, [-2, 2], [0, 1, 2, 3, 4, 5], 
                                            2, [-1.4, 1.4], [4, 5], 
                                            [256, 128])
        SAC_MULTI.construct_primitive_info('train/aux3', primitives, 
                                            6, [-2, 2], [0, 1, 2, 3, 4, 5], 
                                            5, [-1.4, 1.4], [0, 1, 4, 5, 6], 
                                            [256, 128])
        # Pretrained primitives
        policy_zip_path = self.model_path+"test"+"/policy.zip"
        SAC_MULTI.construct_primitive_info('freeze/loaded/reaching', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1, 2, 3, 4, 5], 
                                            act_dimension=None, act_range=None, act_index=[0, 1, 2, 3, 4, 5], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=separate_value)

        policy_zip_path = self.model_path+"test2"+"/policy.zip"
        SAC_MULTI.construct_primitive_info('train/loaded/grasping', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1, 2, 3, 4, 5], 
                                            act_dimension=None, act_range=None, act_index=[0, 1, 2, 3, 4, 5], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=separate_value)
        # Weight definition  
        number_of_primitives = 5
        total_obs_dim = env.get_num_observation()
        SAC_MULTI.construct_primitive_info('train/weight', primitives, 
                                            total_obs_dim, 0, list(range(total_obs_dim)), 
                                            number_of_primitives, [0,1], number_of_primitives, 
                                            [512, 512, 512])

        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes 
        self.trainer = SAC_MULTI.pretrainer_load(policy=MlpPolicy_sac, primitives=primitives, env=env, separate_value=separate_value, tensorboard_log=tb_path)
        print("\033[91mTraining Starts\033[0m")
        self.trainer.learn(total_timesteps=self.num_timesteps)
        print("\033[91mTrain Finished\033[0m")
        self.trainer.save(model_dir+"/policy")

    def test(self, policy):
        print("Testing called")
        self.args.train_log = False
        self.env = JacoMujocoEnv(**vars(self.args))
        model_dir = self.model_path+policy+"/policy.zip"
        test_iter = 100
        self.model = SAC_MULTI.load(model_dir)
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
                    #n_episodes = int(input("How many trials do you want to record?"))
                    con_method = int(input("control method - (keyboard:1, 3d mouse: 2) "))
                    #controller.train_from_expert(n_episodes)
                    controller.train_from_expert(con_method=con_method)
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
