#!/usr/bin/env python

import os, time, curses

from pathlib import Path

import numpy as np
import stable_baselines.common.tf_util as tf_util
from stable_baselines.hpc import HPC
from stable_baselines.hpc.policies import MlpPolicy
from stable_baselines.composeNet import SACComposenet
from stable_baselines.composeNet.policies import ComposenetPolicy

from env_script.env_mujoco import JacoMujocoEnv

from argparser import ArgParser


def _lr_scheduler(frac):
    default_lr = 7e-5
    return default_lr * frac

def set_prefix(model, task, net_size, emb):
    prefix = model+"/Netsize"+str(net_size)
    if emb:
        prefix += '_emb'
    prefix += '/' + task + '_at_'
    prefix += str(time.localtime().tm_year) + "_" \
            + str(time.localtime().tm_mon) + "_" \
            + str(time.localtime().tm_mday) + "_" \
            + str(time.localtime().tm_hour) + ":" \
            + str(time.localtime().tm_min)
    return prefix

def float_to_str(value):
    return '{:1.4f}'.format(value)

class RL_controller:
    def __init__(self):
        # Arguments
        parser = ArgParser()
        args = parser.parse_args()

        self.sess_SRL = tf_util.single_threaded_session()
        args.sess = self.sess_SRL

        self.act_dict = {'w': (1,1), 's': (1,-1), 'a': (0,-1), 'd': (0,1), 'q': (2,-1), 'e': (2,1), 
                        'j': (3,-1), 'l': (3,1), 'i':(4,-1), 'k':(4,1), 'u': (5,-1), 'o': (5,1)}

        # If resume training on pre-trained models with episodes, else None
        package_path = str(Path(__file__).resolve().parent)
        self.model_path = package_path+"/models_baseline/"
        os.makedirs(self.model_path, exist_ok=True)
        self.args = args


    def train_SAC(self):
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True

        prefix = 'baselines/'+self.args.task+"_trained_at_"
        prefix += str(time.localtime().tm_year) + "_" \
                + str(time.localtime().tm_mon) + "_" \
                + str(time.localtime().tm_mday) + "_" \
                + str(time.localtime().tm_hour) + ":" \
                + str(time.localtime().tm_min)
        model_dir = self.model_path + prefix + "_" + str(self.args.seed)
        os.makedirs(model_dir, exist_ok=True)
        print("\033[92m"+model_dir+"\033[0m")

        if self.args.task in ['reaching', 'reaching_GA']:
            traj_dict = np.load(self.model_path+'trajectories/reaching.npz', allow_pickle=True)
            self.args.init_buffer = np.array(traj_dict['obs'])
            self.args.robot_file = "jaco2_reaching_torque"
            obs_relativity = {'subtract':{'ref':[17,18,19,20,21,22],'tar':[1,2,3,4,5,6]}}
            obs_index = [1,2,3,4,5,6, 17,18,19,20,21,22]
        elif self.args.task == 'grasping':
            # buffer = self.create_buffer('trajectory_expert5_mod')
            obs_relativity = {}
            obs_index = [0, 1,2,3,4,5,6, 7, 8,9,10]
        elif self.args.task == 'releasing':
            obs_relativity = {}
            obs_index =  [0, 1,2,3,4,5,6, 7, 8,9,10, 14,15,16]


        env = JacoMujocoEnv(**vars(self.args))
        policy_kwargs = {'obs_relativity':obs_relativity, 'obs_index':obs_index}
        model_dict = {'gamma': 0.99, 'tensorboard_log': model_dir, 'policy_kwargs': policy_kwargs, 
                        'verbose': 1, 'learning_rate':_lr_scheduler, 'base_prim': True,
                        'layers': {'policy':[128,128,128], 'value':[128,128,128]}, }
        self.model = HPC(MlpPolicy, env, **model_dict)

        print("\033[91mTraining Starts\033[0m")
        self.num_timesteps = 10000000
        self.model.learn(total_timesteps=self.num_timesteps, save_interval=10000, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.model.save(model_dir+"/policy")


    def train_ComposeNet(self):
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True

        env = JacoMujocoEnv(**vars(self.args))
        self.model = SACComposenet(policy=ComposenetPolicy,
                                    env=env,
                                    use_embedding=self.args.use_embedding)
        prefix = set_prefix('ComposeNet', self.args.task, self.args.net_size, self.args.use_embedding)
        model_dir = self.model_path + prefix + "_" + str(self.args.seed)
        os.makedirs(model_dir, exist_ok=True)
        print("\033[92m"+model_dir+"\033[0m")

        ######### Pretrained primitives #########
        # primitives for picking & pick-and-place
        if not self.args.task == 'placing':
            prim_name = 'reaching_pick'
            policy_zip_path = self.model_path+'reaching/policy.zip'
            self.model.setup_skills(name=prim_name, obs_idx=[1,2,3,4,5,6, 17,18,19,20,21,22],
                                    obs_relativity={'subtract':{'ref':[17,18,19,20,21,22],'tar':[1,2,3,4,5,6]}},
                                    loaded_policy=SACComposenet._load_from_file(policy_zip_path))

            prim_name = 'grasping'
            policy_zip_path = self.model_path+'grasping/policy.zip'
            self.model.setup_skills(name=prim_name, obs_idx=[0, 1,2,3,4,5,6, 7, 8,9,10],
                                    obs_relativity={},
                                    loaded_policy=SACComposenet._load_from_file(policy_zip_path))
        # primitives for placing & pick-and-place
        if not self.args.task == 'picking':
            prim_name = 'reaching_place'
            policy_zip_path = self.model_path+'reaching/policy.zip'
            self.model.setup_skills(name=prim_name, obs_idx=[1,2,3,4,5,6, 14,15,16, 23,24,25],
                                    obs_relativity={'subtract':{'ref':[14,15,16,23,24,25],'tar':[1,2,3,4,5,6]}},
                                    loaded_policy=SACComposenet._load_from_file(policy_zip_path))
            
            prim_name = 'releasing'
            policy_zip_path = self.model_path+'releasing/policy.zip'
            self.model.setup_skills(name=prim_name, obs_idx=[0, 1,2,3,4,5,6, 7, 8,9,10, 14,15,16], 
                                    obs_relativity={},
                                    loaded_policy=SACComposenet._load_from_file(policy_zip_path))

        layers = {'compose_net': self.args.net_size, 'policy_net': self.args.net_size, 'value': [256,256]}
        model_dict = {'tensorboard_log': model_dir, 'layers': layers, 'verbose': 1, 'seed': self.args.seed,
                        'gamma': 0.99, 'learning_rate':_lr_scheduler, 'learning_starts':10000, 
                        'ent_coef': self.args.ent_coef, 'batch_size': 16, 'noptepochs': 4, 'n_steps': 128}
        self.model.__dict__.update(model_dict)
        self.model.setup_model()
        print("\033[91mTraining Starts\033[0m")
        self.model.learn(total_timesteps=self.args.num_timesteps, save_interval=self.args.save_interval, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.model.save(model_dir+"/policy")

    def test(self):
        print("Testing called")
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        if self.args.task == 'reaching':
            traj_dict = np.load(self.model_path+'trajectories/'+self.args.task+".npz", allow_pickle=True)
            self.args.init_buffer = np.array(traj_dict['obs'])
            self.args.rulebased_subgoal = False
            self.args.robot_file = "jaco2_reaching_torque"
        env = JacoMujocoEnv(**vars(self.args))
        
        model_dir = self.model_path + self.args.task + '/policy.zip'

        if self.args.task in ['picking', 'placing', 'pickAndplace']:
            self.model = SACComposenet(policy=ComposenetPolicy, 
                                env=None, 
                                _init_setup_model=False, 
                                composite_primitive_name=self.args.task)
            if self.args.task == 'pickAndplace':
                obs_idx = [0, 1,2,3,4,5,6, 7, 8,9,10, 17,18,19,20,21,22, 23,24,25]
            elif self.args.task == 'picking':
                obs_idx = [0, 1,2,3,4,5,6, 7, 8,9,10, 17,18,19,20,21,22]
            elif self.args.task == 'placing':
                obs_idx = [0, 1,2,3,4,5,6, 7, 8,9,10, 14,15,16, 23,24,25]
            act_idx = [0,1,2,3,4,5, 6]
            self.model.construct_primitive_info(name=None, freeze=True, level=3,
                                                obs_range=None, obs_index=obs_idx,
                                                act_range=None, act_index=act_idx, act_scale=1,
                                                obs_relativity={},
                                                layer_structure=None,
                                                loaded_policy=SACComposenet._load_from_file(model_dir), 
                                                load_value=True)
            SACComposenet.pretrainer_load(self.model, env)
        else:
            self.model = SACComposenet.load(model_dir, ComposenetPolicy, env)


        test_iter = 100
        success = 0
        for _ in range(test_iter):
            iter = 0
            obs = env.reset()
            done = False

            while not done:
                iter += 1
                if self.args.task in ['picking', 'placing', 'pickAndplace']:
                    action, subgoal, weight = self.model.predict_subgoal(obs, deterministic=False)
                    obs, reward, done, _ = env.step(action, weight=weight, subgoal=subgoal)
                else:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, done, _ = env.step(action)
                if reward > 100 and done:
                    success += 1
        print("Success rate: ",success/test_iter*100)


    def test_manual(self):
        print("Testing called")
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        if self.args.task in ['reaching', 'reaching_GA']:
            traj_dict = np.load(self.model_path+'trajectories/reaching.npz', allow_pickle=True)
            self.args.init_buffer = np.array(traj_dict['obs'])
            self.args.robot_file = "jaco2_reaching_torque"
        env = JacoMujocoEnv(**vars(self.args))

        win = curses.initscr()
        win.addstr(0, 0, 'Action input: ')

        test_iter = 100
        success = 0
        for _ in range(test_iter):
            iter = 0
            obs = env.reset()
            done = False
            while not done:
                iter += 1
                action = [0.0] * 7
                ch = win.getch()
                if ch in range(32, 127): 
                    act_key = chr(ch)
                else:
                    act_key = '1'
                win.timeout(10)
                try:
                    act_idx, act_val = self.act_dict[act_key]
                    action[act_idx] += act_val
                except:
                    pass
                obs, reward, done, target_pos = env.step(action)
                target_pos = list(map(float_to_str, target_pos))
                action = list(map(float_to_str, action))
                win.addstr(5, 0, 'Rewards: '+str(reward))
                win.addstr(7, 0, 'Actions: '+' '.join(action))
                win.addstr(10, 0, 'Target pos: '+' '.join(target_pos))


if __name__ == "__main__":
    controller = RL_controller()
    controller.train_SAC()
    # controller.train_ComposeNet()
    # controller.test()
    # controller.test_manual()
