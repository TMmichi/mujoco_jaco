#!/usr/bin/env python

import os, time

from pathlib import Path

import numpy as np
import stable_baselines.common.tf_util as tf_util
from stable_baselines.composeNet import SACComposenet
from stable_baselines.composeNet.policies import ComposenetPolicy

from env_script.env_mujoco import JacoMujocoEnv

from argparser import ArgParser


def _lr_scheduler(frac):
    default_lr = 7e-5
    return default_lr * frac

class RL_controller:
    def __init__(self):
        # Arguments
        parser = ArgParser()
        args = parser.parse_args()

        self.sess_SRL = tf_util.single_threaded_session()
        args.sess = self.sess_SRL

        # If resume training on pre-trained models with episodes, else None
        package_path = str(Path(__file__).resolve().parent)
        self.model_path = package_path+"/models_baseline/"
        os.makedirs(self.model_path, exist_ok=True)
        self.args = args


    def train_ComposeNet(self):
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True

        composite_primitive_name = self.args.task
        env = JacoMujocoEnv(**vars(self.args))
        self.model = SACComposenet(policy=ComposenetPolicy,
                                    env=env,
                                    _init_setup_model=False)

        if self.args.auxiliary:
            prefix = composite_primitive_name \
                    + "_trained_at_"
        else:
            prefix = 'ComposeNet/'+composite_primitive_name \
                    + "_fixedcoef_trained_at_"
        prefix += str(time.localtime().tm_year) + "_" \
                + str(time.localtime().tm_mon) + "_" \
                + str(time.localtime().tm_mday) + "_" \
                + str(time.localtime().tm_hour) + ":" \
                + str(time.localtime().tm_min)
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

        model_dict = {'tensorboard_log': model_dir, 'verbose': 1, 'seed': self.args.seed,
                        'gamma': 0.99, 'learning_rate':_lr_scheduler, 'learning_starts':10000, 
                        'ent_coef': self.args.ent_coef, 'batch_size': 8, 'noptepochs': 4, 'n_steps': 128}
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
            self.model = SACComposenet.load(model_dir, MlpPolComposenetPolicyicy, env)


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


if __name__ == "__main__":
    controller = RL_controller()
    controller.train_ComposeNet()
    # controller.test()
