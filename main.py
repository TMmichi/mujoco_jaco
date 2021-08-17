#!/usr/bin/env python

import os, time

from pathlib import Path
from configuration import model_configuration

import stable_baselines.common.tf_util as tf_util
from stable_baselines.hpc import HPC
from stable_baselines.hpc.policies import MlpPolicy

from env_script.env_mujoco import JacoMujocoEnv

from argparser import ArgParser
    

default_lr = model_configuration['learning_rate']
def _lr_scheduler(frac):
    return default_lr * frac

class RL_controller:
    def __init__(self):
        # Arguments
        parser = ArgParser()
        args = parser.parse_args()

        self.sess_SRL = tf_util.single_threaded_session()
        args.sess = self.sess_SRL

        # Action
        self.g_angle = 0
        self.g_changed = None
        # 0:Left - Open, 1:Right - Close
        self.pressed = {0:False, 1:False}

        # If resume training on pre-trained models with episodes, else None
        package_path = str(Path(__file__).resolve().parent)
        self.model_path = package_path+"/models_baseline/"
        os.makedirs(self.model_path, exist_ok=True)
        self.args = args


    def train_HPC(self):
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True

        composite_primitive_name = self.args.task
        env = JacoMujocoEnv(**vars(self.args))
        self.model = HPC(policy=MlpPolicy, 
                                env=None, 
                                _init_setup_model=False, 
                                composite_primitive_name=composite_primitive_name)

        if self.args.auxiliary:
            prefix = composite_primitive_name \
                    + "_trained_at_"
        else:
            prefix = composite_primitive_name \
                    + "_noaux_trained_at_"
        prefix += str(time.localtime().tm_year) + "_" \
                + str(time.localtime().tm_mon) + "_" \
                + str(time.localtime().tm_mday) + "_" \
                + str(time.localtime().tm_hour) + ":" \
                + str(time.localtime().tm_min)
        model_dir = self.model_path + prefix + "_" + str(self.args.seed)
        # os.makedirs(model_dir, exist_ok=True)
        print("\033[92m"+model_dir+"\033[0m")

        # Obs for Picking
        # obs_min = [-3, -1,-1,-1,-1,-1,-1, -1, -1,-1,-1, -1,-1,-1,-1,-1,-1]
        # obs_max = [ 3,  1, 1, 1, 1, 1, 1,  1,  1, 1, 1,  1, 1, 1, 1, 1, 1]
        # obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 17,18,19,20,21,22]

        # Obs for Placing
        # obs_min = [-3, -1,-1,-1,-1,-1,-1, -1, -1,-1,-1, -1,-1,-1, -1,-1,-1]
        # obs_max = [ 3,  1, 1, 1, 1, 1, 1,  1,  1, 1, 1,  1, 1, 1,  1, 1, 1]
        # obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 14,15,16, 23,24,25]

        # Obs for pickAndplace
        obs_min = [-3, -1,-1,-1,-1,-1,-1, -1, -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1, -1,-1,-1]
        obs_max = [ 3,  1, 1, 1, 1, 1, 1,  1,  1, 1, 1,  1, 1, 1,  1, 1, 1, 1, 1, 1,  1, 1, 1]
        obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 14,15,16, 17,18,19,20,21,22, 23,24,25]

        # Action
        act_min = [-1,-1,-1,-1,-1,-1, -1]
        act_max = [ 1, 1, 1, 1, 1, 1,  1]
        act_idx = [ 0, 1, 2, 3, 4, 5,  6]

        if self.args.auxiliary:
            self.model.construct_primitive_info(name='aux1', freeze=False, level=0,
                                            obs_range={'min': obs_min, 'max': obs_max}, obs_index=obs_idx, 
                                            act_range={'min': act_min, 'max': act_max}, act_index=act_idx, act_scale=0.1,
                                            obs_relativity={},
                                            layer_structure={'policy':[128, 128, 128]})

        ##### Pretrained primitives #####
        # prim_name = 'reaching'    # Reaching for Picking
        # policy_zip_path = self.model_path+'reaching_trained_at_1_13_17:47:15_31/continue1/policy_3860000.zip'
        # self.model.construct_primitive_info(name=prim_name, freeze=True, level=1,
        #                                 obs_range=None, obs_index=[1,2,3,4,5,6, 17,18,19,20,21,22],
        #                                 act_range=None, act_index=[0,1,2,3,4,5], act_scale=1,
        #                                 obs_relativity={'subtract':{'ref':[17,18,19,20,21,22],'tar':[1,2,3,4,5,6]}},
        #                                 layer_structure=None,
        #                                 loaded_policy=HPC._load_from_file(policy_zip_path),
        #                                 load_value=False)

        # prim_name = 'grasping'
        # policy_zip_path = self.model_path+'comparison_observation_range_sym_discard_0/policy_8070000.zip'
        # self.model.construct_primitive_info(name=prim_name, freeze=True, level=1,
        #                                 obs_range=None, obs_index=[0, 1,2,3,4,5,6, 7, 8,9,10], 
        #                                 act_range=None, act_index=[0,1,2,3,4,5, 6], act_scale=1,
        #                                 obs_relativity={},
        #                                 layer_structure=None,
        #                                 loaded_policy=HPC._load_from_file(policy_zip_path), 
        #                                 load_value=False)

        # prim_name = 'reaching_place'      # Reaching for Placing
        # policy_zip_path = self.model_path+'reaching_trained_at_1_13_17:47:15_31/continue1/policy_3860000.zip'
        # self.model.construct_primitive_info(name=prim_name, freeze=True, level=1,
        #                                 obs_range=None, obs_index=[1,2,3,4,5,6, 14,15,16, 23,24,25],
        #                                 act_range=None, act_index=[0,1,2,3,4,5], act_scale=1,
        #                                 obs_relativity={'subtract':{'ref':[14,15,16,23,24,25],'tar':[1,2,3,4,5,6]}},
        #                                 layer_structure=None,
        #                                 loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
        #                                 load_value=False)

        # prim_name = 'releasing'
        # policy_zip_path = self.model_path+'releasing_trained_at_4_18_22:26:14_58/continue1/policy_2070000.zip'
        # self.model.construct_primitive_info(name=prim_name, freeze=True, level=1,
        #                                 obs_range=None, obs_index=[0, 1,2,3,4,5,6, 7, 8,9,10, 14,15,16], 
        #                                 act_range=None, act_index=[0,1,2,3,4,5, 6], act_scale=1,
        #                                 obs_relativity={},
        #                                 layer_structure=None,
        #                                 loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
        #                                 load_value=False)

        prim_name = 'picking'
        policy_zip_path = self.model_path+'picking/policy.zip'
        self.model.construct_primitive_info(name=prim_name, freeze=True, level=2,
                                        obs_range=None, obs_index=[0, 1,2,3,4,5,6, 7, 8,9,10, 17,18,19,20,21,22], 
                                        act_range=None, act_index=[0,1,2,3,4,5, 6], act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=HPC._load_from_file(policy_zip_path), 
                                        load_value=False)

        prim_name = 'placing'
        policy_zip_path = self.model_path+'placing/policy.zip'
        self.model.construct_primitive_info(name=prim_name, freeze=True, level=2,
                                        obs_range=None, obs_index=[0, 1,2,3,4,5,6, 7, 8,9,10, 14,15,16, 23,24,25], 
                                        act_range=None, act_index=[0,1,2,3,4,5, 6], act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=HPC._load_from_file(policy_zip_path), 
                                        load_value=False)

        ################## Meta-Policy ##################
        number_of_primitives = 3 if self.args.auxiliary else 2
        if self.args.rulebased_subgoal:
            subgoal_dict = None
        else:
            subgoal_dict = {'level1_reaching/level0':[17,18,19,20,21,22]}
        self.model.construct_primitive_info(name='weight', freeze=False, level=3,
                                        obs_range=0, obs_index=obs_idx,
                                        act_range=0, act_index=list(range(number_of_primitives)), act_scale=None,
                                        obs_relativity={},
                                        layer_structure={'policy':[256, 256, 256],'value':[256, 256, 256]},
                                        subgoal=subgoal_dict)


        model_dict = {'tensorboard_log': model_dir, 'verbose': 1, 'seed': self.args.seed,
                        'gamma': 0.99, 'learning_rate':_lr_scheduler, 'learning_starts':0, 
                        'ent_coef': self.args.ent_coef, 'batch_size': 8, 'noptepochs': 4, 'n_steps': 128}
        self.model.pretrainer_load(model=self.model, env=env, **model_dict)
        print("\033[91mTraining Starts\033[0m")
        self.model.learn(total_timesteps=self.args.num_timesteps, save_interval=self.args.save_interval, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.model.save(model_dir+"/policy", hierarchical=True)

    def train_HPC_continue(self):
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True

        env = JacoMujocoEnv(**vars(self.args))
        self.model = HPC(policy=MlpPolicy, env=None, _init_setup_model=False)
        
        model_dir = self.model_path + self.args.task
        self.args.log_dir = model_dir
        sub_dir = '/finetune1'
        print("\033[92m"+model_dir + sub_dir+"\033[0m")
        os.makedirs(model_dir+sub_dir, exist_ok=True)

        # Weight definition
        obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 14,15,16, 17,18,19,20,21,22, 23,24,25]  #PickAndPlace
        # obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 17,18,19,20,21,22]    #Picking
        # obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 14,15,16, 23,24,25]   #Placing
        act_idx = [0,1,2,3,4,5, 6]
        
        policy_zip_path = model_dir+'/policy.zip'
        self.model.construct_primitive_info(name='continue', freeze=False, level=2,
                                        obs_range=None, obs_index=obs_idx,
                                        act_range=None, act_index=act_idx, act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=HPC._load_from_file(policy_zip_path),
                                        load_value=True)

        model_dict = {'verbose': 1, 'seed': self.args.seed,
                'gamma': 0.99, 'learning_rate':_lr_scheduler, 'learning_starts':0, 
                'ent_coef': self.args.ent_coef, 'batch_size': 8, 'noptepochs': 4, 'n_steps': 128}
        self.model.pretrainer_load(model=self.model, env=env, **model_dict)
        
        print("\033[91mTraining Starts\033[0m")
        self.model.learn(total_timesteps=self.args.num_timesteps, save_interval=self.args.save_interval, save_path=model_dir+sub_dir)
        print("\033[91mTrain Finished\033[0m")
        self.model.save(model_dir+sub_dir+"/policy_new",hierarchical=True)
        print("\033[91mPolicy Saved\033[0m")

    def test(self):
        print("Testing called")
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        env = JacoMujocoEnv(**vars(self.args))
        
        prefix = 'pickAndplace/policy.zip'
        # prefix = 'grasping/policy.zip'
        # prefix = 'picking/policy.zip'
        # prefix = 'placing/policy.zip'
        model_dir = self.model_path + prefix
        self.model = HPC(policy=MlpPolicy, env=None, _init_setup_model=False, composite_primitive_name=self.args.task)
        obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 14,15,16, 17,18,19,20,21,22, 23,24,25]  #PickAndPlace
        # obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 17,18,19,20,21,22]    #Picking
        # obs_idx = [ 0,  1, 2, 3, 4, 5, 6,  7,  8, 9,10, 14,15,16, 23,24,25]   #Placing
        act_idx = [0,1,2,3,4,5, 6]
        self.model.construct_primitive_info(name=None, freeze=True, level=3,
                                            obs_range=None, obs_index=obs_idx,
                                            act_range=None, act_index=act_idx, act_scale=1,
                                            obs_relativity={},
                                            layer_structure=None,
                                            loaded_policy=HPC._load_from_file(model_dir), 
                                            load_value=True)
        HPC.pretrainer_load(self.model, env)


        test_iter = 100
        success = 0
        for _ in range(test_iter):
            iter = 0
            obs = env.reset()
            done = False

            while not done:
                iter += 1
                action, subgoal, weight = self.model.predict_subgoal(obs, deterministic=True)
                obs, reward, done, _ = env.step(action, log=False, weight=weight, subgoal=subgoal)
                if reward > 100 and done:
                    success += 1
        print("Success rate: ",success/test_iter*100)


if __name__ == "__main__":
    controller = RL_controller()
    # controller.train_HPC()
    # controller.train_HPC_continue()
    controller.test()