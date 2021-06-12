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

import tensorflow as tf
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
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from env_script.env_mujoco import JacoMujocoEnv
from state_gen.state_generator import State_generator

from argparser import ArgParser

def open_connection():
    try:            
        print("Opening connection to SpaceNav driver ...")
        spacenav.open()
        print("... connection established.")
    except Exception:
        print("No connection to the SpaceNav driver. Is spacenavd running?")

def close_connection():
    atexit.register(spacenav.close)
    

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
        # self.reward_method = None
        # self.reward_module = None
        # args.reward_method = self.reward_method
        # args.reward_module = self.reward_module

        # Action
        self.g_angle = 0
        self.g_changed = None
        # 0:Left - Open, 1:Right - Close
        self.pressed = {0:False, 1:False}

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
        self.trial = 74

    def train_HPC(self):
        task_list = ['picking', 'placing', 'pickAndplace']
        composite_primitive_name = self.args.task = task_list[2]

        self.args.train_log = False
        self.args.visualize = True
        self.args.robot_file = "jaco2_dual_torque"
        self.args.controller = True
        self.args.n_robots = 2
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        self.args.auxiliary = False
        self.args.seed = self.trial

        env = JacoMujocoEnv(**vars(self.args))
        policy = MlpPolicy_hpcsac
        self.model = SAC_MULTI(policy=policy, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)
        save_interval = 10000
        ent_coef = 1e-7

        if self.args.auxiliary:
            prefix = composite_primitive_name + "_trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(
                    time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)
        else:
            prefix = composite_primitive_name +"_SAC_noaux_trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(
                    time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)
        prefix = 'HPCcheck_noQ'
        model_dir = self.model_path + prefix + "_" + str(self.trial)
        self.args.log_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
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
        #                                 loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
        #                                 load_value=False)

        # prim_name = 'grasping'
        # # policy_zip_path = self.model_path+"grasping_trained_at_12_28_17:26:27_15/continue1/policy_2330000.zip"
        # policy_zip_path = self.model_path+'comparison_observation_range_sym_discard_0/policy_8070000.zip'
        # self.model.construct_primitive_info(name=prim_name, freeze=True, level=1,
        #                                 obs_range=None, obs_index=[0, 1,2,3,4,5,6, 7, 8,9,10], 
        #                                 act_range=None, act_index=[0,1,2,3,4,5, 6], act_scale=1,
        #                                 obs_relativity={},
        #                                 layer_structure=None,
        #                                 loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
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
        policy_zip_path = self.model_path+'picking_sac_noaux_trained_at_2021_4_2_21:41_54/policy_4200000.zip'
        self.model.construct_primitive_info(name=prim_name, freeze=True, level=2,
                                        obs_range=None, obs_index=[0, 1,2,3,4,5,6, 7, 8,9,10, 17,18,19,20,21,22], 
                                        act_range=None, act_index=[0,1,2,3,4,5, 6], act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                        load_value=False)

        prim_name = 'placing'
        policy_zip_path = self.model_path+'placing_sac_noaux_trained_at_2021_4_22_16:15_59/policy_120000.zip'
        self.model.construct_primitive_info(name=prim_name, freeze=True, level=2,
                                        obs_range=None, obs_index=[0, 1,2,3,4,5,6, 7, 8,9,10, 14,15,16, 23,24,25], 
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
        self.model.construct_primitive_info(name='weight', freeze=False, level=2,
                                        obs_range=0, obs_index=obs_idx,
                                        act_range=0, act_index=list(range(number_of_primitives)), act_scale=None,
                                        obs_relativity={},
                                        layer_structure={'policy':[256, 256, 256],'value':[256, 256, 256]},
                                        subgoal=subgoal_dict)

        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes 
        model_dict = {'gamma': 0.99, 'tensorboard_log': model_dir,'verbose': 1, 'seed': self.args.seed, \
            'learning_rate':_lr_scheduler, 'learning_starts':0, 'ent_coef': ent_coef, 'batch_size': 8, 'noptepochs': 4, 'n_steps': 128}
        self.model.pretrainer_load(model=self.model, policy=policy, env=env, **model_dict)
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.model.learn(total_timesteps=self.num_timesteps, save_interval=save_interval, save_path=model_dir)
        print("\033[91mTrain Finished\033[0m")
        self.model.save(model_dir+"/policy", hierarchical=True)

    def train_HPC_continue(self):
        task_list = ['picking', 'placing', 'pickAndplace']
        composite_primitive_name = self.args.task = task_list[1]
        algo_list = ['sac','ppo']
        algo = algo_list[0]

        self.args.train_log = False
        self.args.visualize = True
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.controller = True
        self.args.n_robots = 1
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        self.args.auxiliary = False
        self.args.seed = self.trial

        if algo == 'sac':
            env = JacoMujocoEnv(**vars(self.args))
            policy = MlpPolicy_hpcsac
            self.model = SAC_MULTI(policy=policy, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)
            save_interval = 10000
            ent_coef = 1e-7
            # ent_coef = 'auto'
        elif algo == 'ppo':
            env_list = []
            for i in range(2):
                env_list.append(JacoMujocoEnv)
            env = DummyVecEnv(env_list, dict(**vars(self.args)))
            # env = SubprocVecEnv(env_list)
            env = VecNormalize(env, norm_obs=False, norm_reward=False)
            policy = MlpPolicy_hpcppo
            self.model = HPCPPO(policy=policy, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)
            save_interval = 100
            ent_coef = 0

        model_dir = self.model_path + 'HPCtest_46/SACS_1e-7'
        self.args.log_dir = model_dir
        sub_dir = '/finetune1'
        print("\033[92m"+model_dir + sub_dir+"\033[0m")
        os.makedirs(model_dir+sub_dir, exist_ok=True)


        # Weight definition
        obs_idx = [0, 1,2,3,4,5,6, 7, 8,9,10, 17,18,19,20,21,22]
        act_idx = [0,1,2,3,4,5, 6]
        policy_zip_path = model_dir+'/policy_260000.zip'
        self.model.construct_primitive_info(name='continue', freeze=False, level=1,
                                        obs_range=None, obs_index=obs_idx,
                                        act_range=None, act_index=act_idx, act_scale=1,
                                        obs_relativity={},
                                        layer_structure=None,
                                        loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
                                        load_value=True)

        self.num_timesteps = self.steps_per_batch * self.batches_per_episodes * self.num_episodes 
        model_dict = {'gamma': 0.99, 'tensorboard_log': model_dir+sub_dir,'verbose': 1, 'seed': self.args.seed, \
            'learning_rate':_lr_scheduler, 'learning_starts':100, 'ent_coef': ent_coef, 'batch_size': 8, 'noptepochs': 4, 'n_steps': 128}
        self.model.pretrainer_load(model=self.model, policy=policy, env=env, **model_dict)
        self._write_log(model_dir, info)
        print("\033[91mTraining Starts\033[0m")
        self.model.learn(total_timesteps=self.num_timesteps, save_interval=save_interval, save_path=model_dir+sub_dir)
        print("\033[91mTrain Finished\033[0m")
        self.model.save(model_dir+"/policy",hierarchical=True)

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
        self.args.visualize = False
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.n_robots = 1
        self.args.seed = 42

        #                 0           1          2          3            4           5          6             7
        task_list = ['reaching', 'grasping', 'picking', 'carrying', 'releasing', 'placing', 'pushing', 'pickAndplace']
        self.args.task = task_list[1]
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        
        if self.args.task == 'reaching':
            # traj_dict = np.load(self.model_path+'trajectories/'+self.args.task+"2.npz", allow_pickle=True)
            # self.args.init_buffer = np.array(traj_dict['obs'])
            self.args.robot_file = "jaco2_reaching_torque"
        
        env = JacoMujocoEnv(**vars(self.args))


        ##### Grasping #####
        # Upper grasp
        prefix = self.args.task + '_trained_at_12_28_17:26:27_15/continue1/policy_2330000.zip'
        # Side grasp (better)
        # prefix = 'comparison_observation_range_sym_discard_0/policy_8070000.zip'
        # Side grasp
        # prefix = 'comparison_observation_range_sym_nobuffer_2/policy_4330000.zip'

        ##### Reaching #####
        # prefix = self.args.task + '_trained_at_1_13_17:47:15_31/continue1/policy_3860000.zip'

        ##### Picking #####
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_4_2_21:41_54/policy_4200000.zip'

        ##### Releasing #####
        # prefix = self.args.task + '_trained_at_4_18_22:26:14_58/continue1/policy_2070000.zip'
        # prefix = 'policies/releasing/policy_2070000.zip'

        ##### Placing #####
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_4_21_22:7_62/policy_1650000.zip'
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_5_26_15:2_59/policy_5000000.zip'
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_5_27_12:18_60/policy_360000.zip'
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_5_27_12:18_61/policy_1650000.zip'
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_5_27_12:18_62/policy_1650000.zip'
        # prefix = 'policies/'+self.args.task + '/policy_120000.zip'
    
        ##### pickAndplace #####
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_4_23_16:3_74/policy_8220000.zip'


        model_dir = self.model_path + prefix
        test_iter = 20
        if self.args.task in ['picking','placing','pickAndplace']:
            self.model = SAC_MULTI(policy=MlpPolicy_hpcsac, env=None, _init_setup_model=False, composite_primitive_name='picking')
            obs_idx = [0, 1,2,3,4,5,6, 7, 8,9,10, 17,18,19,20,21,22, 23,24,25]
            act_idx = [0,1,2,3,4,5, 6]
            self.model.construct_primitive_info(name=None, freeze=True, level=1,
                                                obs_range=None, obs_index=obs_idx,
                                                act_range=None, act_index=act_idx, act_scale=1,
                                                obs_relativity={},
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(model_dir), 
                                                load_value=True)
            SAC_MULTI.pretrainer_load(self.model, MlpPolicy_hpcsac, env)
        else:
            self.model = SAC_MULTI.load(model_dir, MlpPolicy_hpcsac, env)

        total_iter = 0
        for ti in range(test_iter):
            iter = 0
            obs = env.reset()
            done = False
            # logger_path = "./logger_csv"
            # capture_path = './captures'
            # os.makedirs(logger_path, exist_ok=True)
            # os.makedirs(capture_path, exist_ok=True)
            # if self.args.task == 'pickAndplace':
            #     logger_name = "/weight_pickAndplace_"+str(ti)+".csv"
            #     label_string = 'Total_iter,Step,Picking,Placing,Pick-Reaching,Pick-Grasping,Place-Reaching,Place-Releasing'
            # elif self.args.task == 'picking':
            #     logger_name = "/weight_picking_"+str(ti)+".csv"
            #     label_string = 'Total_iter,Step,Reaching,Grasping'
            # elif self.args.task == 'placing':
            #     logger_name = "/weight_placing_"+str(ti)+".csv"
            #     label_string = 'Total_iter,Step,Reaching,Releasing'
            # file_logger = open(logger_path+logger_name, 'w')
            # file_logger.writelines(label_string+"\n")
            steps = []
            weight_pick = []
            weight_place = []
            weight_pick_reach = []
            weight_pick_grasp = []
            weight_place_reach = []
            weight_place_release = []

            while not done:
                total_iter += 1
                iter += 1
                if self.args.task in ['picking','placing','pickAndplace']:
                    action, subgoal, weight = self.model.predict_subgoal(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action, log=False, weight=weight, subgoal=subgoal)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action, log=False)
                pickAndplace = weight['level2_pickAndplace/weight'][0]
                pick = pickAndplace[0]
                place = pickAndplace[1]
                weight_pick.append(pick)
                weight_place.append(place)
                weight_pick_reach.append(weight['level1_picking/weight'][0][0] * pick)
                weight_pick_grasp.append(weight['level1_picking/weight'][0][1] * pick)
                weight_place_reach.append(weight['level1_placing/weight'][0][0] * place)
                weight_place_release.append(weight['level1_placing/weight'][0][1] * place)
                # if self.args.task == 'pickAndplace':
                #     pickAndplace = weight['level2_pickAndplace/weight'][0]
                #     pick = weight['level1_picking/weight'][0] * pickAndplace[0]
                #     place = weight['level1_placing/weight'][0] * pickAndplace[1]
                #     weight_string = "{0:3f},{1:3f},{2:3f},{3:3f},{4:3f},{5:3f}".format(pickAndplace[0], pickAndplace[1], pick[0], pick[1], place[0], place[1])
                # elif self.args.task == 'picking':
                #     pick = weight['level1_picking/weight'][0]
                #     weight_string = "{0:3f},{1:3f}".format(pick[0], pick[1])
                # elif self.args.task == 'placing':
                #     place = weight['level1_placing/weight'][0]
                #     weight_string = "{0:3f},{1:3f}".format(place[0], place[1])
                # log_string = "{0:4d},{0:4d},".format(total_iter,iter) + weight_string
                # print(log_string, type(log_string))
                # file_logger.writelines(log_string+"\n")
                # env.set_capture_path(capture_path+'/'+self.args.task+str(total_iter)+"_%07d.png")
                # if done:
                #     pass
                #     env.capture()
            # file_logger.close()


if __name__ == "__main__":
    controller = RL_controller()
    # controller.train_HPC()
    # controller.train_HPC_continue()
    controller.test()