#!/usr/bin/env python

import path_config
from pathlib import Path
import matplotlib.pyplot as plt

import stable_baselines.common.tf_util as tf_util
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_hpcsac

from env_script.env_mujoco import JacoMujocoEnv

from argparser import ArgParser


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

        # If resume training on pre-trained models with episodes, else None
        package_path = str(Path(__file__).resolve().parent.parent)
        self.model_path = package_path+"/models_baseline/"
        
        self.steps_per_batch = 100
        self.batches_per_episodes = 5
        args.steps_per_batch = self.steps_per_batch
        args.batches_per_episodes = self.batches_per_episodes
        self.num_episodes = 20000
        self.args = args
        self.trial = 74


    def plot(self):
        self.args.train_log = False
        self.args.visualize = True
        self.args.robot_file = "jaco2_curtain_torque"
        self.args.n_robots = 1
        self.args.seed = 42

        task_list = ['picking','placing','pickAndplace']
        self.args.task = task_list[1]
        self.args.subgoal_obs = False
        self.args.rulebased_subgoal = True
        env = JacoMujocoEnv(**vars(self.args))

        ##### Picking #####
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_4_2_21:41_54/policy_4200000.zip'
        ##### Placing #####
        prefix = 'policies/'+self.args.task + '/policy_120000.zip'
        ##### pickAndplace #####
        # prefix = self.args.task + '_sac_noaux_trained_at_2021_4_23_16:3_74/policy_8220000.zip'


        model_dir = self.model_path + prefix
        test_iter = 20
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

        plt.ion()
        # fig, axes = plt.subplots(2,1, figsize=(10,6), sharex=True)
        # pick, = axes[0].plot([], [], color='red', linewidth=1, label='Picking')
        # place, = axes[0].plot([], [], color='mediumblue', linewidth=1, label='Placing')
        # pick_reach, = axes[1].plot([], [], color='dodgerblue', linestyle='-',linewidth=2, label='Pick-Reaching')
        # pick_grasp, = axes[1].plot([], [], color='yellowgreen', linestyle='-', linewidth=2, label='Pick-Grasping')
        # place_reach, = axes[1].plot([], [], color='aqua', linestyle='-', linewidth=2, label='Place-Reaching')
        # place_release, = axes[1].plot([], [], color='yellow', linestyle='-', linewidth=1, label='Place-Releasing')
        # axes[0].set_title('Intents of level 2 tasks')
        # axes[1].set_title('Intents of level 1 tasks')
        # axes[0].set_ylabel('Intents', fontsize=13)
        # axes[1].set_xlabel('Step', fontsize=13)
        # axes[1].set_ylabel('Intents', fontsize=13)
        # axes[0].grid(axis='y', alpha=0.5, linestyle='--')
        # axes[1].grid(axis='y', alpha=0.5, linestyle='--')
        # axes[0].legend(loc='right', fontsize=13)
        # axes[1].legend(loc='right', fontsize=13)
        # plt.xlim(0, 300)
        # axes[0].set_ylim(-0.05, 1.05)
        # axes[1].set_ylim(-0.05, 1.05)

        fig = plt.figure(figsize=(10,6))
        # pick_reach, = plt.plot([], [], color='dodgerblue', linestyle='-',linewidth=2, label='Pick-Reaching')
        # pick_grasp, = plt.plot([], [], color='yellowgreen', linestyle='-', linewidth=2, label='Pick-Grasping')

        place_reach, = plt.plot([], [], color='aqua', linestyle='-', linewidth=2, label='Place-Reaching')
        place_release, = plt.plot([], [], color='yellow', linestyle='-', linewidth=1, label='Place-Releasing')
        plt.xlabel('Step', fontsize=13)
        plt.ylabel('Intents', fontsize=13)
        plt.grid(axis='y', alpha=0.5, linestyle='--')
        plt.legend(loc='right', fontsize=13)
        plt.xlim(0, 100)
        plt.ylim(-0.05, 1.05)

        total_iter = 0
        for ti in range(test_iter):
            iter = 0
            obs = env.reset()
            done = False
            
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
                
                steps.append(iter)

                # pickAndplace = weight['level2_pickAndplace/weight'][0]
                # pick_val = pickAndplace[0]
                # place_val = pickAndplace[1]
                # weight_pick.append(pick_val)
                # weight_place.append(place_val)
                # weight_pick_reach.append(weight['level1_picking/weight'][0][0] * pick_val)
                # weight_pick_grasp.append(weight['level1_picking/weight'][0][1] * pick_val)
                # weight_place_reach.append(weight['level1_placing/weight'][0][0] * place_val)
                # weight_place_release.append(weight['level1_placing/weight'][0][1] * place_val)

                # weight_pick_reach.append(weight['level1_picking/weight'][0][0])
                # weight_pick_grasp.append(weight['level1_picking/weight'][0][1])

                weight_place_reach.append(weight['level1_placing/weight'][0][0])
                weight_place_release.append(weight['level1_placing/weight'][0][1])

                # pick.set_data(steps, weight_pick)
                # place.set_data(steps, weight_place)
                # pick_reach.set_data(steps, weight_pick_reach)
                # pick_grasp.set_data(steps, weight_pick_grasp)
                place_reach.set_data(steps, weight_place_reach)
                place_release.set_data(steps, weight_place_release)
                fig.canvas.draw()
                fig.canvas.flush_events()
                if iter > 300:
                    done = True


if __name__ == "__main__":
    controller = RL_controller()
    controller.plot()