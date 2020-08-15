import os
import time
import path_config
from pathlib import Path
from collections import OrderedDict

import numpy as np

from env_script.Environment.manipulator_2d import Manipulator2D
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_sac

env = Manipulator2D()

package_path = str(Path(__file__).resolve().parent.parent)
model_path = package_path+"/models_baseline/"
#prefix = "trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)
prefix = "twowheel"
model_dir = model_path + prefix
os.makedirs(model_dir, exist_ok=True)
train = False
load = not train
separate = False
auxilary = False and not separate

if train:
    if separate:
        n_actions = env.action_space.shape[-1]
        n_obs = env.observation_space.shape[-1]
        print("n_obs: ",n_obs)
        print("state: ",env._get_state())
        layers = {"policy": [32, 32], "value": [32, 32]}

        model = SAC_MULTI(MlpPolicy_sac, env, layers=layers)

        model.learn(100000)
        model.save(model_dir+"/linear")
    elif auxilary:
        primitives = OrderedDict()
        SAC_MULTI.construct_primitive_info(name='train/aux1', primitive_dict=primitives, 
                                            obs_dimension=2, obs_range=[float('inf'), np.pi], obs_index=[0, 1], 
                                            act_dimension=2, act_range=[-1, 1], act_index=[0, 1], 
                                            policy_layer_structure=[32, 32])
        policy_zip_path = model_path+"twowheel"+"/linear.zip"
        SAC_MULTI.construct_primitive_info('freeze/loaded/linear', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1], 
                                            act_dimension=None, act_range=None, act_index=[0, 1], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=True)
        policy_zip_path = model_path+"twowheel"+"/angular.zip"    
        SAC_MULTI.construct_primitive_info('freeze/loaded/angular', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1], 
                                            act_dimension=None, act_range=None, act_index=[0, 1], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=True)
        number_of_primitives = 3
        total_obs_dim = env.get_num_observation()
        SAC_MULTI.construct_primitive_info('train/weight', primitives, 
                                            total_obs_dim, 0, list(range(total_obs_dim)), 
                                            number_of_primitives, [0,1], number_of_primitives, 
                                            [64, 64])
        model = SAC_MULTI.pretrainer_load(policy=MlpPolicy_sac, primitives=primitives, env=env, separate_value=True)
        print("\033[91mTraining Starts\033[0m")
        model.learn(100000)
        print("\033[91mTrain Finished\033[0m")
        model.save(model_dir+"/policy", hierarchical=True)

    else:
        # TODO: add sub-primitive layer # of each policies to their name
        # -> Needed for constructing the policy structure
        primitives = OrderedDict()
        policy_zip_path = model_path+"twowheel"+"/linear.zip"
        SAC_MULTI.construct_primitive_info('freeze/loaded/linear', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1], 
                                            act_dimension=None, act_range=None, act_index=[0, 1], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=True)
        policy_zip_path = model_path+"twowheel"+"/angular.zip"    
        SAC_MULTI.construct_primitive_info('freeze/loaded/angular', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1], 
                                            act_dimension=None, act_range=None, act_index=[0, 1], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=True)
        number_of_primitives = 2
        total_obs_dim = 2
        SAC_MULTI.construct_primitive_info('train/weight', primitives, 
                                            total_obs_dim, 0, list(range(total_obs_dim)), 
                                            number_of_primitives, [0,1], number_of_primitives, 
                                            [64, 64])
        model = SAC_MULTI.pretrainer_load(policy=MlpPolicy_sac, primitives=primitives, env=env, separate_value=True)
        quit()
        print("\033[91mTraining Starts\033[0m")
        model.learn(100000)
        print("\033[91mTrain Finished\033[0m")
        model.save(model_dir+"/policy", hierarchical=True)

        print("\033[91mTest Starts\033[0m")
        for i in range(10):
            print("\033[91mTest iter: {0}\033[0m".format(i))
            obs = env.reset()
            n_iter = 0
            while True:
                n_iter += 1
                action, state = model.predict(obs)
                weight = model.get_weight(obs)
                if n_iter % 20:
                    print("dist:\t",obs[0],"\tang:\t",obs[1],"\taction:\t",action,"\tweight:\t",weight)
                obs, reward, done, _ = env.step(action)
                if done:
                    break
            env.render()
        print("\033[91mTest Finished\033[0m")

if load:
    if separate:
        task = ['linear', 'angular']
        model = SAC_MULTI.load(model_path+"twowheel/"+task[1])
        obs = env.reset()

        while True:
            action, state = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        env.render()

    elif auxilary:
        primitives = OrderedDict()
        SAC_MULTI.construct_primitive_info(name='train/aux1', primitive_dict=primitives, 
                                            obs_dimension=2, obs_range=[float('inf'), np.pi], obs_index=[0, 1], 
                                            act_dimension=2, act_range=[-1, 1], act_index=[0, 1], 
                                            policy_layer_structure=[32, 32])
        policy_zip_path = model_path+"twowheel"+"/linear.zip"
        SAC_MULTI.construct_primitive_info('freeze/loaded/linear', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1], 
                                            act_dimension=None, act_range=None, act_index=[0, 1], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=True)
        policy_zip_path = model_path+"twowheel"+"/angular.zip"    
        SAC_MULTI.construct_primitive_info('freeze/loaded/angular', primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1], 
                                            act_dimension=None, act_range=None, act_index=[0, 1], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=True)
        number_of_primitives = 3
        total_obs_dim = env.get_num_observation()
        SAC_MULTI.construct_primitive_info('train/weight', primitives, 
                                            total_obs_dim, 0, list(range(total_obs_dim)), 
                                            number_of_primitives, [0,1], number_of_primitives, 
                                            [64, 64])
        model = SAC_MULTI.pretrainer_load(policy=MlpPolicy_sac, primitives=primitives, env=env, separate_value=True)
        obs = env.reset()

        while True:
            action, state = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        env.render()

    else:
        primitives = OrderedDict()
        policy_zip_path = model_path+"twowheel"+"/policy.zip"
        SAC_MULTI.construct_primitive_info(name=None, primitive_dict=primitives,
                                            obs_dimension=None, obs_range=None, obs_index=[0, 1], 
                                            act_dimension=None, act_range=None, act_index=[0, 1], 
                                            policy_layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), separate_value=True)
        model = SAC_MULTI.pretrainer_load(policy=MlpPolicy_sac, primitives=primitives, env=env, separate_value=True)
        obs = env.reset()

        while True:
            action, state = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        env.render()


