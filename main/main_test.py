import os
import time
import path_config
from pathlib import Path
from collections import OrderedDict

import numpy as np

from env_script.test_env.manipulator_2d import Manipulator2D
from stable_baselines.sac import SAC
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_sac


package_path = str(Path(__file__).resolve().parent.parent)
model_path = package_path+"/models_baseline/"
tb_dir = package_path+"/tensorboard_log/"
#prefix = "trained_at_" + str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + ":" + str(time.localtime().tm_min)
prefix = "twowheel"
model_dir = model_path + prefix
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True)
train = True
load = not train
separate = False
test = False and not separate
auxilary = True and not separate

env = Manipulator2D(action='fused')

if train:
    if separate:
        del env
        action = 'linear'
        #action = 'angular'
        #action = 'fused'
        env = Manipulator2D(action=action)

        n_actions = env.action_space.shape[-1]
        n_obs = env.observation_space.shape[-1]
        layers = {"policy": [128, 128], "value": [128, 128]}
        prefix2 = '/'+action+"_separate2"
        tb_path = tb_dir + prefix + prefix2
        total_time_step = 1000000
        learn_start = int(total_time_step*0.1)

        model = SAC_MULTI(MlpPolicy_sac, env, learning_starts=learn_start, layers=layers, tensorboard_log=tb_path, verbose=1)

        print("\033[91mTraining Starts, action: {0}\033[0m".format(action))
        save_path = model_dir+prefix2
        os.makedirs(save_path, exist_ok=True)
        model.learn(total_time_step, save_interval=10000, save_path=save_path)
        print("\033[91mTraining finished\033[0m")

    elif auxilary:
        composite_primitive_name='Pose_control'
        model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

        aux1_obs_range = {'min': [-float('inf'), -np.pi, -np.pi], \
                          'max': [float('inf'), np.pi, np.pi]}
        aux1_act_range = {'min': [-1, -np.pi], \
                          'max': [1, np.pi]}
        model.construct_primitive_info(name='aux1', freeze=False, level=1,
                                            # obs_dimension=3, obs_range=[float('inf'), np.pi], obs_index=[0, 1, 2],
                                            obs_range=aux1_obs_range, obs_index=[0, 1, 2],
                                            act_range=aux1_act_range, act_index=[0, 1],
                                            layer_structure={'policy':[64, 64]})
        policy_zip_path = model_path+"twowheel/linear_separate/policy_1000000.zip"
        model.construct_primitive_info(name='linear', freeze=True, level=1,
                                            obs_range=None, obs_index=[0, 1],
                                            act_range=None, act_index=[0],
                                            layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), load_value=True)
        policy_zip_path = model_path+"twowheel/angular_separate/policy_1000000.zip"
        model.construct_primitive_info(name='angular', freeze=True, level=1,
                                            obs_range=None, obs_index=[0, 1],
                                            act_range=None, act_index=[1],
                                            layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), load_value=True)
        total_obs_dim = 3
        number_of_primitives = 3
        model.construct_primitive_info(name='weight', freeze=False, level=1,
                                            obs_range=0, obs_index=list(range(total_obs_dim)),
                                            act_range=0, act_index=list(range(number_of_primitives)),
                                            layer_structure={'policy':[128, 128],'value':[128, 128]})
        prefix2 = '/MCP_aux_test_full'
        tb_path = tb_dir + prefix+prefix2
        total_time_step = 1000000
        learn_start = int(total_time_step*0.1)

        model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env,batch_size=5,
                                            buffer_size=100000, learning_starts=learn_start, tensorboard_log=tb_path, ent_coef='auto', verbose=1)#, tensorboard_log=tb_path)
        print("\033[91mTraining Starts\033[0m")
        save_path = model_dir+prefix2
        os.makedirs(save_path, exist_ok=True)
        model.learn(total_time_step, save_interval=1000000, save_path=save_path)
        print("\033[91mTrain Finished\033[0m")

    elif test:
        composite_primitive_name='Pose_control2'
        model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

        aux1_obs_range = {'min': [-float('inf'), -np.pi, -np.pi], \
                          'max': [float('inf'), np.pi, np.pi]}
        aux1_act_range = {'min': [-1, -np.pi], \
                          'max': [1, np.pi]}
        model.construct_primitive_info(name='aux1', freeze=False, level=2,
                                            obs_range=aux1_obs_range, obs_index=[0, 1, 2],
                                            act_range=aux1_act_range, act_index=[0, 1],
                                            layer_structure={'policy':[32, 32], 'value':[32,32]})
        policy_zip_path = model_path+"twowheel/MCP_aux_test/policy_10.zip"
        model.construct_primitive_info(name='posectrl', freeze=True, level=2,
                                            obs_range=None, obs_index=[0, 1, 2],
                                            act_range=None, act_index=[0, 1],
                                            layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), load_value=True)
        total_obs_dim = 3
        number_of_primitives = 2
        model.construct_primitive_info(name='weight', freeze=False, level=2,
                                            obs_range=0, obs_index=list(range(total_obs_dim)),
                                            act_range=0, act_index=list(range(number_of_primitives)),
                                            layer_structure={'policy':[32, 32],'value':[64, 64]})
        total_time_step = 10
        learn_start = int(total_time_step*0.1)

        model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env,
                                            buffer_size=100000, learning_starts=learn_start, ent_coef='auto', verbose=1)#, tensorboard_log=tb_path)
        print("\033[91mTraining Starts\033[0m")
        save_path = model_dir+"/MCP_aux_test2"
        os.makedirs(save_path, exist_ok=True)
        model.learn(total_time_step, save_interval=10, save_path=save_path)
        print("\033[91mTrain Finished\033[0m")

    else:
        composite_primitive_name='MCP'
        model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

        policy_zip_path = model_path+"twowheel"+"/linear2_SAC.zip"
        model.construct_primitive_info(name='linear', freeze=False, level=1,
                                            obs_range=None, obs_index=[0, 1],
                                            act_range=None, act_index=[0],
                                            layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), load_value=True)
        policy_zip_path = model_path+"twowheel"+"/angular.zip"    
        model.construct_primitive_info(name='angular', freeze=True, level=1,
                                            obs_range=None, obs_index=[0, 1],
                                            act_range=None, act_index=[1],
                                            layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), load_value=True)
        total_obs_dim = 2
        number_of_primitives = 2
        model.construct_primitive_info(name='weight', freeze=False, level=1,
                                            obs_range=0, obs_index=list(range(total_obs_dim)),
                                            act_range=0, act_index=list(range(number_of_primitives)),
                                            layer_structure={'policy':[32, 32],'value':[64, 64]})
        tb_path = tb_dir + prefix
        total_time_step = 10
        learn_start = 0 #int(total_time_step*0.1)

        model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env,
                                            buffer_size=100000, learning_starts=learn_start, ent_coef='auto', verbose=1)#, tensorboard_log=tb_path)
        print("\033[91mTraining Starts\033[0m")
        save_path = model_dir+"/fused_aux"
        os.makedirs(save_path, exist_ok=True)
        model.learn(total_time_step, save_interval=10, save_path=save_path)
        print("\033[91mTrain Finished\033[0m")

        print("\033[91mTest Starts\033[0m")
        for i in range(0):
            print("\033[91mTest iter: {0}\033[0m".format(i))
            obs = env.reset()
            n_iter = 0
            while True:
                n_iter += 1
                action, state = model.predict(obs)
                #weight = model.get_weight(obs)
                weight=[[0,0,0]]
                if n_iter % 20:
                    print("dist:\t{0: 2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\taction:\t[{0: 2.3f} {1: 2.3f}]".format(action[0],action[1]),"\tweight:\t",weight[0])
                obs, reward, done, _ = env.step(action, weight[0], test=True)
                if done:
                    break
            env.render()
        print("\033[91mTest Finished\033[0m")

if load:
    if separate:
        del env
        action_type = 'fused'
        env = Manipulator2D(action=action_type)

        task = ['linear_separate', 'angular_separate', 'fused_separate', 'fused_separate2']
        step_num = 1000000
        layers = {"policy": [128, 128], "value": [128, 128]}
        #layers = {"policy": [64, 64], "value": [64, 64]}
        model = SAC_MULTI.load(model_path+"twowheel/"+task[3]+"/policy_"+str(step_num), layers=layers)

        print("\033[91mTest Starts\033[0m")
        for i in range(10):
            print("\033[91mTest iter: {0}\033[0m".format(i))
            obs = env.reset()
            n_iter = 0
            while True:
                n_iter += 1
                action, state = model.predict(obs)
                prim_act = model.get_primitive_action(obs)
                prim_log_std = model.get_primitive_log_std(obs)
                print("action:\t",prim_act)
                print("log_std:",prim_log_std)
                if n_iter % 20:
                    if action_type == 'fused':
                        print("dist:\t{0:2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\ta_diff:\t{0: 2.3f}".format(obs[2]),"\taction:\t[{0: 2.3f} {1: 2.3f}]".format(action[0],action[1]))
                    else:
                        print("dist:\t{0:2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\taction:\t[{0:2.3f}]".format(action[0]))
                obs, reward, done, _ = env.step(action, test=True)
                if done:
                    break
            env.render()
        print("\033[91mTest Finished\033[0m")

    else:
        model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False)
        
        policy_zip_path = model_path+"twowheel/MCP_aux2/policy_1400000.zip"
        model.construct_primitive_info(name=None, freeze=True, level=1,
                                            obs_range=None, obs_index=[0, 1, 2], 
                                            act_range=None, act_index=[0, 1], 
                                            layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), load_value=True)
        model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env)

        print("\033[91mTest Starts\033[0m")
        for i in range(10):
            print("\033[91mTest iter: {0}\033[0m".format(i))
            obs = env.reset()
            n_iter = 0
            while True:
                n_iter += 1
                action, state = model.predict(obs)
                weight = model.get_weight(obs)['level1_Pose_control/weight']
                prim_act = model.get_primitive_action(obs)
                prim_log_std = model.get_primitive_log_std(obs)
                coef00 = weight[0][0] / np.exp(prim_log_std['level1_aux1'][0][0])
                coef01 = weight[0][0] / np.exp(prim_log_std['level1_aux1'][0][1])
                coef1 = weight[0][1] / np.exp(prim_log_std['level1_linear/level0'][0][0])
                coef2 = weight[0][2] / np.exp(prim_log_std['level1_angular/level0'][0][0])
                act_lin = np.tanh((coef00*prim_act['level1_aux1'][0][0] + coef1*prim_act['level1_linear/level0'][0][0]) / (coef00+coef1))
                act_ang = np.tanh((coef01*prim_act['level1_aux1'][0][1] + coef2*prim_act['level1_angular/level0'][0][0]) / (coef01+coef2))
                act_ang = act_ang * np.pi
                #weight=[[0,0,0]]
                if n_iter % 20 == 0:
                    print("action:\t",prim_act)
                    print("log_std:",prim_log_std)
                    print("dist:\t{0: 2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\taction:\t[{0: 2.3f} {1: 2.3f}]".format(action[0],action[1]),"\tweight:\t",weight[0])
                    print("calculated: ", act_lin, act_ang)
                obs, reward, done, _ = env.step(action, weight[0], test=True)
                if done:
                    break
            env.render()
        print("\033[91mTest Finished\033[0m")


