import os
import time
import path_config
from pathlib import Path
from collections import OrderedDict
#os.environ['CUDA_VISIBLE_DEVICES']='3'

import numpy as np

from env_script.test_env.manipulator_2d import Manipulator2D
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy


def _write_log(model_log, info):
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

package_path = str(Path(__file__).resolve().parent.parent)
model_path = package_path+"/models_baseline/"
prefix = "twowheel/"
model_dir = model_path + prefix
os.makedirs(model_dir, exist_ok=True)

train = True
load = not train
separate = False
scratch = True
test = False and not separate
auxilary = True and not separate

if train:
    if separate:
        action_option = ['linear', 'angular', 'fused', 'pickAndplace']
        action = action_option[0]
        trial = 19

        prefix2 = action+"_separate_trial"+str(trial)
        save_path = model_dir+prefix2
        os.makedirs(save_path, exist_ok=True)

        tol = 0.1
        n_robots = 1
        n_target = 1
        episode_length = 2000
        reward_method = 'target'
        #reward_method = 'time'
        #reward_method = None
        observation_method = 'absolute'
        #observation_method = 'relative'
        env = Manipulator2D(action=action, n_robots=n_robots, n_target=n_target, tol=tol, 
                        episode_length=episode_length, reward_method=reward_method, observation_method=observation_method,
                        policy_name=prefix2, visualize=False)

        #layers = {"policy": [256, 256, 128], "value": [256, 256, 128]}
        #layers = {"policy": [256, 256, 256], "value": [256, 256, 256]}
        #layers = {"policy": [64, 64, 64, 32], "value": [64, 64, 64, 32]}
        #layers = {"policy": [64, 64, 32, 32], "value": [64, 64, 32, 32]}
        #layers = {"policy": [256, 256, 128, 128], "value": [256, 256, 128, 128]}
        layers = {"policy": [128, 128, 64, 64, 32], "value": [128, 128, 64, 64, 32]}
        #layers = {"policy": [256, 256, 128, 128, 128, 64, 64], "value": [256, 256, 128, 128, 128, 64 ,64]}
        total_time_step = 5000000
        learn_start = int(total_time_step*0.05)
        ent_coef = 'auto'
        if scratch:
            model = SAC_MULTI(MlpPolicy_sac, env, learning_starts=learn_start, layers=layers, tensorboard_log=save_path, ent_coef=ent_coef, verbose=1)
        else:
            policy_num = 3750000
            path = save_path+'/policy_'+str(policy_num)
            print("loaded_policy_path: ",path)
            model = SAC_MULTI.load(path, env=env, learning_starts=10000, layers=layers, tensorboard_log=save_path, ent_coef=ent_coef)

        print("\033[91mTraining Starts, action: {0}\033[0m".format(action))
        if scratch:
            model_log = open(save_path+"/model_log.txt", 'w')
            info = {'trial': trial, 'action': action, 'layers': layers, 'tolerance': tol, 'total time steps': total_time_step,\
                 'n_robots': n_robots, 'n_targets': n_target, 'episode_length': episode_length, 'reward_method': reward_method, 'observation_method': observation_method, 'ent_coef': ent_coef,\
                 'Additional Info': \
                     'Reset from random initial pos\n\
                      \t\tAgent roates a bit\n\
                      \t\tTarget also moves\n\
                      \t\tDeeper network'}
                     
            _write_log(model_log, info)
            model_log.close()
            model.learn(total_time_step, save_interval=int(total_time_step*0.05), save_path=save_path)
        else:
            model.learn(total_time_step, loaded_step_num=policy_num, save_interval=int(total_time_step*0.05), save_path=save_path)
        print("\033[91mTraining finished\033[0m")

    elif auxilary:
        trial = 20
        action_option = ['linear', 'angular', 'fused', 'pickAndplace']
        action = action_option[2]
        prefix2 = action+"_auxilary_trial"+str(trial)
        save_path = model_dir+prefix2
        os.makedirs(save_path, exist_ok=True)

        tol = 0.1
        n_robots = 1
        n_target = 1
        episode_length = 2000
        reward_method = 'target'
        #reward_method = 'time'
        #reward_method = None
        observation_method = 'absolute'
        #observation_method = 'relative'
        env = Manipulator2D(action=action, n_robots=n_robots, n_target=n_target, tol=tol, 
                        episode_length=episode_length, reward_method=reward_method, observation_method=observation_method, policy_name=prefix2)
        composite_primitive_name='PoseControl'
        model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

        if observation_method == 'absolute':
            aux1_obs_range = {'min': [-float('inf'), -float('inf'), -np.pi, -float('inf'), -float('inf'), -np.pi],
                              'max': [float('inf'), float('inf'), np.pi, float('inf'), float('inf'), np.pi]
                              }
            aux1_obs_index = list(range(6))
            aux1_act_range = {'min': [-1, -np.pi],
                              'max': [1, np.pi]
                              }
            aux1_act_index = list(range(2))
            total_obs_dim = 6
            prim_obs_index = list(range(6))
        elif observation_method == 'relative':
            aux1_obs_range = {'min': [-float('inf'), -np.pi, -np.pi],
                              'max': [float('inf'), np.pi, np.pi]
                              }
            aux1_obs_index = list(range(3))
            aux1_act_range = {'min': [-1, -np.pi],
                              'max': [1, np.pi]
                              }
            aux1_act_index = list(range(2))
            total_obs_dim = 3
            prim_obs_index = list(range(2))

        prim_name = 'aux1'
        model.construct_primitive_info(name=prim_name, freeze=False, level=1,
                                            obs_range=aux1_obs_range, obs_index=aux1_obs_index,
                                            act_range=aux1_act_range, act_index=aux1_act_index,
                                            layer_structure={'policy':[256, 256, 128, 128]})

        prim_name = 'linear'
        prim_trial = 17
        policy_num = 4750000
        policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
        model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                            obs_range=None, obs_index=prim_obs_index,
                                            act_range=None, act_index=[0],
                                            layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), load_value=True)
        
        prim_name = 'angular'
        prim_trial = 8
        policy_num = 5000000
        policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
        model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                            obs_range=None, obs_index=prim_obs_index,
                                            act_range=None, act_index=[1],
                                            layer_structure=None,
                                            loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), load_value=True)

        total_obs_dim = 6
        number_of_primitives = 3
        model.construct_primitive_info(name='weight', freeze=False, level=1,
                                            obs_range=0, obs_index=list(range(total_obs_dim)),
                                            act_range=0, act_index=list(range(number_of_primitives)),
                                            layer_structure={'policy':[256, 256, 128, 128],'value':[256, 256, 128, 128]})
        total_time_step = 10000000
        learn_start = int(total_time_step*0.01)
        batch_size = 256
        model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env, batch_size=batch_size,
                                            buffer_size=50000, learning_starts=learn_start, tensorboard_log=save_path, ent_coef='auto', verbose=1)
        print("\033[91mTraining Starts\033[0m")
        model_log = open(save_path+"/model_log.txt", 'w')
        info = {'trial': trial, 'action': action, 'layers': None, 'tolerance': tol, 'total time steps': total_time_step,\
                 'n_robots': n_robots, 'n_targets': n_target, 'episode_length': episode_length, 'reward_method': reward_method, 'observation_method': observation_method,\
                 'batch_size': batch_size,\
                 'Additional Info': \
                     '0.125* scaled rewards + linear prim trained with 0.125* scaled reward\n\
                     \t\tValue from pre-trained primitive'}
        _write_log(model_log, info)
        model_log.close()
        model.learn(total_time_step, save_interval=int(total_time_step*0.01), save_path=save_path)
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
        action_list = ['linear', 'angular', 'fused', 'pickAndplace']
        action_type = action_list[0]
        trial = 18

        tol = 0.1
        n_robots = 1
        n_target = 1
        episode_length = 1000
        reward_method = 'target'
        #reward_method = 'time'
        #reward_method = None
        observation_method = 'absolute'
        #observation_method = 'relative'
        env = Manipulator2D(action=action_type, n_robots=n_robots, n_target=n_target, tol=tol, 
                        episode_length=episode_length, reward_method=reward_method, observation_method=observation_method)

        policy_num = 5000000 #1500000
        #layers = {"policy": [256, 256, 128, 128, 64], "value": [256, 256, 128, 128, 64]}
        layers = {"policy": [256, 256, 128, 128], "value": [256, 256, 128, 128]}
        #layers = {"policy": [128, 64, 64, 32], "value": [128, 64, 64, 32]}
        #layers = {"policy": [64, 64, 32, 32], "value": [64, 64, 32, 32]}
        #layers = {"policy": [256, 256, 128], "value": [256, 256, 128]}
        #layers = {"policy": [128, 128, 128], "value": [128, 128, 128]}
        #layers = {"policy": [128, 128], "value": [128, 128]}
        #layers = {"policy": [256, 256], "value": [256, 256]}

        prefix2 = action_type+"_separate_trial"+str(trial)
        model = SAC_MULTI.load(model_path+prefix+prefix2+"/policy_"+str(policy_num), layers=layers)

        print("\033[91mTest Starts\033[0m")
        for i in range(10):
            print("  \033[91mTest iter: {0}\033[0m".format(i))
            obs = env.reset()
            n_iter = 0
            while True:
                n_iter += 1
                action, state = model.predict(obs, deterministic=True)
                prim_act = model.get_primitive_action(obs)
                prim_log_std = model.get_primitive_log_std(obs)
                if n_iter % 20:
                    if action_type == 'fused':
                        print("  dist:\t{0:2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\ta_diff:\t{0: 2.3f}".format(obs[2]),"\taction:\t[{0: 2.3f} {1: 2.3f}]".format(action[0],action[1]))
                    elif action_type in ['linear', 'angular']:
                        print("  dist:\t{0:2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\taction:\t[{0:2.3f}]".format(action[0]))
                    elif action_type == 'pickAndplace':
                        #print("  x, y, w: {0: 2.3f}, {1: 2.3f}, {2: 2.3f}".format(obs[0], obs[1], obs[2])," action: [{0: 2.3f}, {1: 2.3f}, {2: 2.3f}, {3: 2.3f}]".format(action[0],action[1],action[2],action[3]))
                        pass
                obs, reward, done, _ = env.step(action, test=True)
                if done:
                    print(reward)
                    break
            env.render()
        print("\033[91mTest Finished\033[0m")

    else:
        action_list = ['linear', 'angular', 'fused', 'pickAndplace']
        action_type = action_list[2]
        trial = 20
        prefix2 = action_type+"_auxilary_trial"+str(trial)

        tol = 0.1
        n_robots = 1
        n_target = 1
        episode_length = 2000
        reward_method = 'target'
        #reward_method = 'time'
        #reward_method = None
        observation_method = 'absolute'
        #observation_method = 'relative'
        env = Manipulator2D(action=action_type, n_robots=n_robots, n_target=n_target, tol=tol, 
                        episode_length=episode_length, reward_method=reward_method, observation_method=observation_method)
        composite_primitive_name='PoseControl'
        model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

        if observation_method == 'absolute':
            observation_index = list(range(6))
        elif observation_method == 'relative':
            observation_index = list(range(3))
        policy_num = 100000
        policy_zip_path = model_path+prefix+prefix2+"/policy_"+str(policy_num)+".zip"
        model.construct_primitive_info(name=None, freeze=True, level=1,
                                            obs_range=None, obs_index=observation_index,
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
                weight = model.get_weight(obs)['level1_PoseControl/weight']
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
                if n_iter % 1 == 0:
                    print("Mu:",end='\t')
                    for name, item in prim_act.items():
                        print(name, item, end='\t')
                    print()
                    print("Log_std",end='\t')
                    for name, item in prim_log_std.items():
                        print(name, item, end='\t')
                    print()
                    #print("action:\t",prim_act)
                    #print("log_std:",prim_log_std)
                    print("action:\t[{0: 2.3f} {1: 2.3f}]".format(action[0],action[1]),"\tweight:\t",weight[0])
                obs, reward, done, _ = env.step(action, weight[0], test=True)
                if done:
                    break
            env.render()
        print("\033[91mTest Finished\033[0m")


