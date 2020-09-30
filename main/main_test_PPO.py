import os
import time
import path_config
from pathlib import Path
from collections import OrderedDict
#os.environ['CUDA_VISIBLE_DEVICES']='3'

import numpy as np
import tensorflow as tf

from env_script.test_env.manipulator_2d import Manipulator2D
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_sac
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds


def _write_log(model_log, info):
    if info['layers'] != None:
        model_log.writelines("Layers:\n")
        model_log.write("\tpolicy:\t[")
        for i in range(len(info['layers']['pi'])):
            model_log.write(str(info['layers']['pi'][i]))
            if i != len(info['layers']['pi'])-1:
                model_log.write(", ")
            else:
                model_log.writelines("]\n")
        model_log.write("\tvalue:\t[")
        for i in range(len(info['layers']['vf'])):
            model_log.write(str(info['layers']['vf'][i]))
            if i != len(info['layers']['vf'])-1:
                model_log.write(", ")
            else:
                model_log.writelines("]\n\n")
        info.pop('layers')
    
    for name, item in info.items():
        model_log.writelines(name+":\t\t{0}\n".format(item))

def make_env(rank, seed=0, **kwargs):
    def _init():
        env = Manipulator2D(visualize=False, **kwargs)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    package_path = str(Path(__file__).resolve().parent.parent)
    model_path = package_path+"/models_baseline/"
    prefix = "twowheel/"
    model_dir = model_path + prefix
    os.makedirs(model_dir, exist_ok=True)

    train = True
    load = not train
    separate = True
    scratch = True
    test = False and not separate
    auxilary = True and not separate

    if train:
        if separate:
            action_option = ['linear', 'angular', 'fused', 'pickAndplace']
            observation_option = ['absolute', 'relative']
            reward_option = ['target', 'time', None]
            action = action_option[0]
            trial = 0

            prefix2 = action+"_separate_trial"+str(trial)
            save_path = model_dir+prefix2
            os.makedirs(save_path, exist_ok=True)

            n_robots = 1
            n_target = 1
            tol = 0.1
            episode_length = 2000
            reward_method = reward_option[0]
            observation_method = observation_option[0]
            info_dict = {'action': action, 'n_robots': n_robots, 'n_target':n_target, 'tol':tol, 
                        'episode_length':episode_length, 'reward_method':reward_method, 'observation_method':observation_method, 
                        'policy_name':prefix2}
            num_cpu = 8 #16
            env = SubprocVecEnv([make_env(i, **info_dict) for i in range(num_cpu)])
            #env = env = Manipulator2D(visualize=False, **info_dict)
            
            layer_structure_list = [[256, 256, 128, 128, 128, 64, 64], \
                                    [256, 256, 128, 128, 64], [128, 128, 64, 64, 32], [64, 128, 256, 128, 64], \
                                    [256, 256, 128, 128], [128, 64, 64, 32], [64, 64, 32, 32], \
                                    [512, 256, 256], [256, 256, 128], [128, 128, 128], \
                                    [256, 256], [128, 128]]
            layer_structure = layer_structure_list[7]
            net_arch = {"pi": layer_structure, "vf": layer_structure}
            policy_kwargs={'net_arch': [net_arch], 'act_fun': tf.nn.relu, 'squash':False, 'box_dist': 'beta'}

            total_time_step = 50000000
            model_dict = {'learning_rate': 5e-5, 'gamma':0.99, 'n_steps':4096, 'nminibatches': 256, 'cliprange': 0.02,
                            'tensorboard_log': save_path, 'policy_kwargs': policy_kwargs}
            if scratch:
                model = PPO2(MlpPolicy, env, **model_dict)
            else:
                load_policy_num = 29958658
                path = save_path+'/policy_'+str(load_policy_num)
                print("loaded_policy_path: ",path)
                model = PPO2.load(path, env=env, **model_dict)

            print("\033[91mTraining Starts, action: {0}\033[0m".format(action))
            if scratch:
                info = {'trial': trial, 'action': action, 'layers': net_arch, 'tolerance': tol, 'total time steps': total_time_step,\
                    'n_robots': n_robots, 'n_targets': n_target, 'episode_length': episode_length, 'reward_method': reward_method, 'observation_method': observation_method,\
                    'Additional Info': \
                        'Reset from random initial pos\n\
                        \t\tAgent rotates a bit less\n\
                        \t\tTarget does not stay in position\n\
                        \t\tshallower network\n\
                        \t\tPositive reward\n\
                        \t\tInitial pose a bit inward\n\
                        \t\tPPO MPI\n\
                        \t\tusing tanh to squash action\n\
                        \t\tlearning_rate: 5e-5, gamma:0.99, nminibatches: 256, cliprange: 0.02\n\
                        \t\tBeta policy test'}
                model_log = open(save_path+"/model_log.txt", 'w')
                _write_log(model_log, info)
                model_log.close()
                model.learn(total_time_step, save_interval=int(total_time_step*0.05), save_path=save_path)
            else:
                model.learn(total_time_step, loaded_step_num=load_policy_num, save_interval=int(total_time_step*0.05), save_path=save_path)
            print("\033[91mTraining finished\033[0m")

        elif auxilary:
            trial = 21
            action_option = ['linear', 'angular', 'fused', 'pickAndplace']
            observation_option = ['absolute', 'relative']
            reward_option = ['target', 'time', None]
            action = action_option[2]
            prefix2 = action+"_auxilary_trial"+str(trial)
            save_path = model_dir+prefix2
            os.makedirs(save_path, exist_ok=True)

            tol = 0.1
            n_robots = 1
            n_target = 1
            episode_length = 2000
            reward_method = reward_option[0]
            observation_method = observation_option[0]
            env = Manipulator2D(action=action, n_robots=n_robots, n_target=n_target, tol=tol, 
                            episode_length=episode_length, reward_method=reward_method, observation_method=observation_method, policy_name=prefix2)
            composite_primitive_name='PoseControl'
            model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

            if observation_method == 'absolute':
                aux1_obs_range = {'min': [-float('inf'), -float('inf'), -np.pi, -float('inf'), -float('inf'), -np.pi],
                                'max': [float('inf'), float('inf'), np.pi, float('inf'), float('inf'), np.pi]}
                aux1_obs_index = list(range(6))
                aux1_act_range = {'min': [-1, -np.pi],
                                'max': [1, np.pi]}
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
            prim_trial = 28
            policy_num = 1025
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=prim_obs_index,
                                                act_range=None, act_index=[0],
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
                                                load_value=False)
            
            prim_name = 'angular'
            prim_trial = 8
            policy_num = 5000000
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=prim_obs_index,
                                                act_range=None, act_index=[1],
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                                load_value=True)

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
                        \t\tPrimitives unbounded\n\
                        \t\tValue from pre-trained primitive\n\
                        \t\tWeight activation sigmoid\n\
                        \t\tPPO+SAC primitive test'}
            _write_log(model_log, info)
            model_log.close()
            model.learn(total_time_step, save_interval=int(total_time_step*0.01), save_path=save_path)
            print("\033[91mTrain Finished\033[0m")

    if load:
        if separate:
            action_list = ['linear', 'angular', 'fused', 'pickAndplace']
            observation_option = ['absolute', 'relative']
            reward_option = ['target', 'time', None]
            action = action_list[0]
            trial = 26

            tol = 0.1
            n_robots = 1
            n_target = 1
            episode_length = 1000
            reward_method = reward_option[0]
            observation_method = observation_option[0]
            info_dict = {'action': action, 'n_robots': n_robots, 'n_target':n_target, 'tol':tol, 
                        'episode_length':episode_length, 'reward_method':reward_method, 'observation_method':observation_method}
            env = Manipulator2D(visualize=False, **info_dict)

            policy_num = 2496513
            layer_structure_list = [[256, 256, 128, 128, 64], [128, 128, 64, 64, 32], [64, 128, 256, 128, 64], \
                                    [256, 256, 128, 128], [128, 64, 64, 32], [64, 64, 32, 32], \
                                    [512, 256, 256], [256, 256, 128], [128, 128, 128], \
                                    [256, 256], [128, 128]]
            layer_structure = layer_structure_list[6]
            net_arch = {"pi": layer_structure, "vf": layer_structure}
            policy_kwargs={'net_arch': [net_arch], 'act_fun': tf.nn.relu, 'squash':True}

            prefix2 = action+"_separate_trial"+str(trial)
            model = PPO2.load(model_path+prefix+prefix2+"/policy_"+str(policy_num), policy_kwargs=policy_kwargs)

            print("\033[91mTest Starts\033[0m")
            for i in range(10):
                print("  \033[91mTest iter: {0}\033[0m".format(i))
                obs = env.reset()
                n_iter = 0
                while True:
                    n_iter += 1
                    action, state = model.predict(obs, deterministic=False)
                    print(action)
                    logstd = model.logstd(obs)
                    if n_iter % 20:
                        print('logstd: ',logstd)
                        if action == 'fused':
                            print("  dist:\t{0:2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\ta_diff:\t{0: 2.3f}".format(obs[2]),"\taction:\t[{0: 2.3f} {1: 2.3f}]".format(action[0],action[1]))
                        elif action in ['linear', 'angular']:
                            print("  dist:\t{0:2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\taction:\t[{0:2.3f}]".format(action[0]))
                        elif action == 'pickAndplace':
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
            observation_option = ['absolute', 'relative']
            reward_option = ['target', 'time', None]
            action_type = action_list[2]
            trial = 23
            prefix2 = action_type+"_auxilary_trial"+str(trial)

            tol = 0.1
            n_robots = 1
            n_target = 1
            episode_length = 1000
            reward_method = reward_option[0]
            observation_method = observation_option[0]
            env = Manipulator2D(action=action_type, n_robots=n_robots, n_target=n_target, tol=tol, 
                            episode_length=episode_length, reward_method=reward_method, observation_method=observation_method)
            composite_primitive_name='PoseControl'
            model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

            if observation_method == 'absolute':
                observation_index = list(range(6))
            elif observation_method == 'relative':
                observation_index = list(range(3))
            policy_num = 250000
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
                    real_weight_lin = [coef00/(coef00+coef1), coef1/(coef00+coef1)]
                    real_weight_ang = [coef01/(coef01+coef2), coef2/(coef01+coef2)]
                    if n_iter % 1 == 0:
                        print()
                        print("Mu:",end='\t')
                        for name, item in prim_act.items():
                            print(name, item, end='\t')
                        print()
                        print("Log_std",end='\t')
                        for name, item in prim_log_std.items():
                            print(name, item, end='\t')
                        print()
                        print("action:\t[{0: 2.3f} {1: 2.3f}]".format(action[0],action[1]),"\tweight:\t",weight[0])
                        print("Real weight: ",real_weight_lin, real_weight_ang)
                        
                    obs, reward, done, _ = env.step(action, weight[0], test=True)
                    if done:
                        break
                env.render()
            print("\033[91mTest Finished\033[0m")


