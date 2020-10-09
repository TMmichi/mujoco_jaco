import os, sys
import time
import path_config
from configuration import env_configuration, model_configuration, info, total_time_step
from pathlib import Path
from collections import OrderedDict
#os.environ['CUDA_VISIBLE_DEVICES']='3'

try:
    import spacenav, atexit
except Exception:
    pass

import numpy as np
import tensorflow as tf

from env_script.test_env.manipulator_2d import Manipulator2D
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_sac
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds


def _write_log(save_path, info, trial):
    model_log = open(save_path+"/model_log.txt", 'w')
    model_log.writelines("Trial: "+str(trial))
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


def _expert_3d(_obs):
    if sys.platform in ['linux', 'linux2']:
        event = spacenav.poll()
        if type(event) is spacenav.MotionEvent:
            action = np.array([event.x/350*0.99, -event.ry/350*3.14])
        else:
            action = [0,0,0,0,0,0,0,0]
        spacenav.remove_events(1)
        return action
    else:
        action = [0,0,0,0,0,0,0,0]
        return action


if __name__ == '__main__':
    package_path = str(Path(__file__).resolve().parent.parent)
    model_path = package_path+"/models_baseline/"
    prefix = "twowheel/"
    model_dir = model_path + prefix
    os.makedirs(model_dir, exist_ok=True)

    train = True
    separate = True
    train_mode_list = ['expert', 'scratch', 'load']
    train_mode = train_mode_list[0]
    test = False and not separate
    auxilary = True and not separate

    if train:
        if separate:
            trial = 0

            prefix2 = env_configuration['action']+"_separate_trial"+str(trial)
            save_path = model_dir+prefix2
            os.makedirs(save_path, exist_ok=True)
            env_configuration['policy_name'] = prefix2
            env = Manipulator2D(**env_configuration)

            model_configuration['tensorboard_log'] = save_path

            print("\033[91mTraining Starts, action: {0}\033[0m".format(env_configuration['action']))
            if train_mode == 'expert':
                n_episodes = 10000
                #traj_dict = generate_expert_traj(env.calculate_desired_action, model_dir+"/trajectory_10000", env, n_episodes=n_episodes)
                traj_dict = generate_expert_traj(_expert_3d, model_dir+"/trajectory_test", env, n_episodes=n_episodes)
                quit()
                traj_dict = np.load(model_dir+"/trajectory_10000.npz", allow_pickle=True)
                for name, elem in traj_dict.items():
                    print('nan in ',name, np.any(np.isnan(elem)))
                dataset = ExpertDataset(traj_data=traj_dict, batch_size=1024)
                model = SAC_MULTI(MlpPolicy_sac, env, **model_configuration)
                model.pretrain(dataset, n_epochs=1000, learning_rate=1e-6, val_interval=10)
                del dataset
                model.learn(total_time_step, save_interval=model_configuration['learning_starts'], save_path=save_path)
            elif train_mode == 'scratch':
                _write_log(save_path, info, trial)
                model = SAC_MULTI(MlpPolicy_sac, env, **model_configuration)
                model.learn(total_time_step, save_interval=model_configuration['learning_starts'], save_path=save_path)
            elif train_mode == 'load':
                load_policy_num = 6000000
                path = save_path+'/policy_'+str(load_policy_num)
                model = SAC_MULTI.load(path, env=env, **model_configuration)
                model.learn(total_time_step, loaded_step_num=load_policy_num, save_interval=model_configuration['learning_starts'], save_path=save_path)
            print("\033[91mTraining finished\033[0m")

        elif auxilary:
            trial = 21

            prefix2 = env_configuration['action']+"_auxilary_trial"+str(trial)
            save_path = model_dir+prefix2
            os.makedirs(save_path, exist_ok=True)
            env_configuration['policy_name'] = prefix2
            env = Manipulator2D(**env_configuration)

            model_configuration['tensorboard_log'] = save_path

            composite_primitive_name='PoseControl'
            model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

            if env_configuration['observation_method'] == 'absolute':
                aux1_obs_range = {'min': [-float('inf'), -float('inf'), -np.pi, -float('inf'), -float('inf'), -np.pi],
                                'max': [float('inf'), float('inf'), np.pi, float('inf'), float('inf'), np.pi]}
                aux1_obs_index = list(range(6))
                aux1_act_range = {'min': [-1, -np.pi],
                                'max': [1, np.pi]}
                aux1_act_index = list(range(2))
                total_obs_dim = 6
                prim_obs_index = list(range(6))
            elif env_configuration['observation_method'] == 'relative':
                aux1_obs_range = {'min': [-float('inf'), -np.pi, -np.pi],
                                'max': [float('inf'), np.pi, np.pi]}
                aux1_obs_index = list(range(3))
                aux1_act_range = {'min': [-1, -np.pi],
                                'max': [1, np.pi]}
                aux1_act_index = list(range(2))
                total_obs_dim = 3
                prim_obs_index = list(range(2))

            prim_name = 'aux1'
            model.construct_primitive_info(name=prim_name, freeze=False, level=1,
                                                obs_range=aux1_obs_range, obs_index=aux1_obs_index,
                                                act_range=aux1_act_range, act_index=aux1_act_index,
                                                layer_structure={'policy':[256, 256, 128, 128]})

            prim_name = 'linear'
            prim_trial = 19
            policy_num = 5000000
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=prim_obs_index,
                                                act_range=None, act_index=[0],
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
                                                load_value=True)
            
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

            model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env, **model_configuration)

            print("\033[91mTraining Starts\033[0m")
            info = {'trial': trial, 'action': action, 'layers': None, 'tolerance': tol, 'total time steps': total_time_step,\
                    'n_robots': n_robots, 'n_targets': n_target, 'episode_length': episode_length, 'reward_method': reward_method, 'observation_method': observation_method,\
                    'batch_size': batch_size,\
                    'Additional Info': \
                        '0.125* scaled rewards + linear prim trained with 0.125* scaled reward\n\
                        \t\tPrimitives unbounded\n\
                        \t\tValue from pre-trained primitive\n\
                        \t\tWeight activation sigmoid'}
            _write_log(save_path, info, trial)
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

    else:
        if separate:
            trial = 60

            prefix2 = env_configuration['action']+"_separate_trial"+str(trial)
            env_configuration['policy_name'] = prefix2
            env = Manipulator2D(visualize=False, **env_configuration)

            policy_num = 50000
            model_dict = {'layers': model_configuration['layers'], 'box_dist': model_configuration['box_dist'], 'policy_kwargs': model_configuration['policy_kwargs']}
            model = SAC_MULTI.load(model_path+prefix+prefix2+"/policy_"+str(policy_num), **model_dict)

            print("\033[91mTest Starts\033[0m")
            for i in range(10):
                print("  \033[91mTest iter: {0}\033[0m".format(i))
                obs = env.reset()
                while True:
                    actions, state = model.predict(obs, deterministic=True)                    
                    obs, reward, done, _ = env.step(actions, test=True)
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

