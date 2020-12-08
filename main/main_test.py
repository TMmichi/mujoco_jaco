import os, sys
import time
import path_config
from pathlib import Path
from configuration import env_configuration, model_configuration, pretrain_configuration, info, total_time_step
#os.environ['CUDA_VISIBLE_DEVICES']='3'

try:
    import spacenav, atexit
except Exception:
    pass

import numpy as np

from env_script.test_env.manipulator_2d import Manipulator2D
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines.ppo2 import PPO2
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

def _open_connection():
    try:            
        print("Opening connection to SpaceNav driver ...")
        spacenav.open()
        print("... connection established.")
    except spacenav.ConnectionError:
        print("No connection to the SpaceNav driver. Is spacenavd running?")

def _close_connection():
    atexit.register(spacenav.close)

def _expert_3d(_obs):
    if sys.platform in ['linux', 'linux2']:
        event = spacenav.poll()
        if type(event) is spacenav.MotionEvent:
            action = np.array([event.z/350*0.99, event.ry/350*3.14])
        else:
            action = [0,0]
        spacenav.remove_events(1)
        return action
    else:
        action = [0,0,0,0,0,0,0,0]
        return action

default_lr = model_configuration['learning_rate']
def _lr_scheduler(frac):
    return default_lr * frac

if __name__ == '__main__':
    package_path = str(Path(__file__).resolve().parent.parent)
    model_path = package_path+"/models_baseline/"
    prefix = "twowheel/"
    model_dir = model_path + prefix
    os.makedirs(model_dir, exist_ok=True)

    train = False
    separate = False
    train_mode_list = ['human', 'auto', 'scratch', 'load']
    train_mode = train_mode_list[2]
    auxilary = True and not separate
    test = False and not separate

    if train:
        if separate:
            trial = 89

            prefix2 = env_configuration['action']+"_separate_trial"+str(trial)
            save_path = model_dir+prefix2
            os.makedirs(save_path, exist_ok=True)
            env_configuration['policy_name'] = prefix2
            env = Manipulator2D(**env_configuration)

            model_configuration['tensorboard_log'] = save_path

            print("\033[91mTraining Starts, action: {0}\033[0m".format(env_configuration['action']))
            if train_mode == 'human':
                n_episodes = 100
                _open_connection()
                traj_dict = generate_expert_traj(_expert_3d, model_dir+"/trajectory_expert", env, n_episodes=n_episodes)
                _close_connection()
                traj_dict = np.load(model_dir+"/trajectory_expert.npz", allow_pickle=True)
                for name, elem in traj_dict.items():
                    if np.any(np.isnan(elem)):
                        print('NAN in ',name)
                        quit()
                dataset = ExpertDataset(traj_data=traj_dict, batch_size=1024)
                model = SAC_MULTI(MlpPolicy_sac, env, **model_configuration)
                model.pretrain(dataset, **pretrain_configuration)
                del dataset
                _write_log(save_path, info, trial)
                model.learn(total_time_step, save_interval=int(total_time_step*0.05), save_path=save_path)
            if train_mode == 'auto':
                n_episodes = 1000
                traj_dict = generate_expert_traj(env.calculate_desired_action, model_dir+env_configuration['action']+"_10000", env, n_episodes=n_episodes)
                quit()
                traj_dict = np.load(model_dir+env_configuration['action']+"_10000.npz", allow_pickle=True)
                for name, elem in traj_dict.items():
                    if np.any(np.isnan(elem)):
                        print('NAN in ',name)
                        quit()
                dataset = ExpertDataset(traj_data=traj_dict, batch_size=1024)
                model_configuration['learning_rate'] = _lr_scheduler
                model = SAC_MULTI(MlpPolicy_sac, env, **model_configuration)
                model.pretrain(dataset, **pretrain_configuration)
                model.save(save_path+"/policy_0")
                del dataset
                _write_log(save_path, info, trial)
                model.learn(total_time_step, save_interval=int(total_time_step*0.05), save_path=save_path)
            elif train_mode == 'scratch':
                _write_log(save_path, info, trial)
                model_configuration['learning_rate'] = _lr_scheduler
                model = SAC_MULTI(MlpPolicy_sac, env, **model_configuration)
                model.learn(total_time_step, save_interval=int(total_time_step*0.05), save_path=save_path)
            elif train_mode == 'load':
                load_policy_num = 1000000
                path = save_path+'/policy_'+str(load_policy_num)
                model_configuration['learning_rate'] = _lr_scheduler
                model = SAC_MULTI.load(path, env=env, **model_configuration)
                model.learn(total_time_step, loaded_step_num=load_policy_num, save_interval=int(total_time_step*0.05), save_path=save_path)
            print("\033[91mTraining finished\033[0m")

        elif auxilary:
            trial = 91

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
                                                act_range=aux1_act_range, act_index=aux1_act_index, act_scale=0.1,
                                                obs_relativity={},
                                                layer_structure={'policy':[64, 64]})

            prim_name = 'linear'
            prim_trial = 89
            policy_num = 2265473 #1785473 #1375873
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=[0,1,2,3,4],
                                                act_range=None, act_index=[0], act_scale=1,
                                                obs_relativity={'subtract':{'ref':[3,4], 'tar':[0,1]}},
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
                                                load_value=False)

            prim_name = 'angular'
            prim_trial = 89
            policy_num = 1330177
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=[0,1,2,3,4],
                                                act_range=None, act_index=[1], act_scale=1,
                                                layer_structure=None,
                                                obs_relativity={'subtract':{'ref':[3,4], 'tar':[0,1]}},
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                                load_value=False)
            
            prim_name = 'angular_adj'
            prim_trial = 89
            policy_num = 152577
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=[2,5],
                                                act_range=None, act_index=[1], act_scale=1,
                                                layer_structure=None,
                                                obs_relativity={'subtract':{'ref':[5], 'tar':[2]}},
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                                load_value=False)

            total_obs_dim = 6
            number_of_primitives = 4
            model.construct_primitive_info(name='weight', freeze=False, level=1,
                                                obs_range=0, obs_index=list(range(total_obs_dim)),
                                                act_range=0, act_index=list(range(number_of_primitives)), act_scale=None,
                                                obs_relativity={},
                                                layer_structure={'policy':[64, 64],'value':[64, 64]})
            
            model_configuration['learning_rate'] = _lr_scheduler
            model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env, **model_configuration)

            print("\033[91mTraining Starts\033[0m")
            if train_mode == 'human':
                # n_episodes = 100
                # _open_connection()
                # traj_dict = generate_expert_traj(env.calculate_desired_action, model_dir+env_configuration['action']+"_100", env, n_episodes=n_episodes)
                # _close_connection()
                # quit()
                traj_dict = np.load(model_dir+env_configuration['action']+"_100.npz", allow_pickle=True)
                for name, elem in traj_dict.items():
                    if np.any(np.isnan(elem)):
                        print('NAN in ',name)
                        quit()
                dataset = ExpertDataset(traj_data=traj_dict, batch_size=1024)
                model_configuration['learning_rate'] = _lr_scheduler
                model.pretrain(dataset, **pretrain_configuration)
                model.save(save_path+"/policy_0")
                del dataset
                _write_log(save_path, info, trial)
                model.learn(total_time_step, save_interval=int(total_time_step*0.01), save_path=save_path)
            elif train_mode == 'scratch':
                _write_log(save_path, info, trial)
                model.learn(total_time_step, save_interval=int(total_time_step*0.01), save_path=save_path)
            elif train_mode == 'load':
                load_policy_num = 0
                path = save_path+'/policy_'+str(load_policy_num)
                model_configuration['learning_rate'] = _lr_scheduler
                policy_zip_path = model_path+prefix+prefix2+"/policy_"+str(load_policy_num)+".zip"
                model.construct_primitive_info(name=None, freeze=True, level=1,
                                                    obs_range=None, obs_index=list(range(total_obs_dim)),
                                                    act_range=None, act_index=[0, 1],
                                                    layer_structure=None,
                                                    loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                                    load_value=True)
                model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env, **model_configuration)
                model.learn(total_time_step, loaded_step_num=load_policy_num, save_interval=int(total_time_step*0.05), save_path=save_path)
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
            trial = 32

            prefix2 = env_configuration['action']+"_auxilary_trial"+str(trial)
            save_path = model_dir+prefix2
            os.makedirs(save_path, exist_ok=True)
            env_configuration['policy_name'] = prefix2
            env = Manipulator2D(**env_configuration)

            model_configuration['tensorboard_log'] = save_path

            composite_primitive_name='PoseControl'
            model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

            if env_configuration['observation_method'] == 'absolute':
                total_obs_dim = 6
                prim_obs_index = list(range(6))
            elif env_configuration['observation_method'] == 'relative':
                total_obs_dim = 3
                prim_obs_index = list(range(2))

            prim_name = 'linear'
            prim_trial = 75
            policy_num = 100000
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=prim_obs_index,
                                                act_range=None, act_index=[0],
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path),
                                                load_value=True)

            prim_name = 'angular'
            prim_trial = 15
            policy_num = 100000
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=prim_obs_index,
                                                act_range=None, act_index=[1],
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                                load_value=True)
            
            prim_name = 'angular_adj'
            prim_trial = 2
            policy_num = 0
            policy_zip_path = model_path+prefix+prim_name+"_separate_trial"+str(prim_trial)+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=prim_name, freeze=True, level=1,
                                                obs_range=None, obs_index=prim_obs_index,
                                                act_range=None, act_index=[1],
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                                load_value=True)

            total_obs_dim = 6
            number_of_primitives = 2
            model.construct_primitive_info(name='weight', freeze=False, level=1,
                                                obs_range=0, obs_index=list(range(total_obs_dim)),
                                                act_range=0, act_index=list(range(number_of_primitives)),
                                                layer_structure={'policy':[512, 256, 256],'value':[512, 256, 256]})
            
            model_configuration['learning_rate'] = _lr_scheduler
            model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env, **model_configuration)

            print("\033[91mTraining Starts\033[0m")
            if train_mode == 'human':
                # n_episodes = 100
                # _open_connection()
                # traj_dict = generate_expert_traj(_expert_3d, model_dir+"/fused_100", env, n_episodes=n_episodes)
                # _close_connection()
                # quit()
                traj_dict = np.load(model_dir+env_configuration['action']+"_100.npz", allow_pickle=True)
                for name, elem in traj_dict.items():
                    if np.any(np.isnan(elem)):
                        print('NAN in ',name)
                        quit()
                dataset = ExpertDataset(traj_data=traj_dict, batch_size=1024)
                model_configuration['learning_rate'] = _lr_scheduler
                model.pretrain(dataset, **pretrain_configuration)
                model.save(save_path+"/policy_0")
                del dataset
                _write_log(save_path, info, trial)
                model.learn(total_time_step, save_interval=int(total_time_step*0.01), save_path=save_path)
            else:
                _write_log(save_path, info, trial)
                model.learn(total_time_step, save_interval=int(total_time_step*0.01), save_path=save_path)
            print("\033[91mTrain Finished\033[0m")

    else:
        if separate:
            trial = 89

            prefix2 = env_configuration['action']+"_separate_trial"+str(trial)
            env_configuration['policy_name'] = prefix2
            env = Manipulator2D(**env_configuration)

            policy_num = 2265473
            # policy_num = 1330177
            # policy_num = 152577
            net_arch = {'pi': model_configuration['layers']['policy'], 'vf': model_configuration['layers']['value']}
            obs_relativity = {'subtract':{'ref':[3,4],'tar':[0,1]}}
            obs_index = [0,1,2,3,4]
            # obs_relativity = {'subtract':{'ref':[1],'tar':[0]}}
            # obs_index = [0,1]
            policy_kwargs = {'net_arch': [net_arch], 'obs_relativity':obs_relativity, 'obs_index':obs_index}
            policy_kwargs.update(model_configuration['policy_kwargs'])
            model_dict = {'layers': model_configuration['layers'], 'policy_kwargs': policy_kwargs}
            model = PPO2.load(model_path+prefix+prefix2+"/policy_"+str(policy_num), **model_dict)

            print("\033[91mTest Starts\033[0m")
            for i in range(10):
                print("  \033[91mTest iter: {0}\033[0m".format(i))
                obs = env.reset()
                while True:
                    actions, state = model.predict(obs, deterministic=False)                    
                    obs, reward, done, _ = env.step(actions, test=True)
                    if done:
                        print(reward)
                        break
            print("\033[91mTest Finished\033[0m")

        else:
            trial = 89
            prefix2 = env_configuration['action']+"_auxilary_trial"+str(trial)

            env = Manipulator2D(**env_configuration)
            composite_primitive_name='PoseControl'
            model = SAC_MULTI(policy=MlpPolicy_sac, env=None, _init_setup_model=False, composite_primitive_name=composite_primitive_name)

            if env_configuration['observation_method'] == 'absolute':
                observation_index = list(range(6))
            elif env_configuration['observation_method'] == 'relative':
                observation_index = list(range(3))
            policy_num = 1100000
            policy_zip_path = model_path+prefix+prefix2+"/policy_"+str(policy_num)+".zip"
            model.construct_primitive_info(name=None, freeze=True, level=1,
                                                obs_range=None, obs_index=observation_index,
                                                act_range=None, act_index=[0, 1], act_scale=1,
                                                obs_relativity={},
                                                layer_structure=None,
                                                loaded_policy=SAC_MULTI._load_from_file(policy_zip_path), 
                                                load_value=True)
            model = SAC_MULTI.pretrainer_load(model=model, policy=MlpPolicy_sac, env=env, **model_configuration)

            print("\033[91mTest Starts\033[0m")
            for i in range(10):
                print("\033[91mTest iter: {0}\033[0m".format(i))
                obs = env.reset()
                while True:
                    actions, state = model.predict(obs)
                    weight = model.get_weight(obs)['level1_PoseControl/weight']
                    obs, reward, done, _ = env.step(actions, weight[0], test=True)
                    if done:
                        break
            print("\033[91mTest Finished\033[0m")

