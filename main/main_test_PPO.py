import os
import time
import path_config
from configuration import pretrain_configuration
from pathlib import Path
from configuration import env_configuration, model_configuration, pretrain_configuration, info, total_time_step

import numpy as np
import tensorflow as tf

from env_script.test_env.manipulator_2d import Manipulator2D
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
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

def make_env(rank, seed=0, **kwargs):
    def _init():
        env = Manipulator2D(**kwargs)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def _lr_scheduler(frac):
    return 1e-6 * frac


if __name__ == '__main__':
    package_path = str(Path(__file__).resolve().parent.parent)
    model_path = package_path+"/models_baseline/"
    prefix = "twowheel/"
    model_dir = model_path + prefix
    os.makedirs(model_dir, exist_ok=True)

    train = True
    scratch = False

    if train:
        trial = 89

        prefix2 = env_configuration['action']+"_separate_trial"+str(trial)
        save_path = model_dir+prefix2
        os.makedirs(save_path, exist_ok=True)
        env_configuration['policy_name'] = prefix2

        num_cpu = 8
        # env = SubprocVecEnv([make_env(i, **env_configuration) for i in range(num_cpu)])
        env = Manipulator2D(**env_configuration)
        model_configuration['tensorboard_log'] = save_path

        net_arch = {'pi': model_configuration['layers']['policy'], 'vf': model_configuration['layers']['value']}
        obs_relativity = {'subtract':{'ref':[3,4],'tar':[0,1]}}
        # obs_relativity = {'subtract':{'ref':[1],'tar':[0]}}
        obs_index = [0,1,2,3,4]
        policy_kwargs = {'net_arch': [net_arch], 'obs_relativity':obs_relativity, 'obs_index':obs_index}
        policy_kwargs.update(model_configuration['policy_kwargs'])
        model_dict = {'gamma': 0.99, 'tensorboard_log': save_path, 'policy_kwargs': policy_kwargs, 'verbose':1}

        if scratch:
            model = PPO2(MlpPolicy, env, **model_dict)
            print("model created")
            _write_log(save_path, info, trial)
            model.learn(total_timesteps=total_time_step, save_interval=50, save_path=save_path)
        else:
            traj_dict = np.load(model_dir+env_configuration['action']+"_10000.npz", allow_pickle=True)
            dataset = ExpertDataset(traj_data=traj_dict, batch_size=32764)
            model = PPO2(MlpPolicy, env, **model_dict)
            model.pretrain(dataset, **pretrain_configuration)
            model.save(save_path+"/policy_0")
            del dataset
            _write_log(save_path, info, trial)
            model.learn(total_timesteps=total_time_step, save_interval=50, save_path=save_path)

        # print("\033[91mTraining Starts, action: {0}\033[0m".format(action))
        # if scratch:
        #     model.learn(total_time_step, save_interval=int(total_time_step*0.05), save_path=save_path)
        # else:
        #     model.learn(total_time_step, loaded_step_num=load_policy_num, save_interval=int(total_time_step*0.05), save_path=save_path)
        # print("\033[91mTraining finished\033[0m")

    else:
        action_list = ['linear', 'angular', 'fused', 'pickAndplace']
        observation_option = ['absolute', 'relative']
        reward_option = ['target', 'time', 'sparse', None]
        action = action_list[0]
        trial = 53

        tol = 0.1
        n_robots = 1
        n_target = 1
        episode_length = 1000
        reward_method = reward_option[0]
        observation_method = observation_option[0]
        info_dict = {'action': action, 'n_robots': n_robots, 'n_target':n_target, 'tol':tol, 
                    'episode_length':episode_length, 'reward_method':reward_method, 'observation_method':observation_method,
                    'visualize': True}
        env = SubprocVecEnv([make_env(i, **info_dict) for i in range(1)])

        policy_num = 49741825
        layer_structure_list = [[256, 256, 128, 128, 64], [128, 128, 64, 64, 32], [64, 128, 256, 128, 64], \
                                [256, 256, 128, 128], [128, 64, 64, 32], [64, 64, 32, 32], \
                                [512, 256, 256], [256, 256, 128], [128, 128, 128], \
                                [256, 256], [128, 128]]
        layer_structure = layer_structure_list[6]
        net_arch = {"pi": layer_structure, "vf": layer_structure}
        policy_kwargs={'net_arch': [net_arch], 'act_fun': tf.nn.swish, 'squash':False, 'box_dist':'beta'}

        prefix2 = action+"_separate_trial"+str(trial)
        model = PPO2.load(model_path+prefix+prefix2+"/policy_"+str(policy_num), env=env, policy_kwargs=policy_kwargs)

        print("\033[91mTest Starts\033[0m")
        for i in range(10):
            print("  \033[91mTest iter: {0}\033[0m".format(i))
            obs = env.reset()
            n_iter = 0
            while True:
                n_iter += 1
                action, state = model.predict(obs, deterministic=False)
                print(action)
                #logstd = model.logstd(obs)
                if n_iter % 20:
                    #print('logstd: ',logstd)
                    if action == 'fused':
                        print("  dist:\t{0:2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\ta_diff:\t{0: 2.3f}".format(obs[2]),"\taction:\t[{0: 2.3f} {1: 2.3f}]".format(action[0],action[1]))
                    elif action in ['linear', 'angular']:
                        print("  dist:\t{0:2.3f}".format(obs[0]),"\tang:\t{0: 2.3f}".format(obs[1]),"\taction:\t[{0:2.3f}]".format(action[0]))
                    elif action == 'pickAndplace':
                        #print("  x, y, w: {0: 2.3f}, {1: 2.3f}, {2: 2.3f}".format(obs[0], obs[1], obs[2])," action: [{0: 2.3f}, {1: 2.3f}, {2: 2.3f}, {3: 2.3f}]".format(action[0],action[1],action[2],action[3]))
                        pass
                obs, reward, done, _ = env.step(action)
                if done:
                    print(reward)
                    break
        print("\033[91mTest Finished\033[0m")
