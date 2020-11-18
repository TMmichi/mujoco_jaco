import path_config
from configuration import env_configuration, model_configuration, pretrain_configuration, info, total_time_step

import numpy as np
import gym

from stable_baselines.sac import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.sac_multi import SAC_MULTI
from stable_baselines.sac_multi.policies import MlpPolicy as MlpPolicy_sac


default_lr = model_configuration['learning_rate']
def _lr_scheduler(frac):
    return default_lr * frac

if __name__ == '__main__':

    env = gym.make("Pendulum-v0")

    print("\033[91mTraining Starts, action: {0}\033[0m".format(env_configuration['action']))
    model_configuration['learning_rate'] = _lr_scheduler
    model = SAC_MULTI(MlpPolicy_sac, env, **model_configuration)
    # model = SAC(MlpPolicy, env, verbose=1)
    model.learn(total_time_step, save_interval=int(total_time_step*0.05))
    # model.learn(total_time_step)
    print("\033[91mTraining finished\033[0m")