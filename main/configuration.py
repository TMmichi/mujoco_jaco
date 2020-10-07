import tensorflow as tf

env_configuration = {}
model_configuration = {}
info = {}

#####################################################################################################
action_option = ['linear', 'angular', 'fused', 'pickAndplace']
observation_option = ['absolute', 'relative']
reward_option = ['target', 'time', 'sparse', None]

env_configuration['action'] = action_option[2]
env_configuration['n_robots'] = 1
env_configuration['n_target'] = 1
env_configuration['arm1'] = 1
env_configuration['arm2'] = 1
env_configuration['dt'] = 0.01
env_configuration['tol'] = 0.1
env_configuration['episode_length'] = 2000
env_configuration['reward_method'] = reward_option[2]
env_configuration['observation_method'] = observation_option[0]
env_configuration['her'] = False
env_configuration['visualize'] = False
#####################################################################################################
#####################################################################################################
layer_structure_list = [[256, 256, 128, 128, 128, 64, 64], \
                        [256, 256, 128, 128, 64], [128, 128, 64, 64, 32], [64, 128, 256, 128, 64], \
                        [256, 256, 128, 128], [128, 64, 64, 32], [64, 64, 32, 32], \
                        [512, 256, 256], [256, 256, 128], [128, 128, 128], \
                        [256, 256], [128, 128]]
layer_structure = layer_structure_list[7]
layers = {"policy": layer_structure, "value": layer_structure}

total_time_step = 10000000
model_configuration['learning_starts'] = int(total_time_step*0.05)
model_configuration['layers'] = layers
model_configuration['batch_size'] = 2048
model_configuration['buffer_size'] = 1000000
model_configuration['gamma'] = 0.995
model_configuration['learning_rate'] = 1e-5
model_configuration['ent_coef'] = 'auto'
model_configuration['train_freq'] = 10
model_configuration['verbose'] = 1
model_configuration['box_dist'] = 'beta'
model_configuration['policy_kwargs'] = {'act_fun':tf.nn.swish}
#####################################################################################################
#####################################################################################################
info['action'] = action_option[2]
info['total_time_steps'] = 10000000
info['layers'] = layers
info['batch_size'] = 2048
info['buffer_size'] = 1000000
info['gamma'] = 0.995
info['learning_rate'] = 1e-5
info['ent_coef'] = 'auto'
info['train_freq'] = 10
info['verbose'] = 1
info['box_dist'] = 'beta'
info['non_lin'] = 'swish'

info['n_robots'] = 1
info['n_target'] = 1
info['tolerance'] = 0.1
info['episode_length'] = 2000
info['reward_method'] = reward_option[2]
info['observation_method'] = observation_option[0]
info['Additional Info']= \
            'Reset from random initial pos\n\
            \t\tAgent roates a bit less\n\
            \t\tTarget also moves (randomly)\n\
            \t\tInitial pose a bit inward\n\
            \t\tweights no biases'
#####################################################################################################