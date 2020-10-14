import tensorflow as tf

env_configuration = {}
model_configuration = {}
pretrain_configuration = {}
info = {}

#####################################################################################################
action_option = ['linear', 'angular', 'angular_adj', 'fused', 'pickAndplace']
observation_option = ['absolute', 'relative']
reward_option = ['target', 'time', 'sparse', None]

env_configuration['action'] = action_option[2]
env_configuration['n_robots'] = 1
env_configuration['n_target'] = 1
env_configuration['arm1'] = 1
env_configuration['arm2'] = 1
env_configuration['dt'] = 0.01
env_configuration['tol'] = 0.2
env_configuration['episode_length'] = 1500
env_configuration['observation_method'] = observation_option[0]
env_configuration['reward_method'] = reward_option[2]
env_configuration['her'] = False
env_configuration['visualize'] = True
#####################################################################################################
#####################################################################################################
total_time_step = 1000000
layer_structure_list = [[256, 256, 128, 128, 128, 64, 64], \
                        [256, 256, 128, 128, 64], [128, 128, 64, 64, 32], [64, 128, 256, 128, 64], \
                        [256, 256, 128, 128], [128, 64, 64, 32], [64, 64, 32, 32], \
                        [512, 256, 256], [256, 256, 128], [128, 128, 128], \
                        [256, 256], [128, 128]]
layer_structure = layer_structure_list[7]
layers = {"policy": layer_structure, "value": layer_structure}

model_configuration['learning_starts'] = 10
model_configuration['layers'] = layers
model_configuration['batch_size'] = 1024
model_configuration['buffer_size'] = 1000000
model_configuration['gamma'] = 0.995
model_configuration['learning_rate'] = 3e-6
model_configuration['ent_coef'] = 'auto'
model_configuration['train_freq'] = 5
model_configuration['verbose'] = 1
model_configuration['box_dist'] = 'beta'
model_configuration['policy_kwargs'] = {'act_fun':tf.nn.swish}
pretrain_configuration['n_epochs'] = 200
pretrain_configuration['learning_rate'] = 5e-7
pretrain_configuration['val_interval'] = 1
#####################################################################################################
info['non_lin'] = 'swish'
info['Additional Info']='Weight normalized\n\
    \t\tTarget does not move\n\
    \t\tSparse Positive reward\n\
    \t\tNo aux\n\
    \t\tBaseline alpha, beta = 0.5'
# info['Additional Info']= \
#             'Reset from random initial pos\n\
#             \t\tAgent roates a bit less\n\
#             \t\tTarget also moves (randomly)\n\
#             \t\tInitial pose a bit inward\n\
#             \t\tweights no biases\n\
#             \t\tBeta pre-training with all params\n\
#             \t\tPre-train with more expert data\n\
#             \t\tLearning rate with e decay'














#####################################################################################################
info['action'] = action_option[2]
info['total_time_steps'] = total_time_step
info['layers'] = layers
info['batch_size'] = model_configuration['batch_size']
info['buffer_size'] = model_configuration['buffer_size']
info['gamma'] = model_configuration['gamma']
info['learning_rate'] = model_configuration['learning_rate']
info['ent_coef'] = model_configuration['ent_coef']
info['train_freq'] = model_configuration['train_freq']
info['verbose'] = model_configuration['verbose']
info['box_dist'] = model_configuration['box_dist']

info['n_robots'] = 1
info['n_target'] = 1
info['tolerance'] = 0.1
info['episode_length'] = 2000
info['reward_method'] = reward_option[2]
info['observation_method'] = observation_option[0]
#####################################################################################################