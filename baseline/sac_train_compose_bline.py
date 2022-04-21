import os, sys
import numpy as np
import tensorflow as tf
import itertools
from datetime import datetime

from environments.jaco_env.env_mujoco import JacoMujocoEnv
from rl_agents.policy_gradient.composenet.estimators import (Skill, 
															CompositionModule, 
															PolicyModule, 
															ValueModule)


tf.flags.DEFINE_string("model_dir", "experiment_logs/compose", "Directory to save checkpoints to.")
tf.flags.DEFINE_string("trained_primitives", "experiment_logs/skills/", "Directory to load saved skills from.")
tf.flags.DEFINE_string("transfer_dir", None, "Directory to load pre-trained compose layer from.")
tf.flags.DEFINE_string("env", "objects_env", "Name of environment.")
tf.flags.DEFINE_string("task", "collect_0_evade_1", "Name of environment.")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update.")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 2, "Evaluate the policy every N seconds.")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", 5, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS


# NOTE:
'''
How to retrieve skills?
How to compose skills?
Skill composing by the namescope?
'''
def create_compositions(n, skills, env):
	skill_embedders = []
	for sk in skills:
		name_scope = 'global_{}'.format(sk)
		# global_reaching
		# global_grasping
		skill_embedders.append(Skill(
			name_scope=name_scope,
			state_dims=env.get_state_shape()),
			state_idx=env.obsidx_dict[sk])
	# now all the compositions from left to right
	policy_compositions = [skill_embedders[0]]
	value_compositions = [skill_embedders[0]]
	# starting from the first skill
	for i in range(n-1):
		next_embedding = skill_embedders[i+1]
		name_scope='compose_policy_{}'.format(i)
		policy_compositions.append(CompositionModule(
			name_scope=name_scope,
			embedder_1=policy_compositions[-1], # skill i-1
			embedder_2=next_embedding))			# skill i

		name_scope='compose_value_{}'.format(i)
		value_compositions.append(CompositionModule(
			name_scope=name_scope,
			embedder_1=value_compositions[-1],
			embedder_2=next_embedding))
	
	# final layers sits on top of the last embedding
	# Train only the compose policy
	policy_trainable_scopes = \
		['compose_policy_{}/'.format(i) for i in range(n-1)]
	policy_net = PolicyModule(
		name_scope=name_scope,
		trainable_scopes=policy_trainable_scopes,
		num_outputs=len(VALID_ACTIONS),
		embedder=policy_compositions[-1],
		global_final_layer=global_final_layer)

	value_trainable_scopes = \
		['compose_value_{}/'.format(i) for i in range(n-1)]
	value_net = ValueModule(
		name_scope=name_scope,
		trainable_scopes=value_trainable_scopes,
		embedder=value_compositions[-1],
		global_final_layer=global_final_layer)
	return policy_net, value_net


env = JacoMujocoEnv(task='picking', robot_file='jaco2_curtain_torque', n_robot=1, visualize=False)
VALID_ACTIONS = list(range(7))

# Set the number of workers
NUM_WORKERS = FLAGS.parallelism

# directories for logs and model files
MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints", FLAGS.task)
LOG_DIR = os.path.join(MODEL_DIR, "logs", FLAGS.task)
PRIMITIVES_DIR = os.path.join(FLAGS.trained_primitives, "checkpoints", FLAGS.env)
TRANSFER_DIR = FLAGS.transfer_dir
SKILLS = ['reaching1', 'grasping', 'reaching2', 'releasing']
NUM_SKILLS = len(SKILLS)

if not os.path.exists(CHECKPOINT_DIR):
	os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)


with tf.device("/cpu:0"):
	# Keeps track of the number of updates we've performed
	global_step = tf.Variable(0, name="global_step", trainable=False)
	policy_net, value_net = create_compositions(NUM_SKILLS, SKILLS, env)

	# Global step iterator
	global_counter = itertools.count()

	saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=2)

	logfile = os.path.join(
		LOG_DIR, '{:%Y-%m-%d_%H:%M:%S}.log'.format(datetime.now()))
	

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()

	# Load the primitives
	latest_checkpoint = tf.train.latest_checkpoint(PRIMITIVES_DIR)
	if latest_checkpoint:
		# only load these variables
		to_load = []
		for sk in SKILLS:
			to_load += tf.contrib.slim.get_variables(
				scope='global_{}/'.format(sk),
				collection=tf.GraphKeys.TRAINABLE_VARIABLES)
		to_load += tf.contrib.slim.get_variables(
			scope='policy_net/',
			collection=tf.GraphKeys.TRAINABLE_VARIABLES)
		to_load += tf.contrib.slim.get_variables(
			scope='value_net/',
			collection=tf.GraphKeys.TRAINABLE_VARIABLES)

		loader = tf.train.Saver(to_load)
		sys.stderr.write("\nLoading Primitives from: {}\n".format(latest_checkpoint))
		loader.restore(sess, latest_checkpoint)

	# Load the transfer compose layer if any
	if TRANSFER_DIR:
		latest_checkpoint = tf.train.latest_checkpoint(TRANSFER_DIR)
		if latest_checkpoint:
			# only load these variables
			to_load = []
			for p in range(NUM_SKILLS-1):
				to_load += tf.contrib.slim.get_variables(
					scope='compose_policy_{}/'.format(p),
					collection=tf.GraphKeys.TRAINABLE_VARIABLES)
				to_load += tf.contrib.slim.get_variables(
					scope='compose_value_{}/'.format(p),
					collection=tf.GraphKeys.TRAINABLE_VARIABLES)
			loader = tf.train.Saver(to_load)
			sys.stderr.write("\nLoading composition layers from: {}\n".format(latest_checkpoint))
			loader.restore(sess, latest_checkpoint)
	