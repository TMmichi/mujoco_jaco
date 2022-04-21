"""
@author: himanshusahni
Modified from a3c code by dennybritz

Defines policy/value modules used by composenet agent
"""

import numpy as np
import tensorflow as tf
from rl_agents.utils.utils import *


EPS = 1e-6

class Skill(object):
	"""
	Base skill Embedding generartor. Given a state, will generate an embedding
	for a particular skill.
	The same network is used for generating policy and value embeddings.
	Args:
		name_scope: Prepended to scope of every variable
		reuse: If true, an existing shared network will be re-used.
		state_dims: Dimensions of the input state.
	"""

	def __init__(self, name_scope='', reuse=False, state_dims=None, state_idx=None):
		with tf.variable_scope(name_scope):
			# Placeholders for our input
			# self.states = [tf.placeholder(shape=[
			# 	None, state_dims[0], state_dims[1], 1], dtype=tf.uint8, name="X")]
			self.state_idx = state_idx
			self.states = [tf.placeholder(shape=[None, state_dims], dtype=tf.float32, name="X")]

			# Normalize
			# X = tf.to_float(self.states[0]) / 255.0
			self.batch_size = tf.shape(self.states[0])[0]

			# the graph structure
			with tf.variable_scope("shared", reuse=reuse):
				self.embedding = self.build_shared_network(self.states[0])

	def build_shared_network(self, X):
		"""
		Builds a 2-layer network fc -> fc.
		This network is shared by both the policy and value network.
		Args:
			X: Inputs
		Returns:
			Final layer activations.
		"""

		# Two convolutional layers.
		fc1 = tf.contrib.layers.fully_connected(
			inputs=X,
			num_outputs=256,
			scope="fc1")

		# Fully connected layer
		fc2 = tf.contrib.layers.fully_connected(
			inputs=fc1,
			num_outputs=256,
			scope="fc2")

		return fc2

class CompositionModule(object):
	"""
	Composed embedding generator. Creates a composed embedding out of two base
	embeddings
	Args:
		name_scope: Prepended to scope of every variable.
		embedder_1: first embedding object to be composed
		embedder_2: second embedding object to be composed
	"""
	def __init__(self, name_scope, embedder_1, embedder_2):

		# assuming batch size is same for everything
		self.batch_size = embedder_1.batch_size
		# collect all the state input placeholders
		self.states = embedder_1.states + embedder_2.states
		# concatenate the two trunks together
		self.concat_layer = tf.concat([embedder_1.embedding, embedder_2.embedding], 1)
		# fully connected layer for compose
		self.embedding= tf.contrib.layers.fully_connected(
			inputs=self.concat_layer,
			num_outputs=256,
			reuse=tf.AUTO_REUSE,
			scope=name_scope+'/fully_connected')

class PolicyModule(object):
	"""
	Converts an embedding into a policy, i.e. a distribution over actions
	Args:
		name_scope: Prepended to scope of loss related variables and policy layer
			for worker threads
		trainable_scopes: list of scopes to apply gradient to
		num_outputs: number of actions
		embedder: embedding object for task
		global_final_layer: whether the policy layer is shared
	"""

	def __init__(self, name_scope, trainable_scopes, num_outputs, embedder, global_final_layer=False):
		print(name_scope)
		print(trainable_scopes)

		# assuming batch size is same for everything
		self.batch_size = embedder.batch_size
		# collect all the state input placeholders
		self.states = embedder.states
		self.state_idx = embedder.state_idx
		self.summaries = []

		with tf.variable_scope(name_scope, auxiliary_name_scope=False) as wns:
			self.num_outputs = num_outputs
			# The TD target value
			self.advantages = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="advantage")
			# Previous actions
			self.act_prev  = tf.placeholder(shape=[None, num_outputs], dtype=tf.float32, name="act_prev")
			self.unsquashed_act_prev = tf.math.atanh(self.act_prev, name='unsquashed_act_prev')
			self.unsquashed_act_prev_clip = tf.clip_by_value(self.act_prev, -8, 8, name='unsquashed_act_prev_clip')

			self.unsquashed_mu_prev  = tf.placeholder(shape=[None, num_outputs], dtype=tf.float32, name="unsquashed_mu_prev")
			
			self.summaries.append(tf.summary.histogram('prev_actions', self.act_prev))

		# shared policy layer
		if global_final_layer:
			with tf.variable_scope("policy_net", auxiliary_name_scope=False, reuse=tf.AUTO_REUSE) as wpns:
				mu = tf.contrib.layers.fully_connected(
						embedder.embedding, num_outputs, activation_fn=None, scope='mu')
				self.unsquashed_mu = mu
				# logstd = tf.contrib.layers.fully_connected(
				# 		embedder.embedding, num_outputs, activation_fn=None, scope='logstd')
				# Reparameterization
				logstd = tf.constant([-4.605]*num_outputs)
				action = mu + tf.random_normal(tf.shape(mu)) * tf.exp(logstd)
		else:
			with tf.variable_scope(wns, auxiliary_name_scope=False):
				with tf.variable_scope("policy_net", auxiliary_name_scope=False) as wpns:
					mu = tf.contrib.layers.fully_connected(
							embedder.embedding, num_outputs, activation_fn=None, scope='mu')
					self.unsquashed_mu = mu
					# logstd = tf.contrib.layers.fully_connected(
					# 		embedder.embedding, num_outputs, activation_fn=None, scope='logstd')
					# Reparameterization
					logstd = tf.constant([-4.605]*num_outputs)
					action = mu + tf.random_normal(tf.shape(mu)) * tf.exp(logstd)

		# get loss/gradients
		with tf.variable_scope(wns, auxiliary_name_scope=False):
			with tf.variable_scope(wpns, auxiliary_name_scope=False):
				logp_prev_actions = -0.5  * (((self.unsquashed_act_prev - mu) / (tf.exp(logstd) + EPS)) ** 2\
										+ 2 * logstd + np.log(2 * np.pi))
				logp_actions = -0.5  * (((action - mu) / (tf.exp(logstd) + EPS)) ** 2\
										+ 2 * logstd + np.log(2 * np.pi))
				self.summaries.append(tf.summary.histogram('mu_diff', self.unsquashed_mu_prev - mu))
				self.summaries.append(tf.summary.histogram('act_diff', self.unsquashed_act_prev - mu))
				self.summaries.append(tf.summary.histogram('std', tf.exp(logstd)))
				sumlogp_prev_action = tf.reduce_sum(logp_prev_actions, axis=1)
				sumlogp_action = tf.reduce_sum(logp_actions, axis=1)
				self.summaries.append(tf.summary.histogram('sumlogp_prev_action bef squash', sumlogp_prev_action))
				self.summaries.append(tf.summary.histogram('sumlogp_action bef squash', sumlogp_action))

				self.mu = tf.tanh(mu)
				self.logstd = logstd
				self.action = tf.tanh(action)
				sumlogp_prev_action -= tf.reduce_sum(tf.math.log(1-tf.tanh(self.unsquashed_act_prev) ** 2 + EPS), axis=1)
				sumlogp_action -= tf.reduce_sum(tf.math.log(1-self.action ** 2 + EPS), axis=1)
				self.summaries.append(tf.summary.histogram('sumlogp_prev_action aft squash', sumlogp_prev_action))
				self.summaries.append(tf.summary.histogram('sumlogp_action aft squash', sumlogp_action))

				# We add entropy to the loss to encourage exploration
				self.entropy = tf.reduce_sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1, name='entropy')
				self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")
				
				# self.losses = - (sumlogp_prev_action * self.advantages + 4e-7 * self.entropy)
				self.losses = - (sumlogp_prev_action * self.advantages)
				self.loss = tf.reduce_sum(self.losses, name="policy_loss")

				self.summaries.append(tf.summary.scalar('policy loss', self.loss))
				self.summaries.append(tf.summary.scalar('policy entropy', self.entropy_mean))

				# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
				self.optimizer = tf.train.AdamOptimizer(learning_rate=7e-5, name='Adam_policy')
				self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
				# only take gradients for the trainable layers
				self.grads_and_vars = [[grad, var] \
					for grad, var in self.grads_and_vars \
						if grad is not None and \
						any([cs in var.name for cs in trainable_scopes])]
				if len(self.grads_and_vars) > 0:
					self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
						global_step=tf.contrib.framework.get_global_step())
				
				self.predictions = {
					"mu": self.mu,
					"logstd": self.logstd,
					"action": self.action,
					'unsquashed': self.unsquashed_mu
				}

class ValueModule(object):
	"""
	Converts an embedding into a value.
	Args:
		name_scope: Prepended to scope of every variable
		trainable_scopes: list of scopes to apply gradient to
		embedder: embedder for task
		global_final_layer: whether the policy layer is shared
	"""

	def __init__(self, name_scope, trainable_scopes, embedder, global_final_layer=False):

		# collect all the state input placeholders
		self.states = embedder.states
		self.state_idx = embedder.state_idx
		self.summaries = []


		with tf.variable_scope(name_scope, auxiliary_name_scope=False):
			# The TD target value
			self.targets = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="target_value")

		# now add the original value net on top
		if global_final_layer:
			with tf.variable_scope('value_net', auxiliary_name_scope=False, reuse=tf.AUTO_REUSE):
				self.value = tf.contrib.layers.fully_connected(
					embedder.embedding, num_outputs=1, activation_fn=None,
					reuse=tf.AUTO_REUSE, scope="value")
		else:
			with tf.variable_scope(name_scope, auxiliary_name_scope=False):
				with tf.variable_scope("value_net", auxiliary_name_scope=False):
					self.value = tf.contrib.layers.fully_connected(
						embedder.embedding, num_outputs=1, activation_fn=None, scope='value')

		with tf.variable_scope(name_scope, auxiliary_name_scope=False):
			with tf.variable_scope("value_net", auxiliary_name_scope=False):
				self.losses = tf.squared_difference(self.value, self.targets)
				self.loss = tf.reduce_sum(self.losses, name="value_loss")
				self.summaries.append(tf.summary.scalar('value loss', self.loss))

				self.predictions = {
					"value": self.value
				}

				# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
				self.optimizer = tf.train.AdamOptimizer(learning_rate=7e-5, name='Adam_value')
				self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
				# only take gradients for the trainable layers
				self.grads_and_vars = [[grad, var] \
					for grad, var in self.grads_and_vars \
					if grad is not None and \
					any([cs in var.name for cs in trainable_scopes])]
				if len(self.grads_and_vars) > 0:
					self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
						global_step=tf.contrib.framework.get_global_step())
