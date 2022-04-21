"""
@author: himanshusahni
Modified from a3c code by dennybritz

Worker thread for a3c training.
"""

import sys
import os
import itertools
import collections
from threading import local
import numpy as np
import tensorflow as tf

from inspect import getsourcefile
from rl_agents.utils.utils import make_copy_params_op

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done", 'unsquashed'])

def make_train_op(local_estimator, global_estimator, key_func, step):
	"""
	Creates an op that applies local estimator gradients
	to the global estimator.
	"""
	local_grads, local_vars = zip(*list(sorted(
		local_estimator.grads_and_vars,
		key=lambda grad_and_var: key_func(grad_and_var[1]))))
	# Clip gradients
	local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
	_, global_vars = zip(*global_estimator.grads_and_vars)
	global_vars = list(sorted(global_vars, key=key_func))
	local_global_grads_and_vars = list(zip(local_grads, global_vars))
	return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
					global_step=step)


class Worker(object):
	"""
	An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.
	Args:
		name: A unique name for this worker.
		env: The environment used by this worker.
		policy_net: Instance of the globally shared policy network.
		value_net: Instance of the globally shared value network.
		global_counter: Iterator that holds the global step.
		max_global_steps: If set, stop coordinator when global_counter > max_global_steps.
	"""
	def __init__(
			self, name, task_id, env, policy_net, value_net, global_scopes,
			global_counter, task_counter, task_step, max_global_steps=None):
		self.name = name
		self.task_id = task_id
		self.discount_factor = 0.99
		self.max_global_steps = max_global_steps
		self.global_step = tf.contrib.framework.get_global_step()
		self.global_policy_net = policy_net
		self.global_value_net = value_net
		self.global_counter = global_counter
		self.task_counter = task_counter
		self.env = env
		self.epochs = 0
		self.task_step = task_step
		self.global_scopes = global_scopes

		self.state = None

	def create_copy_ops(self, key_func):
		'''
		hook op global/local copy
		'''
		global_variables = []
		for gs in self.global_scopes:
			global_variables += tf.contrib.slim.get_variables(
				scope=gs,
				collection=tf.GraphKeys.TRAINABLE_VARIABLES)
		local_variables = tf.contrib.slim.get_variables(
			scope=self.name+'/',
			collection=tf.GraphKeys.TRAINABLE_VARIABLES)
		# print(global_variables)
		# print(local_variables)
		# quit()
		self.copy_params_op = make_copy_params_op(global_variables, local_variables, key_func)

	def create_train_ops(self, key_func):
		'''
		hook up the local/global gradients
		'''
		self.vnet_train_op = make_train_op(
			self.value_net,
			self.global_value_net,
			key_func,
			self.task_step)
		self.pnet_train_op = make_train_op(
			self.policy_net,
			self.global_policy_net,
			key_func,
			self.task_step)

	def run(self, sess, coord, t_max, writer):
		with sess.as_default(), sess.graph.as_default():
			# Initial state
			self.state = self.env.reset()
			self.ep_r = 0
			self.ep_s = 0
			try:
				while not coord.should_stop():
					# Copy Parameters from the global networks
					sess.run(self.copy_params_op)

					# Collect some experience
					transitions, task_t, global_t = self.run_n_steps(t_max, sess)

					if self.max_global_steps is not None and global_t >= self.max_global_steps:
						tf.logging.info("Reached global step {}. Stopping.".format(global_t))
						coord.request_stop()
						return

					# Update the global networks
					smr = self.update(transitions, sess)
					writer.add_summary(smr, global_t)
					writer.flush()

			except tf.errors.CancelledError:
				return

	def _policy_net_predict(self, state, sess):
		# feed in the state to all embedders
		feed_dict = {}
		for i in range(len(self.policy_net.states)):
			feed_dict[self.policy_net.states[i]] = [state[self.policy_net.state_idx]]
		preds = sess.run(self.policy_net.predictions, feed_dict)
		return preds["action"][0], preds['unsquashed'][0]

	def _value_net_predict(self, state, sess):
		# feed in the state to all embedders
		feed_dict = {}
		for i in range(len(self.value_net.states)):
			feed_dict[self.value_net.states[i]] = [state]
		preds = sess.run(self.value_net.predictions, feed_dict)
		return preds["value"][0]

	def run_n_steps(self, n, sess):
		transitions = []
		for _ in range(n):
			# Take a step
			action, unsquashed = self._policy_net_predict(self.state, sess)
			next_state, reward, done = self.env.step(action)

			# Store transition
			transitions.append(Transition(state=self.state[self.policy_net.state_idx], 
											action=action, 
											reward=reward, 
											next_state=next_state[self.policy_net.state_idx], 
											done=done,
											unsquashed=unsquashed))

			# Increase local and global counters
			global_t = next(self.global_counter)
			task_t = next(self.task_counter)
			self.ep_r += reward
			self.ep_s += 1

			if done:
				sys.stderr.write("Thread {}, task {}, Task steps {}, Global steps {}, episode reward {}, episode steps {}\n".format(self.name, self.task_id, task_t, global_t, self.ep_r, self.ep_s))
				self.ep_r = 0
				self.ep_s = 0
				if task_t > (self.epochs + 1) * 50000:
					self.epochs += 1
				self.state = self.env.reset()
				break
			else:
				self.state = next_state
		return transitions, task_t , global_t

	def update(self, transitions, sess):
		"""
		Updates global policy and value networks based on collected experience
		Args:
			transitions: A list of experience transitions
			sess: A Tensorflow session
		"""

		# If we episode was not done we bootstrap the value from the last state
		next_value = np.array([0.0])
		if not transitions[-1].done:
			next_value = self._value_net_predict(transitions[-1].next_state, sess)

		# Accumulate minibatch exmaples
		states = []
		advantages = []
		value_targets = []
		actions = []
		unsquasheds = []

		for transition in transitions[::-1]:
			next_value = transition.reward + self.discount_factor * next_value
			# GAE
			advantage = (next_value - self._value_net_predict(transition.state, sess))
			# Accumulate updates
			states.append(transition.state)
			actions.append(transition.action)
			advantages.append(advantage)
			value_targets.append(next_value)
			unsquasheds.append(transition.unsquashed)

		feed_dict = {
			self.policy_net.advantages: advantages,
			self.policy_net.act_prev: actions,
			self.value_net.targets: value_targets,
			self.policy_net.unsquashed_mu_prev: unsquasheds
		}
		for i in range(len(self.policy_net.states)):
			feed_dict[self.policy_net.states[i]] = np.array(states)
		for i in range(len(self.value_net.states)):
			feed_dict[self.value_net.states[i]] = np.array(states)

		# Train the global estimators using local gradients
		smr, global_step, pnet_loss, vnet_loss, _, _ = sess.run([
			self.summary,
			self.global_step,
			self.policy_net.loss,
			self.value_net.loss,
			self.pnet_train_op,
			self.vnet_train_op
		], feed_dict)

		return smr

	def create_summary(self):
		summaries = []
		summaries.extend(self.policy_net.summaries)
		summaries.extend(self.value_net.summaries)
		self.summary = tf.summary.merge(summaries)