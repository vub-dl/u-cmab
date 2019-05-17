import numpy as np
import math
import random
import pandas as pd
import pylift
import itertools

class Fourier:
	def __init__(self, env, order=2, regular=False):
		self.env = env
		self.order = order
		self.regular = regular

		columns = []
		for i in range(env.dim_count):
			columns.append("x{}".format(i))
		columns.extend(["C", "E", "E1", "E0", "optimal_cause"])

		self.RP_run_history_cols = columns
		self.RP_run_history = pd.DataFrame(columns=columns)

	def get_sim_cause(self, s, tau):
		uplift = self.env.get_sim_uplift(s)
		if uplift > tau:
			return 1
		else:
			return 0

	def what_if_cause(self, s, C):
		p = self.env.apply_noise(self.env.get_response_rate(C, s, drift=True))

		return np.random.binomial(1, p)

	def run(self, epsilon=.1, alpha=.005, tau=.2, window=100, lifetime=30000, decay_eps=.999):
		start_eps = 1.0

		w_u, w_r = np.zeros((self.order + 1) ** self.env.dim_count * 2), np.zeros((self.order + 1) ** self.env.dim_count * 2) # see Fourier paper
		
		bandit_result_u, bandit_result_r = np.array([]), np.array([])
		optimal_result = np.array([])
		difference_u, difference_r = np.array([]), np.array([])
		incremental_diff_u, incremental_diff_r = np.array([]), np.array([])

		total_reward_u, total_reward_r = np.array([]), np.array([])
		tot_r_u, tot_r_r = 0, 0
		exec_action_u, exec_action_r = np.array([]), np.array([])
		tot_ac_u, tot_ac_r = 0, 0

		drift_moments = np.array([])

		RP_history = []

		self.env.reset()

		s = self.env.get_new_state()
		for i in range(lifetime):
			
			# choose action (e-greedy)
			a_u, a_reg = -1, -1
			if np.random.binomial(1, max(epsilon, start_eps)):
				a_reg = np.random.binomial(1, .5)
				a_u = a_reg
			else:
				feedback = np.array([])
				reward = np.array([])
				for a in range(2):
					reward = np.append(reward, self.estimate(s, a, w_r))
					feedback = np.append(feedback, self.estimate(s, a, w_u) - (tau * a))
				a_reg = np.argmax(reward)
				a_u = np.argmax(feedback)
			
			start_eps*=decay_eps		

			# apply action on environment
			n_s, r_r = self.env.choose_cause(a_reg)
			r_u = self.what_if_cause(s, a_u)

			tot_r_u += r_u
			total_reward_u = np.append(total_reward_u, tot_r_u)
			tot_ac_u += 1 if a_u != 0 else 0
			exec_action_u = np.append(exec_action_u, tot_ac_u)

			tot_r_r += r_r
			total_reward_r = np.append(total_reward_r, tot_r_r)
			tot_ac_r += 1 if a_reg != 0 else 0
			exec_action_r = np.append(exec_action_r, tot_ac_r)


			if self.env.sudden_drift and self.env.current_drift:
				drift_moments = np.append(drift_moments, self.env.time)
			
			w_u = self.update_w(w_u, r_u - self.estimate(s, a_u, w_u), s, a_u, alpha)
			w_r = self.update_w(w_r, r_r - self.estimate(s, a_reg, w_r), s, a_reg, alpha)
			
			s_cause = self.get_sim_cause(s, tau)

			# register state and Random Policy r
			RP_C = np.random.binomial(1, .5)
			RP_E = self.what_if_cause(s, RP_C)
			E1 = self.what_if_cause(s, 1)
			E0 = self.what_if_cause(s, 0)

			RP_history.append([*s, RP_C, RP_E, E1, E0, s_cause])
			
			bandit_result_u = np.append(bandit_result_u, a_u)
			bandit_result_r = np.append(bandit_result_r, a_reg)
			optimal_result = np.append(optimal_result, s_cause)
			difference_u = np.append(difference_u, abs(a_u - s_cause))
			difference_r = np.append(difference_r, abs(a_reg - s_cause))
			incremental_diff_u = np.append(incremental_diff_u, np.average(difference_u[-window:]))
			incremental_diff_r = np.append(incremental_diff_r, np.average(difference_r[-window:]))
			
			s = n_s
		
		self.RP_run_history = self.RP_run_history.append(pd.DataFrame(RP_history, columns=self.RP_run_history_cols))

		return incremental_diff_u, drift_moments, incremental_diff_r, total_reward_u, exec_action_u, total_reward_r, exec_action_r
			
	def gather_c(self):
		cart_prod = list(itertools.product(np.arange(0, self.order+1), repeat=self.env.dim_count))
		return [list(elem) for elem in cart_prod]


	def basis(self, s, a):
		a_1 = np.zeros((self.order + 1) ** self.env.dim_count)
		a_0 = np.zeros((self.order + 1) ** self.env.dim_count)
		
		
		c = self.gather_c()
		basis = np.ndarray((self.order + 1) ** self.env.dim_count)
		
		for i in range(len(basis)):
			basis[i] = math.cos(math.pi * np.dot(c[i], s))
			
		if a:
			a_1 = basis
		else:
			a_0 = basis
		return np.append(a_0, a_1)

	def estimate(self, s, a, w):
		features = self.basis(s, a) 
		return np.dot(w, features)

	def update_w(self, w, t, s, a, alpha):		
		return w + (alpha * t * self.basis(s, a))

