import numpy as np
import math
import random
import pandas as pd
import pylift
import itertools

class Fourier:
	def __init__(self, env, order=2):
		self.env = env
		self.order = order

		columns = []
		for i in range(env.dim_count):
			columns.append("x{}".format(i))
		columns.extend(["C", "E", "optimal_cause"])

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

	def run(self, epsilon=.1, alpha=.005, tau=.2, window=100, lifetime=30000):
		w = np.zeros((self.order + 1) ** self.env.dim_count * 2) # see Fourier paper
		
		bandit_result = np.array([])
		optimal_result = np.array([])
		difference = np.array([])
		incremental_diff = np.array([])

		drift_moments = np.array([])

		RP_history = []

		self.env.reset()

		s = self.env.get_new_state()
		for i in range(lifetime):
			
			# choose action (e-greedy)
			a = -1
			if np.random.binomial(1, epsilon):
				a = np.random.binomial(1, .5)
			else:
				feedback = np.array([])
				for a in range(2):
					feedback = np.append(feedback, self.estimate(s, a, w) - (tau * a))
				a = np.argmax(feedback)
			
					

			# apply action on environment
			n_s, r = self.env.choose_cause(a)
			if self.env.sudden_drift and self.env.current_drift:
				drift_moments = np.append(drift_moments, self.env.time)
			
			w = self.update_w(w, r - self.estimate(s, a, w), s, a, alpha)
			
			s_cause = self.get_sim_cause(s, tau)

			# register state and Random Policy r
			RP_C = np.random.binomial(1, .5)
			RP_E = self.what_if_cause(s, RP_C)

			RP_history.append([*s, RP_C, RP_E, s_cause])
			
			bandit_result = np.append(bandit_result, a)
			optimal_result = np.append(optimal_result, s_cause)
			difference = np.append(difference, abs(a - s_cause))
			incremental_diff = np.append(incremental_diff, np.average(difference[-window:]))
			
			s = n_s
		
		self.RP_run_history = self.RP_run_history.append(pd.DataFrame(RP_history, columns=self.RP_run_history_cols))

		return incremental_diff, drift_moments
			
	""" def gather_c(self):
		base = np.arange(self.order + 1)
		c = np.ndarray(((self.order + 1) ** self.env.dim_count, self.env.dim_count))
		shift = -1
		for i in range(len(c)):
			if not i%(self.order+1):
				shift += 1             
			c[i] = [base[i%(self.order+1)], base[(i + shift * 1)%(self.order+1)]] # extension for multiple dims!!
		return c """

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

