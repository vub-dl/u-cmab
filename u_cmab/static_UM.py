import numpy as np
import pandas as pd

from pylift import TransformedOutcome
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class Static_UM:
	def __init__(self, RP_run_history=pd.DataFrame(columns=[])):
		"""
			Parameters
			----------
			RP_run_history : pd.dataframe
				The Random Process run history (RP_run_history) is a dataframe of following
				format:
				| *state | C (cause) | E (effect) | optimal_cause |
				
				Every C must be conducted randomly as this is a requirement of a dataset for
				uplift modeling, E is the registered outcome and optimal_cause is what the 
				oracle simulation would apply for C.
		"""


		self.RP_run_history = RP_run_history
	
	def run(self, static_dataset_size=6000, total_experiment_count=14000, tau=.2, window=100):
		if total_experiment_count > len(self.RP_run_history):
			raise ValueError("Not enough experiments in Run History (RP_run_history) to accommodate {}".format(total_experiment_count))

		"""
			Parameters
			----------
			static_dataset_size : int
				This is the dataset size you wish to use as a training set for the static
				model. Of course, the larger you set this size, the longer the situation
				will perform randomly (as indicated in the plots) and the lesser data can
				be used as a test set.
			total_experiment_count : int
				The amount of experiments considered in this run. While it does not have
				to be the same as len(self.RP_run_history), it cannot exceed the length.
			tau : float
				As with the bandit setting, tau will be used to translate uplift predictions
				to a decision output. In order to truly compare the bandit with a static model,
				the outputs must tell the same.
			window : int
				When the average incremental difference is calculated, window is the size
				of its moving window. 
		"""
		
		# Setup SKL model
		up = TransformedOutcome((
			self.RP_run_history.iloc[:,0:self.RP_run_history.shape[0] - 1][0:static_dataset_size], 
			self.RP_run_history.iloc[:,0:self.RP_run_history.shape[0] - 1][static_dataset_size:total_experiment_count+1]), 
			col_treatment='C', col_outcome='E', 
			stratify=None, # No stratification as we need to match states in test set with Fourier
			sklearn_model=RandomForestRegressor)
		
		# Random search over model parameters (3 folds * 50 iterations = 150 fits)
		# eventual parameters are returned for documentation
		up.randomized_search(
			param_distributions={'max_depth': range(2,100), 'min_samples_split': range(2,1000)}, 
			n_iter=50, n_jobs=10)

		up.fit(**up.rand_search_.best_params_)


		# x_test is
		predictions = up.model.predict(up.x_test)		
		decisions = [
			*self.RP_run_history['C'].head(static_dataset_size).tolist(), # the random policy decisions in 'data collection' are also included for evaluation. 
			*list(map(lambda p: 1 if p>tau else 0, predictions))]		
		optimal_decisions = self.RP_run_history['optimal_cause'].tolist()

		difference = np.array([])
		static_id = np.array([])
		for i in range(len(decisions)):
			difference = np.append(difference, abs(decisions[i] - optimal_decisions[i]))
			static_id = np.append(static_id, np.average(difference[-window:]))

		# parameters can be consulted using up.rand_search_.best_params_
		return up, static_id

