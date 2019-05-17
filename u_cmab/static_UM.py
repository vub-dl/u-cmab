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
	
	def run(self, static_dataset_size=6000, total_experiment_count=14000, tau=.2, window=100, PH_alpha=.005, PH_init_length=200, PH_lambda=.1):
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
		
		start_location = 0
		
		model_decisions = np.zeros(self.RP_run_history.shape[0])
		optimal_decisions = self.RP_run_history['optimal_cause'].tolist()

		while self.RP_run_history.shape[0] - start_location > static_dataset_size:
			
			up, decisions = self.train_new_model(train_size=static_dataset_size, start_location=start_location, tau=tau)
			if len(decisions) + start_location != self.RP_run_history.shape[0]:
				raise ValueError(F"Amount of decisions ({len(decisions)}) from start location ({start_location}) does not match with RP_run_history ({self.RP_run_history.shape[0]})")
			
			model_decisions[start_location:self.RP_run_history.shape[0] + 1] = decisions

			start_location, _ = self.get_PH_location(decisions, start_of_section=start_location, train_size=static_dataset_size)
			print(start_location)

		difference = np.array([])
		static_id = np.array([])
		
		for i in range(len(model_decisions)):
			difference = np.append(difference, abs(model_decisions[i] - optimal_decisions[i]))
			static_id = np.append(static_id, np.average(difference[-window:]))
				
			"""	
				PH_windowed_avg = 0
				PH_total_avg = 0		
				PH_value = np.array([0])
				PH_total_avgs = np.array([0])
				PH_windowed_avgs = np.array([0])
				PH_N = 1						
				for i in range(len(decisions[static_dataset_size:])):
					if decisions[i + static_dataset_size] == 1:
						PH_total_avg = (1 - 1/(PH_N)) * PH_total_avg + 1/(PH_N) * self.RP_run_history.iloc[i + static_dataset_size + start_location,:]["E1"]
						PH_windowed_avg = (1 - PH_alpha) * PH_windowed_avg + PH_alpha * self.RP_run_history.iloc[i + static_dataset_size + start_location,:]["E1"]
						
						PH_value = np.append(PH_value, abs(PH_total_avg - PH_windowed_avg))
						PH_windowed_avgs = np.append(PH_windowed_avgs, PH_windowed_avg)
						PH_total_avgs = np.append(PH_total_avgs, PH_total_avg)

						PH_N += 1
					else:
						PH_value = np.append(PH_value, PH_value[-1])
						PH_windowed_avgs = np.append(PH_windowed_avgs, PH_windowed_avgs[-1])
						PH_total_avgs = np.append(PH_total_avgs, PH_total_avgs[-1])

					
				stop_locations = np.where(PH_value[1:]>PH_lambda)[0] + static_dataset_size
				if stop_locations.size == 0 or stop_locations[stop_locations>static_dataset_size + 1000].size == 0:
					stop_location = self.RP_run_history.shape[0]
				else:
					stop_location = int(np.min(stop_locations[stop_locations>static_dataset_size + 1000]))

				print(F"STOP AT {stop_location + start_location}")
				end_decisions = np.append(end_decisions, decisions[0:stop_location])
				s_ids = np.append(s_ids, static_id[0:stop_location])
				PH_v = np.append(PH_v, PH_value[1:stop_location])
				start_location += stop_location
			"""

		# parameters can be consulted using up.rand_search_.best_params_
		return up, static_id #, PH_v #, PH_total_avgs[1:], PH_windowed_avgs[1:], PH_value[1:]

	def get_PH_location(self, decisions, start_of_section=0, train_size=6000, PH_lambda=.2, PH_alpha=.05, PH_offset=2000):
		action_results = self.RP_run_history[start_of_section + train_size:]["E1"].tolist()
		decisions_after_random = decisions[train_size:]

		print(f"len(action_results) = {len(action_results)} AND len(decisions_after_r...) = {len(decisions_after_random)}")
		
		t_avg, w_avg, n = 0, 0, 1
		p_vals = np.array([0])

		for i in range(len(decisions_after_random)):
			if decisions_after_random[i] == 1:
				t_avg = (1 - 1/n)*t_avg + (1/n)*action_results[i]
				w_avg = (1 - PH_alpha)*w_avg + PH_alpha*action_results[i]

				p_val = abs(t_avg - w_avg)
				n+=1			
			else:
				p_val = p_vals[-1]
			
			p_vals = np.append(p_vals, p_val) # p_vals from start_of_section + train_size

		locations = np.where(p_vals[PH_offset:]>PH_lambda)[0] # evaluate p_vals > PH_lambda from start_of_section + train_size + PH_offset (PH_offset is necessary for stabilised values)
		if locations.size == 0:
			return self.RP_run_history.shape[0] + 1, p_vals
		return np.min(locations) + PH_offset + train_size + start_of_section, p_vals # return location relative to complete RP_run_history


	def train_new_model(self, train_size=6000, start_location=0, tau=.2):
		# Setup SKL model
		up = TransformedOutcome((
			self.RP_run_history.iloc[start_location:train_size + start_location,0:self.RP_run_history.shape[1] - 3], 
			self.RP_run_history.iloc[train_size + start_location:self.RP_run_history.shape[0]+1,0:self.RP_run_history.shape[1] - 3]), 
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
			*self.RP_run_history['C'][start_location:train_size+start_location].tolist(), # the random policy decisions in 'data collection' are also included for evaluation. 
			*list(map(lambda p: 1 if p>tau else 0, predictions))]		
		return up, decisions