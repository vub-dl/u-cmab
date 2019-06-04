import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from pylift.eval import UpliftEval
from pylift import TransformedOutcome

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import math
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 22
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20


class Train:
	def __init__(self, data, data_parser, train_ratio=.7):
		self.data = shuffle(data)		

		self.train_ratio = train_ratio
		self.train_size = math.floor(self.data.shape[0] * self.train_ratio)
		self.data_parser = data_parser
		self.train_data, self.test_data = self.data[0:self.train_size], self.data[self.train_size:]
		
		self.data['strat'] = self.data["segment"].astype(str) + self.data["visit"].astype(str)
		self.data_w = self.data.loc[self.data['segment'].isin(["Womens E-Mail",'No E-Mail'])]
		self.data_m = self.data.loc[self.data['segment'].isin(["Mens E-Mail",'No E-Mail'])]


		self.train_data, self.test_data = train_test_split(self.data, test_size=(1-train_ratio), random_state=0, stratify=self.data[['strat']])
		self.train_data_w, self.test_data_w = train_test_split(self.data_w, test_size=(1-train_ratio), random_state=0, stratify=self.data_w[['strat']])
		self.train_data_m, self.test_data_m = train_test_split(self.data_m, test_size=(1-train_ratio), random_state=0, stratify=self.data_m[['strat']])

	def nn(self, model, batch_size=32, epochs=50, loss_f=nn.MSELoss(), learning_rate=1e-5, weight_decay=.0003):
		opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
		contexts, treatments, rewards = self.data_parser(self.train_data)
		ds = TensorDataset(
			torch.tensor(contexts, dtype=torch.float), 
			torch.tensor(pd.get_dummies(treatments).values, dtype=torch.long), 
			torch.tensor(rewards, dtype=torch.float))
		dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

		for epoch in range(epochs):
			l = []
			for c, t, r in dl:
				target = model(c).detach()

				mask = t.byte()

				target[mask] = r
				est = model(c)

				loss = loss_f(est, target)
				l.append(loss.item())

				opt.zero_grad()
				loss.backward()
				opt.step()

			print(f"epoch {epoch} done with avg training loss of {np.average(l)}")

	
	def prep_rf_data(self, treatment_col, treatment, response, treat_value, response_value):
		df = self.data.loc[self.data[treatment_col].isin([treatment, response])]				
		train, test = train_test_split(df, test_size=(1 - self.train_ratio), random_state=0, stratify=df[['strat']])
		
		c_tr, t_tr, r_tr = self.data_parser(train)
		c_te, t_te, r_te = self.data_parser(test)

		t_tr = (t_tr.codes == treat_value).astype(int)
		t_te = (t_te.codes == treat_value).astype(int)

		return c_tr, t_tr, r_tr, c_te, t_te, r_te
	
	def rf(self,  segment='w', context_size=18):#treatment_col, treatment, response, treat_value, response_value, context_size=18): #segment="w",
		if segment == 'w':
			c_tr, t_tr, r_tr = self.data_parser(self.train_data_w)
			t_tr = t_tr.codes - 1
			c_te, t_te, r_te = self.data_parser(self.test_data_w)
			t_te = t_te.codes - 1
		else:
			c_tr, t_tr, r_tr = self.data_parser(self.train_data_m)
			t_tr = abs(t_tr.codes - 1)
			c_te, t_te, r_te = self.data_parser(self.test_data_m)
			t_te = abs(t_te.codes - 1)

		#c_tr, t_tr, r_tr, c_te, t_te, r_te = self.prep_rf_data(treatment_col, treatment, response, treat_value, response_value)

		td_tr = np.concatenate((c_tr, np.array([t_tr]).T, np.array([r_tr]).T), axis=1)
		td_te = np.concatenate((c_te, np.array([t_te]).T, np.array([r_te]).T), axis=1)

		cols = [f'x{i}' for i in range(context_size)]
		cols.extend(['t', 'y'])
		df_tr = pd.DataFrame(data=td_tr, columns=cols)
		df_te = pd.DataFrame(data=td_te, columns=cols)

		up = TransformedOutcome(
			(df_tr, df_te), 
			col_treatment='t', col_outcome='y', 
			stratify=None,
			sklearn_model=RandomForestRegressor)

		up.randomized_search(
			param_distributions={'max_depth': range(2,100), 'min_samples_split': range(2,1000)}, 
			n_iter=50, n_jobs=10)

		up.fit(**up.rand_search_.best_params_)

		return up


def qini(ts, rs, u_hats, ups, treatment_names=['T=1'], urf_label=['Uplift Random Forest'], urf_colors=['firebrick'], urf_colors_bands=['deeppink'], yticks=[0, 0.04]):
    fig, ax = plt.subplots()
    
    for r in range(len(u_hats)):
        
        b_x, b_y, s_x, s_y, rand = [], [], [], [], []

        for i in range(len(u_hats[r])):
            ue = UpliftEval(ts[r][i], rs[r][i], u_hats[r][i])
            x, y = ue.calc('aqini')
            x_stat, y_stat = ups[r][i].test_results_.calc(plot_type='aqini', n_bins=20)

            b_x.append(x)
            b_y.append(y)
            s_x.append(x_stat)
            s_y.append(y_stat)

            rand.append(getattr(ue, 'aqini_y')[-1])

        b_x_avg = np.average(b_x, axis=0)
        b_y_avg = np.average(b_y, axis=0)

        s_x_avg = np.average(s_x, axis=0)
        s_y_avg = np.average(s_y, axis=0)

        b_y_std = np.std(b_y, axis=0)
        s_y_std = np.std(s_y, axis=0)

        nn_label = "_no_label_" if r > 0 else 'Batch ANN'
        random_label = "_no_label_" if r < (len(u_hats)-1) else 'Random selection'
        
        ax.plot(b_x_avg, b_y_avg,'.-', color='dodgerblue', label=nn_label, markersize=13, linewidth=3, zorder=103)
        
        ax.plot(s_x_avg, s_y_avg,'.-', color=urf_colors[r], label=urf_label[r], markersize=13, linewidth=3, zorder=102)
        ax.plot([0,1], [0, np.average(rand)], '--', color='black', label=random_label, linewidth=3, dashes=(5,5), zorder=0)
        ax.text(.95, np.average(rand) - .1*yticks[-1], treatment_names[r], horizontalalignment='left')
        
        
        ax.fill_between(
            b_x_avg, b_y_avg - b_y_std, b_y_avg + b_y_std, **{"color": "turquoise", "alpha": .22, "linewidth":0, "zorder": 101}
        )

        ax.fill_between(
            s_x_avg, s_y_avg - s_y_std, s_y_avg + s_y_std, **{"color": urf_colors_bands[r], "alpha": .15, "linewidth":0, "zorder": 100}
        )

    ax.title.set_fontsize(30)
    ax.title.set_text("Hillstrom dataset")
    ax.title.set_fontfamily("sans-serif")
    ax.xaxis.label.set_fontsize(30)
    ax.xaxis.label.set_fontfamily("sans-serif")
    ax.xaxis.label.set_text('Fraction of data')

    ax.yaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontfamily("sans-serif")
    ax.yaxis.label.set_text('Uplift')

    ax.set_yticks(yticks)
    ax.set_xticks([0, .5, 1])

    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)

    ax.figure.set_size_inches(13.5, 10.5)

    font = fm.FontProperties(family='sans-serif', size=23)

    ax.legend(prop=font)

    return ax
