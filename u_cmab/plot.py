import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_results(avg_ids, stds, d_m=np.array([]), line_args=[{}], band_args=[{}], all_ids=np.array([[]]), all_lines_args=[{}], std_args=[{}], stop_RP=5000, stop_RP_args={}, stop_RP_text_args={}, drift_line_args={}):

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 22
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20

    fig, ax = plt.subplots(figsize=(13,10))

    for i in range(len(avg_ids)):
        ax.plot(avg_ids[i], **line_args[i])
        #plt.plot(stds[i], **std_args[i])
        ax.fill_between(
            np.linspace(0, len(avg_ids[i]), num=len(avg_ids[i])), 
            avg_ids[i] - stds[i], 
            avg_ids[i] + stds[i],
            **band_args[i] )

    for i in range(len(all_ids)):
        for j in range (len(all_ids[i])):
            ax.plot(all_ids[i][j], **all_lines_args[i])

    ax.set_xlabel("Experiment count")
    ax.set_ylabel("Uplift regret")
    ax.axis([0, len(avg_ids[0]), 0, 1])
    ax.set_yticks(np.linspace(0, 1, 5))
    
    xt = np.linspace(0, len(avg_ids[0]), 3)
    ax.set_xticks(xt[np.isin(xt, [*d_m, stop_RP], invert=True)])

    if stop_RP:
        ax.axvline(x=stop_RP, **stop_RP_args)
        ax.text(x=stop_RP-650, s="data collection", rotation=90, **stop_RP_text_args)
        ax.text(x=stop_RP+150, s="trained model", rotation=90, **stop_RP_text_args)        

        xt = ax.get_xticks()
        xt = np.append(xt, stop_RP)

        ax.set_xticks(xt)
        ax.get_xticklabels()[-1].set_color("tab:gray")

    ax.legend()

    for m in d_m:
        ax.axvline(x=m, **drift_line_args)
        xt = ax.get_xticks()[xt!=m]
        xt = np.append(xt, m)

        ax.set_xticks(xt)
        ax.get_xticklabels()[-1].set_color("tab:green")
    
    return fig, ax