import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

root = 'test crank-ms'
data = 'test crank-ms'
mode = 'svm'

dir_ = './{}/{}/{}'.format(root, data, mode)

df = pd.read_csv('{}/summary_results_svm.csv'.format(dir_))
df_roc = df.iloc[:, np.r_[0]]
df_pr = df.iloc[:, np.r_[1]]


def plot_box_swarm(data, plot_title):
    """Plot box-plot and swarm plot for data list.
 
    Args:
        data (list of list): List of lists with data to be plotted.
        y_axis_label (str): Y- axis label.
        x_labels (list of str): List with labels of x-axis.
        plot_title (str): Plot title.
        figure_name (str): Path to output figure.
         
    """
    sns.set(color_codes=True)
    plt.figure(figsize=(9, 6))
 
    # add title to plot
    plt.title(plot_title)
 
    # plot data on swarmplot and boxplot
    sns.swarmplot(data=data, color=".25")
    ax = sns.boxplot(data=data)
 
    # y-axis label
    ax.set(ylabel=y_axis_label)
    ax.set(xlabel=x_labels)
    ax.set()
    
    #reference line
    # # reference = []
    # left, right = plt.xlim()
    # plt.hlines(reference, xmin=left, xmax=right, color='r', linestyles='--')


x_labels = 'Model'
y_axis_label = 'AUC'

plot_box_swarm(df_roc, 'Box-Swarm ROC plot {}'.format(mode))
plt.ylim(0, 1.05)
plt.savefig('./{}/ROC swarm plot {}.png'.format(dir_, mode), bbox_inches='tight', dpi=600)


plot_box_swarm(df_pr, 'Box-Swarm PR plot {}'.format(mode))
plt.ylim(0, 1.05)
plt.savefig('./{}/PR swarm plot {}.png'.format(dir_, mode), bbox_inches='tight', dpi=600)

