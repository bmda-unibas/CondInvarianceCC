#  Author: Maxim Samarin
#  Contact: maxim.samarin@unibas.ch
#  Date: 20.09.2021
#

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def bin_data(values, num_bins=10):
    t_min = min(values)
    t_max = max(values)
    delta = (t_max - t_min) / float(num_bins)

    array = np.zeros(num_bins)

    for i in range(num_bins):
        array[i] = t_min
        t_min = t_min + delta

    bins = np.digitize(values, array)

    return bins


def plot_input_data(x, y, original_dim):

    matplotlib.rcParams.update({'font.size': 18})

    bins = bin_data(y, num_bins=10)

    if original_dim == 2:
        fig, ax = plt.subplots(1,1, figsize=(5,5))

        ax.scatter(x[:,0], x[:, 1], marker=".", c=np.squeeze(bins), cmap='viridis_r')

        ax.axhline(y=0, xmin=-1, xmax=1, c='black', linestyle='--')
        ax.axvline(x=0, ymin=-1, ymax=1, c='black', linestyle='--')

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')

    elif original_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')

        ax.scatter(x[:,0], x[:, 1], x[:, 2], marker=".", c=np.squeeze(bins), cmap='viridis_r')

    plt.savefig("plots/input_{}D_before_high-dim-mapping.png".format(original_dim), bbox_inches='tight')
    plt.close()


def plot_selected_dimensions(variance, stddev_noise=1, pz_y=3, pz=8, iter=None, check_point_name='test', mode_suffix='test'):
    fig, ax = plt.subplots(1, 1)

    ax.bar(np.arange(start=0, stop=pz, step=1), np.sqrt(variance), color='orange')
    ax.bar(np.arange(start=0, stop=pz, step=1), stddev_noise, color='darkgrey')
    ax.axvline(x=pz_y - 0.5, ymin=0, ymax=10000, c='red', linestyle='--')

    if iter:
        ax.set_title('Iter {} Selected Dim.'.format(iter))
    else:
        ax.set_title('Selected Dim')

    plt.savefig("plots/{}_{}_selected-dimensions.png".format(check_point_name, mode_suffix), bbox_inches='tight')
    plt.close()
