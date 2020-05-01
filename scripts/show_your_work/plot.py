import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import expected_max_performance


def main():
    # The N validation accuracies can go here:
    example_valid_perf = [.8, .75, .9, .24, .45, .6, .65, .67, .49, .49, .59, .52, .78,
                          .84, .63, .42, .85, .76, .68, .6]

    data = expected_max_performance.samplemax(example_valid_perf)

    # In most cases only one plot will be necessary.
    # These examples are meant to show what the plots look like with and without log-scaling the X-axis and
    # shading +/- the standard error (similar to standard deviation).
    one_plot(data, "Experiment Name", logx=False, plot_errorbar=False, avg_time=0)
    one_plot(data, "Experiment Name", logx=True, plot_errorbar=False, avg_time=0)
    one_plot(data, "Experiment Name", logx=False, plot_errorbar=True, avg_time=0)
    one_plot(data, "Experiment Name", logx=True, plot_errorbar=True, avg_time=0)

    # If the average runtime for the experiments is included, the X-axis will be time instead of trials.
    one_plot(data, "Experiment Name", logx=False, plot_errorbar=False, avg_time=10)
    one_plot(data, "Experiment Name", logx=True, plot_errorbar=False, avg_time=10)
    one_plot(data, "Experiment Name", logx=False, plot_errorbar=True, avg_time=10)
    one_plot(data, "Experiment Name", logx=True, plot_errorbar=True, avg_time=10)



def one_plot(data, data_name, logx=False, plot_errorbar=True, avg_time=0, performance_metric="accuracy"):
    # to set default values
    linestyle = "-"
    linewidth = 3
    errorbar_kind = 'shade'
    errorbar_alpha = 0.1
    fontsize = 16
    x_axis_time = avg_time != 0

    _, cur_ax = plt.subplots(1,1)
    cur_ax.set_title(data_name, fontsize=fontsize)
    cur_ax.set_ylabel("Expected validation " + performance_metric, fontsize=fontsize)

    if x_axis_time:
        cur_ax.set_xlabel("Training duration",fontsize=fontsize)
    else:
        cur_ax.set_xlabel("Hyperparameter assignments",fontsize=fontsize)

    if logx:
        cur_ax.set_xscale('log')


    means = data['mean']
    vars = data['var']
    max_acc = data['max']
    min_acc = data['min']

    if x_axis_time:
        x_axis = [avg_time * (i+1) for i in range(len(means))]
    else:
        x_axis = [i+1 for i in range(len(means))]

    if plot_errorbar:
        if errorbar_kind == 'shade':
            minus_vars = [x - y if (x - y) >= min_acc else min_acc for x,y in zip(means, vars)]
            plus_vars = [x + y if (x + y) <= max_acc else max_acc for x,y in zip(means, vars)]
            plt.fill_between(x_axis,
                             minus_vars,
                             plus_vars,
                             alpha=errorbar_alpha)
        else:
            cur_ax.errorbar(x_axis,
                            means,
                            yerr=vars,
                            linestyle=linestyle,
                            linewidth=linewidth)
    cur_ax.plot(x_axis,
                means,
                linestyle=linestyle,
                linewidth=linewidth)

    left, right = cur_ax.get_xlim()

    plt.xlim((left, right))
    plt.locator_params(axis='y', nbins=10)
    plt.tight_layout()

    save_plot(data_name, logx, plot_errorbar, avg_time)

def save_plot(data_name, logx, plot_errorbar, avg_time):
    name = "plots/{}_logx={}_errorbar={}_avgtime={}.pdf".format(data_name, logx, plot_errorbar, avg_time)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(name, dpi=300)


if __name__ == "__main__":
    main()
