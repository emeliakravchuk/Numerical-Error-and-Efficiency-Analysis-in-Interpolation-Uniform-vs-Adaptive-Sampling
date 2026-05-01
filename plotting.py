# plotting.py

import numpy as np
import matplotlib.pyplot as plt


def plot_subplots_compare_known_interp(known_signal=None, interp_signal=None, paired=None, 
                                        x_dense=None, sampling_rate=None, string_title=None):

    """
    plot_subplots(known_signal=None, interp_signal=None, paired=None, 
                    x_dense=None, sampling_rate=None, string_title=None)

    takes the known signal, interpolated signal, paired (x,y) data points, dense x grid, sampling_rate, title string
    plots side-by-side comparison for a given sampling rate 
    """
    if known_signal is None or interp_signal is None or paired is None:
        raise ValueError("\nAll function parameters have to be provided.\n")
    if x_dense is None or sampling_rate is None or string_title is None:
        raise ValueError("\nAll function parameters have to be provided.\n")
    if len(paired) < 2:
        raise ValueError("\nAt least two data points are required.\n")
    
    paired.sort()
    x, y = zip(*paired) # unzip
    x = np.array(x, dtype=float) # to float array
    y = np.array(y, dtype=float)

    if np.any(np.diff(x) <= 0):
        raise ValueError("\nThe provided x values must be strictly increasing.\n")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, # number rows, number cols
        figsize=(16, 8) # width by height 
        )
    
    fig.suptitle(f"Compare {string_title}\nFor Sampling Rate = {sampling_rate} samples/sec")

    ax1.plot(x_dense, known_signal(x_dense), linewidth=3)
    ax1.set_title("Known Signal")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (s)")
    ax1.axhline(0, linewidth=1)
    ax1.axvline(0, linewidth=1)

    ax2.plot(x_dense, interp_signal(x_dense), linewidth=3)
    if "Derivative" not in string_title:
        ax2.plot(x, y, 'o', markersize=8, label='Sampled nodes')
    ax2.set_title("Interpolated Signal")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Time (s)")
    ax2.axhline(0, linewidth=1)
    ax2.axvline(0, linewidth=1)

    plt.show()


def plot_subplots_compare_known_interp_error(known_signal=None, interp_signal=None, error_signal=None, 
                            deviation_signal=None, x_dense=None, sampling_rate=None, string_title=None):

    """
    plot_subplots_compare_known_interp_error(known_signal=None, interp_signal=None, error_signal=None, 
                            deviation_signal=None, x_dense=None, sampling_rate=None, string_title=None):

    takes the known signal, interp_signal, error_signal, deviation signal, dense x grid, sampling_rate, title string
    plots side-by-side comparisons for a given sampling rate 
    """
    if known_signal is None or interp_signal is None or error_signal is None:
        raise ValueError("\nAll function parameters have to be provided.\n")
    if deviation_signal is None or x_dense is None or sampling_rate is None or string_title is None:
        raise ValueError("\nAll function parameters have to be provided.\n")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, # number rows, number cols
        figsize=(14, 24) # width by height 
        )
    
    fig.suptitle(f"Compare {string_title}\nFor Sampling Rate = {sampling_rate} samples/sec")

    ax1.plot(x_dense, known_signal(x_dense), linewidth=3)
    ax1.set_title("Known Signal")
    ax1.set_ylabel("Amplitude")
    ax1.axhline(0, linewidth=1)
    ax1.axvline(0, linewidth=1)

    ax2.plot(x_dense, interp_signal(x_dense), linewidth=3)
    ax2.set_title("Interpolated Signal")
    ax2.set_ylabel("Amplitude")
    ax2.axhline(0, linewidth=1)
    ax2.axvline(0, linewidth=1)

    ax3.plot(x_dense, error_signal, linewidth=3)
    ax3.set_title("Error Signal")
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (s)")
    ax3.axhline(0, linewidth=1)
    ax3.axvline(0, linewidth=1)

    ax4.plot(x_dense, deviation_signal, linewidth=3)
    ax4.set_title("Deviation Signal")
    ax4.set_ylabel("Amplitude")
    ax4.set_xlabel("Time (s)")
    ax4.axhline(0, linewidth=1)
    ax4.axvline(0, linewidth=1)

    plt.show()


def plot_error_per_rate(stored_data=None, test_list=None, string_title=None, string_type="Samples"):

    """
    plot_error_per_rate(stored_data=None, test_list=None, string_title=None)

    stored_data is a dict of the form:
        stored_data[M] = {
            "avg_error": ...,
            "max_error": ...,
            "sampling_rate": ...,
            "samples": ...,
            }

    where M is the scaling of the signal frequency
    test_list is the list of M values

    plots 
    """
    if stored_data is None:
        raise ValueError("\nstored_data must be provided.\n")
    if test_list is None:
        raise ValueError("\ntest_list must be provided.\n")
    if string_title is None:
        raise ValueError("\nstring_title must be provided.\n")
    if len(stored_data) < 1:
        raise ValueError("\nstored_data cannot be empty.\n")
    if len(test_list) < 1:
        raise ValueError("\ntest_list cannot be empty.\n")

    M_vals = test_list
    x_vals = np.zeros(len(test_list), dtype=float)
    avg_err = np.zeros(len(test_list), dtype=float)
    max_dev = np.zeros(len(test_list), dtype=float)
    temp_dict = {}
    for i in range(len(M_vals)):
        temp_dict = stored_data[M_vals[i]]
        if string_type == "h Step": # to handle h steps case
            x_vals[i] = M_vals[i]
        elif string_type == "noise mag" or string_type == "filtered noise mag": # to handle noise mag case
            x_vals[i] = M_vals[i]
        elif string_type == "filter window size": # to handle filter window sweep
            x_vals[i] = M_vals[i]
        elif string_type == "alpha": # to handle alpha sweep
            x_vals[i] = M_vals[i]
        else:
            x_vals[i] = len(temp_dict["x_samples"])
        avg_err[i] = temp_dict["avg_error"]
        max_dev[i] = temp_dict["max_deviation"]


    fig, (ax1, ax2) = plt.subplots(
        1, 2, # number rows, number cols
        figsize=(16, 8) # width by height 
        )
    
    fig.suptitle(f"Interpolated Error vs {string_type} Used\n{string_title}")

    ax1.plot(x_vals, avg_err, linewidth=3)
    ax1.set_title(f"Average Error vs {string_type} Used")
    ax1.set_ylabel("Average Error")
    ax1.set_xlabel(f"{string_type} Used")
    ax1.axhline(0, linewidth=1)
    ax1.axvline(0, linewidth=1)

    ax2.plot(x_vals, max_dev, linewidth=3)
    ax2.set_title(f"Max Deviation vs {string_type} Used")
    ax2.set_ylabel("Max Deviation")
    ax2.set_xlabel(f"{string_type} Used")
    ax2.axhline(0, linewidth=1)
    ax2.axvline(0, linewidth=1)

    plt.show()
