# linear_interp_tests_uniform_sampling.py

import numpy as np
from cubic_spline import Cubic_Spline
from plotting import plot_subplots_compare_known_interp, plot_subplots_compare_known_interp_error
from methods import (centered_difference_2nd_order, centered_difference_4th_order, 
                     white_guassian_noise, linear_interp, averaging_noise_filter, error_analysis)

def run_test1(interp_type=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None):

    """
    run_test1(interp_type=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None)

    M is scaling of the signal frequency

    plotting will happen unless disabled 

    returns stored_data dict
    """
    if M is None or interp_type is None or string_title_known_to_interp is None or string_title_error is None:
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if M < 2:
        raise ValueError("\nM must be >= 2.\n")
    plot_flag = bool(plot_flag)
    
    stored_data = {}
    stored_data[M] = None
    
    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    interp_signal = None
    if interp_type == "linear":
        interp_signal = linear_interp(list(zip(x_samples, y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(x_samples, y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        interp_signal = cubic.cubic_spline_interpolation(coeff)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    if plot_flag:
        plot_subplots_compare_known_interp(signal, interp_signal, list(zip(x_samples, y_samples)), 
                                            x_dense, sampling_rate, string_title_known_to_interp)

    error_analysis_dict = {}
    error_analysis_dict = error_analysis(known_signal=signal, interp_signal=interp_signal, 
                                                                            x_dense=x_dense) 

    if plot_flag:
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=interp_signal,
                                                error_signal=error_analysis_dict["error_signal"], 
                                                deviation_signal=error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=string_title_error)
    
    print(f"\nTest 1 results for sampling rate {sampling_rate} samples/sec:\n")
    print(f"Average Error: {error_analysis_dict["avg_error"]}")
    print(f"Max Error: {error_analysis_dict["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict["max_deviation_time"]} seconds\n")

    temp_dict = {}
    temp_dict = error_analysis_dict.copy()
    temp_dict["sampling_rate"] = sampling_rate
    temp_dict["frequency"] = frequency
    temp_dict["x_samples"] = x_samples
    stored_data[M] = temp_dict

    return stored_data[M]


def run_test2_1(interp_type=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None):

    """
    run_test2_1(interp_type=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None)

    M is scaling of the signal frequency

    plotting will happen unless disabled

    returns list [ stored_data dict order2, stored_data dict order4 ]
    """
    if M is None or interp_type is None or string_title_known_to_interp is None or string_title_error is None:
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if M < 2:
        raise ValueError("\nM must be >= 2.\n")
    plot_flag = bool(plot_flag)
    
    stored_data_order2 = {}
    stored_data_order2[M] = None
    stored_data_order4 = {}
    stored_data_order4[M] = None

    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)
    d_dt_signal = lambda t: 1/3 * (2*np.pi * frequency) * np.cos(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    fixed_h = (x_end - x0) * 0.01

    interp_signal = None
    if interp_type == "linear":
        interp_signal = linear_interp(list(zip(x_samples, y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(x_samples, y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        interp_signal = cubic.cubic_spline_interpolation(coeff)

    d_dt_interp_signal_order2 = centered_difference_2nd_order(interp_signal, fixed_h)
    d_dt_interp_signal_order4 = centered_difference_4th_order(interp_signal, fixed_h)

    dense = 2000
    x_dense_order2 = np.linspace(x0 + fixed_h, x_end - fixed_h, dense)
    x_dense_order4 = np.linspace(x0 + 2*fixed_h, x_end - 2*fixed_h, dense)

    error_analysis_dict_order2 = {}
    error_analysis_dict_order2 = error_analysis(known_signal=d_dt_signal, 
                                                interp_signal=d_dt_interp_signal_order2, 
                                                                    x_dense=x_dense_order2)

    error_analysis_dict_order4 = {}
    error_analysis_dict_order4 = error_analysis(known_signal=d_dt_signal, 
                                                interp_signal=d_dt_interp_signal_order4, 
                                                                    x_dense=x_dense_order4)
    
    if plot_flag:
        # order 2 plots
        plot_subplots_compare_known_interp(d_dt_signal, d_dt_interp_signal_order2, 
                                            list(zip(x_samples, y_samples)), 
                                            x_dense_order2, sampling_rate, 
                                            (string_title_known_to_interp + ", Order 2"))
        
        plot_subplots_compare_known_interp_error(known_signal=d_dt_signal, 
                                                interp_signal=d_dt_interp_signal_order2,
                                                error_signal=error_analysis_dict_order2["error_signal"], 
                                                deviation_signal=error_analysis_dict_order2["deviation_signal"],
                                                x_dense=x_dense_order2, sampling_rate=sampling_rate,
                                                string_title=(string_title_error+ ", Order 2"))
        # order 4 plots
        plot_subplots_compare_known_interp(d_dt_signal, d_dt_interp_signal_order4, 
                                           list(zip(x_samples, y_samples)), 
                                            x_dense_order4, sampling_rate, 
                                            (string_title_known_to_interp + ", Order 4"))
        
        plot_subplots_compare_known_interp_error(known_signal=d_dt_signal, 
                                                 interp_signal=d_dt_interp_signal_order4,
                                                error_signal=error_analysis_dict_order4["error_signal"], 
                                                deviation_signal=error_analysis_dict_order4["deviation_signal"],
                                                x_dense=x_dense_order4, sampling_rate=sampling_rate,
                                                string_title=(string_title_error+ ", Order 4"))

    print(f"\nTest 2.1 results for sampling rate {sampling_rate} samples/sec:\nOrder 2 Derivative\n")
    print(f"Average Error: {error_analysis_dict_order2["avg_error"]}")
    print(f"Max Error: {error_analysis_dict_order2["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict_order2["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict_order2["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict_order2["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict_order2["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict_order2["max_deviation_time"]} seconds\n")

    print(f"\nTest 2.1 results for sampling rate {sampling_rate} samples/sec:\nOrder 4 Derivative\n")
    print(f"Average Error: {error_analysis_dict_order4["avg_error"]}")
    print(f"Max Error: {error_analysis_dict_order4["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict_order4["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict_order4["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict_order4["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict_order4["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict_order4["max_deviation_time"]} seconds\n")

    temp_dict_order2 = {}
    temp_dict_order4 = {}
    temp_dict_order2 = error_analysis_dict_order2.copy()
    temp_dict_order4 = error_analysis_dict_order4.copy()
    temp_dict_order2["sampling_rate"] = sampling_rate
    temp_dict_order2["frequency"] = frequency
    temp_dict_order2["x_samples"] = x_samples
    temp_dict_order4["sampling_rate"] = sampling_rate
    temp_dict_order4["frequency"] = frequency
    temp_dict_order4["x_samples"] = x_samples

    stored_data_order2[M] = temp_dict_order2
    stored_data_order4[M] = temp_dict_order4

    return stored_data_order2[M], stored_data_order4[M]


def run_test2_2(interp_type=None, h=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None):

    """
    run_test2_2(interp_type=None, h=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None)

    M is scaling of the signal frequency

    plotting will happen unless disabled

    returns list [ stored_data dict order2, stored_data dict order4 ]
    """
    if any(v is None for v in [M, interp_type, h, string_title_known_to_interp, string_title_error]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if not isinstance(h, (float, int)):
        raise TypeError("\nh must be float or int type.\n")
    if h <= 0:
        raise ValueError("\nh must be > 0.\n")
    plot_flag = bool(plot_flag)
    
    stored_data_order2 = {}
    stored_data_order4 = {}

    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)
    d_dt_signal = lambda t: 1/3 * (2*np.pi * frequency) * np.cos(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    interp_signal = None
    if interp_type == "linear":
        interp_signal = linear_interp(list(zip(x_samples, y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(x_samples, y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        interp_signal = cubic.cubic_spline_interpolation(coeff)

    d_dt_interp_signal_order2 = centered_difference_2nd_order(interp_signal, h)
    d_dt_interp_signal_order4 = centered_difference_4th_order(interp_signal, h)

    dense = 2000
    x_dense_order2 = np.linspace(x0 + h, x_end - h, dense)
    x_dense_order4 = np.linspace(x0 + 2*h, x_end - 2*h, dense)

    error_analysis_dict_order2 = {}
    error_analysis_dict_order2 = error_analysis(known_signal=d_dt_signal, 
                                                interp_signal=d_dt_interp_signal_order2, 
                                                                    x_dense=x_dense_order2)

    error_analysis_dict_order4 = {}
    error_analysis_dict_order4 = error_analysis(known_signal=d_dt_signal, 
                                                interp_signal=d_dt_interp_signal_order4, 
                                                                    x_dense=x_dense_order4)
    
    if plot_flag:
        # order 2 plots
        plot_subplots_compare_known_interp(d_dt_signal, d_dt_interp_signal_order2, 
                                            list(zip(x_samples, y_samples)), 
                                            x_dense_order2, sampling_rate, 
                                            (string_title_known_to_interp + f", Order 2, h={h}"))
        
        plot_subplots_compare_known_interp_error(known_signal=d_dt_signal, 
                                                interp_signal=d_dt_interp_signal_order2,
                                                error_signal=error_analysis_dict_order2["error_signal"], 
                                                deviation_signal=error_analysis_dict_order2["deviation_signal"],
                                                x_dense=x_dense_order2, sampling_rate=sampling_rate,
                                                string_title=(string_title_error+ f", Order 2, h={h}"))
        # order 4 plots
        plot_subplots_compare_known_interp(d_dt_signal, d_dt_interp_signal_order4, 
                                            list(zip(x_samples, y_samples)), 
                                            x_dense_order4, sampling_rate, 
                                            (string_title_known_to_interp + f", Order 4, h={h}"))
        
        plot_subplots_compare_known_interp_error(known_signal=d_dt_signal, 
                                                interp_signal=d_dt_interp_signal_order4,
                                                error_signal=error_analysis_dict_order4["error_signal"], 
                                                deviation_signal=error_analysis_dict_order4["deviation_signal"],
                                                x_dense=x_dense_order4, sampling_rate=sampling_rate,
                                                string_title=(string_title_error+ f", Order 4, h={h}"))

    print(f"\nTest 2.2 results for sampling rate {sampling_rate} samples/sec:\nOrder 2 Derivative\n")
    print(f"Average Error: {error_analysis_dict_order2["avg_error"]}")
    print(f"Max Error: {error_analysis_dict_order2["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict_order2["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict_order2["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict_order2["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict_order2["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict_order2["max_deviation_time"]} seconds\n")

    print(f"\nTest 2.2 results for sampling rate {sampling_rate} samples/sec:\nOrder 4 Derivative\n")
    print(f"Average Error: {error_analysis_dict_order4["avg_error"]}")
    print(f"Max Error: {error_analysis_dict_order4["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict_order4["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict_order4["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict_order4["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict_order4["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict_order4["max_deviation_time"]} seconds\n")

    temp_dict_order2 = {}
    temp_dict_order4 = {}
    temp_dict_order2 = error_analysis_dict_order2.copy()
    temp_dict_order4 = error_analysis_dict_order4.copy()
    temp_dict_order2["sampling_rate"] = sampling_rate
    temp_dict_order2["frequency"] = frequency
    temp_dict_order2["x_samples"] = x_samples
    temp_dict_order4["sampling_rate"] = sampling_rate
    temp_dict_order4["frequency"] = frequency
    temp_dict_order4["x_samples"] = x_samples

    stored_data_order2[h] = temp_dict_order2
    stored_data_order4[h] = temp_dict_order4

    return stored_data_order2[h], stored_data_order4[h]



def run_test3_1(interp_type=None, mag=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None):

    """
    run_test3_1(interp_type=None, mag=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None)

    M is scaling of the signal frequency

    plotting will happen unless disabled

    returns dict stored_data_noisy
    """
    if any(v is None for v in [M, interp_type, mag, string_title_known_to_interp, string_title_error]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if not isinstance(mag, (float, int)):
        raise TypeError("\nmag must be float or int type.\n")
    if mag <= 0:
        raise ValueError("\nmag must be > 0.\n")
    plot_flag = bool(plot_flag)

    stored_data_noisy = {}

    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    noise_signal = mag * white_guassian_noise(samples)
    noisy_y_samples = y_samples + noise_signal
    noisy_x_samples = x_samples.copy()

    noisy_interp_signal = None
    if interp_type == "linear":
        noisy_interp_signal = linear_interp(list(zip(noisy_x_samples, noisy_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(noisy_x_samples, noisy_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        noisy_interp_signal = cubic.cubic_spline_interpolation(coeff)

    error_analysis_dict_noisy = {}
    error_analysis_dict_noisy = error_analysis(known_signal=signal, interp_signal=noisy_interp_signal, 
                                                                            x_dense=x_dense)
    
    if plot_flag:
        # with noise of magnitude = mag
        plot_subplots_compare_known_interp(signal, noisy_interp_signal, 
                                            list(zip(noisy_x_samples, noisy_y_samples)), 
                                            x_dense, sampling_rate, 
                                            (string_title_known_to_interp + 
                                            f", with White Gaussian Noise, mag={mag}"))
        
        plot_subplots_compare_known_interp_error(known_signal=signal, 
                                                interp_signal=noisy_interp_signal,
                                                error_signal=error_analysis_dict_noisy["error_signal"], 
                                                deviation_signal=error_analysis_dict_noisy["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error + 
                                                f", with White Gaussian Noise, mag={mag}"))

    print(f"\nTest 3.1 results for sampling rate {sampling_rate} samples/sec:\nWhite Gaussian Noise, mag={mag}\n")
    print(f"Average Error: {error_analysis_dict_noisy["avg_error"]}")
    print(f"Max Error: {error_analysis_dict_noisy["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict_noisy["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict_noisy["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict_noisy["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict_noisy["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict_noisy["max_deviation_time"]} seconds\n")

    temp_dict_noisy = {}
    temp_dict_noisy = error_analysis_dict_noisy.copy()
    temp_dict_noisy["sampling_rate"] = sampling_rate
    temp_dict_noisy["frequency"] = frequency
    temp_dict_noisy["x_samples"] = x_samples

    stored_data_noisy[mag] = temp_dict_noisy

    return stored_data_noisy[mag]


def run_test3_2(interp_type=None, mag=None, h=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None):

    """
    run_test3_2(interp_type=None, mag=None, h=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None)

    M is scaling of the signal frequency

    plotting will happen unless disabled

    returns dict stored_data_noisy
    """
    if any(v is None for v in [M, interp_type, mag, h, string_title_known_to_interp, string_title_error]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if not isinstance(mag, (float, int)):
        raise TypeError("\nmag must be float or int type.\n")
    if mag <= 0:
        raise ValueError("\nmag must be > 0.\n")
    if not isinstance(h, (float, int)):
        raise TypeError("\nh must be float or int type.\n")
    if h <= 0:
        raise ValueError("\nh must be > 0.\n")
    plot_flag = bool(plot_flag)

    stored_data_noisy = {}

    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)
    d_dt_signal = lambda t: 1/3 * (2*np.pi * frequency) * np.cos(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    dense = 2000
    x_dense = np.linspace(x0 + 2*h, x_end - 2*h, dense)

    noise_signal = mag * white_guassian_noise(samples)
    noisy_y_samples = y_samples + noise_signal
    noisy_x_samples = x_samples.copy()

    noisy_interp_signal = None
    if interp_type == "linear":
        noisy_interp_signal = linear_interp(list(zip(noisy_x_samples, noisy_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(noisy_x_samples, noisy_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        noisy_interp_signal = cubic.cubic_spline_interpolation(coeff)

    d_dt_noisy_interp_signal = centered_difference_4th_order(noisy_interp_signal, h)

    error_analysis_dict_noisy = {}
    error_analysis_dict_noisy = error_analysis(known_signal=d_dt_signal, interp_signal=d_dt_noisy_interp_signal, 
                                                                                                x_dense=x_dense)
    
    if plot_flag:
        # with noise of magnitude = mag
        plot_subplots_compare_known_interp(d_dt_signal, d_dt_noisy_interp_signal, 
                                            list(zip(noisy_x_samples, noisy_y_samples)), 
                                            x_dense, sampling_rate, 
                                            (string_title_known_to_interp + 
                                            f", with White Gaussian Noise, mag={mag}"))
        
        plot_subplots_compare_known_interp_error(known_signal=d_dt_signal, 
                                                interp_signal=d_dt_noisy_interp_signal,
                                                error_signal=error_analysis_dict_noisy["error_signal"], 
                                                deviation_signal=error_analysis_dict_noisy["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error + 
                                                f", with White Gaussian Noise, mag={mag}"))

    print(f"\nTest 3.2 results for sampling rate {sampling_rate}" + 
          f"samples/sec:\nWhite Gaussian Noise, mag={mag}, Order 4 Derivative\n")
    print(f"Average Error: {error_analysis_dict_noisy["avg_error"]}")
    print(f"Max Error: {error_analysis_dict_noisy["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict_noisy["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict_noisy["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict_noisy["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict_noisy["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict_noisy["max_deviation_time"]} seconds\n")

    temp_dict_noisy = {}
    temp_dict_noisy = error_analysis_dict_noisy.copy()
    temp_dict_noisy["sampling_rate"] = sampling_rate
    temp_dict_noisy["frequency"] = frequency
    temp_dict_noisy["x_samples"] = x_samples

    stored_data_noisy[mag] = temp_dict_noisy

    return stored_data_noisy[mag]



def run_test4_1(interp_type=None, filter_window=5, mag=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None, 
                                                    filter_sweep_flag=False):

    """
    run_test4_1(interp_type=None, filter_window=5, mag=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None,
                                                    filter_sweep_flag=False)

    M is scaling of the signal frequency

    plotting will happen unless disabled

    returns dict stored_data_noisy_filtered
    """
    if any(v is None for v in [M, interp_type, mag, string_title_known_to_interp, string_title_error]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if not isinstance(mag, (float, int)):
        raise TypeError("\nmag must be float or int type.\n")
    if mag <= 0:
        raise ValueError("\nmag must be > 0.\n")
    if filter_window is None:
        raise ValueError("\nfilter_window cannot be None.\n")
    if not isinstance(filter_window, int):
        raise TypeError("\nfilter_window must be int type.\n")
    if filter_window < 1:
        raise ValueError("\nfilter_window must be >= 1.\n")
    if filter_window % 2 == 0:
        raise ValueError("\nfilter_window must be odd for symmetric averaging.\n")
    plot_flag = bool(plot_flag)
    filter_sweep_flag = bool(filter_sweep_flag)

    stored_data_noisy_filtered = {}

    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    noise_signal = mag * white_guassian_noise(samples)
    noisy_y_samples = y_samples + noise_signal
    noisy_x_samples = x_samples.copy()

    # implement filter
    filter_window = filter_window
    filtered_noisy_x_samples, filtered_noisy_y_samples = averaging_noise_filter(M=filter_window,
                                            paired=list(zip(noisy_x_samples, noisy_y_samples)))
    
    filtered_noisy_interp_signal = None
    if interp_type == "linear":
        filtered_noisy_interp_signal = linear_interp(list(zip(filtered_noisy_x_samples, filtered_noisy_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(filtered_noisy_x_samples, filtered_noisy_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        filtered_noisy_interp_signal = cubic.cubic_spline_interpolation(coeff)

    error_analysis_dict_noisy_filtered = {}
    error_analysis_dict_noisy_filtered = error_analysis(known_signal=signal, 
                                                        interp_signal=filtered_noisy_interp_signal, 
                                                                                    x_dense=x_dense)
    
    if plot_flag:
        # with noise of magnitude = mag
        plot_subplots_compare_known_interp(signal, filtered_noisy_interp_signal, 
                                    list(zip(filtered_noisy_x_samples, filtered_noisy_y_samples)), 
                                    x_dense, sampling_rate, 
                                    (string_title_known_to_interp + 
                                    f", with Filtering White Gaussian Noise, mag={mag}, filter_window={filter_window}"))
        
        plot_subplots_compare_known_interp_error(known_signal=signal, 
                                    interp_signal=filtered_noisy_interp_signal,
                                    error_signal=error_analysis_dict_noisy_filtered["error_signal"], 
                                    deviation_signal=error_analysis_dict_noisy_filtered["deviation_signal"],
                                    x_dense=x_dense, sampling_rate=sampling_rate,
                                    string_title=(string_title_error + 
                                    f", with Filtering White Gaussian Noise, mag={mag}, filter_window={filter_window}"))

    print(f"\nTest 4.1 results for sampling rate {sampling_rate}" + 
          f"samples/sec:\nFiltering White Gaussian Noise, mag={mag}, filter_window={filter_window}\n")
    print(f"Average Error: {error_analysis_dict_noisy_filtered["avg_error"]}")
    print(f"Max Error: {error_analysis_dict_noisy_filtered["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict_noisy_filtered["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict_noisy_filtered["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict_noisy_filtered["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict_noisy_filtered["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict_noisy_filtered["max_deviation_time"]} seconds\n")

    temp_dict_noisy_filtered = {}
    temp_dict_noisy_filtered = error_analysis_dict_noisy_filtered.copy()
    temp_dict_noisy_filtered["sampling_rate"] = sampling_rate
    temp_dict_noisy_filtered["frequency"] = frequency
    temp_dict_noisy_filtered["x_samples"] = x_samples

    if filter_sweep_flag:
        stored_data_noisy_filtered[filter_window] = temp_dict_noisy_filtered
        return stored_data_noisy_filtered[filter_window]
    else:
        stored_data_noisy_filtered[mag] = temp_dict_noisy_filtered
        return stored_data_noisy_filtered[mag]



def run_test4_2(interp_type=None, filter_window=5, mag=None, h=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None, 
                                                    filter_sweep_flag=False):

    """
    run_test3_2(interp_type=None, filter_window=5, mag=None, h=None, M=None, plot_flag=True, 
                string_title_known_to_interp=None, string_title_error=None,
                                                    filter_sweep_flag=False)

    M is scaling of the signal frequency

    plotting will happen unless disabled

    returns dict stored_data_noisy_filtered
    """
    if any(v is None for v in [M, interp_type, mag, h, string_title_known_to_interp, string_title_error]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if not isinstance(mag, (float, int)):
        raise TypeError("\nmag must be float or int type.\n")
    if mag <= 0:
        raise ValueError("\nmag must be > 0.\n")
    if not isinstance(h, (float, int)):
        raise TypeError("\nh must be float or int type.\n")
    if h <= 0:
        raise ValueError("\nh must be > 0.\n")
    if filter_window is None:
        raise ValueError("\nfilter_window cannot be None.\n")
    if not isinstance(filter_window, int):
        raise TypeError("\nfilter_window must be int type.\n")
    if filter_window < 1:
        raise ValueError("\nfilter_window must be >= 1.\n")
    if filter_window % 2 == 0:
        raise ValueError("\nfilter_window must be odd for symmetric averaging.\n")
    plot_flag = bool(plot_flag)
    filter_sweep_flag = bool(filter_sweep_flag)

    stored_data_noisy_filtered = {}

    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)
    d_dt_signal = lambda t: 1/3 * (2*np.pi * frequency) * np.cos(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    dense = 2000
    x_dense = np.linspace(x0 + 2*h, x_end - 2*h, dense)

    noise_signal = mag * white_guassian_noise(samples)
    noisy_y_samples = y_samples + noise_signal
    noisy_x_samples = x_samples.copy()

    # implement filter
    filter_window = filter_window
    filtered_noisy_x_samples, filtered_noisy_y_samples = averaging_noise_filter(M=filter_window,
                                            paired=list(zip(noisy_x_samples, noisy_y_samples)))

    filtered_noisy_interp_signal = None
    if interp_type == "linear":
        filtered_noisy_interp_signal = linear_interp(list(zip(filtered_noisy_x_samples, filtered_noisy_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(filtered_noisy_x_samples, filtered_noisy_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        filtered_noisy_interp_signal = cubic.cubic_spline_interpolation(coeff)
    
    d_dt_filtered_noisy_interp_signal = centered_difference_4th_order(filtered_noisy_interp_signal, h)

    error_analysis_dict_noisy_filtered = {}
    error_analysis_dict_noisy_filtered = error_analysis(known_signal=d_dt_signal, 
                                                        interp_signal=d_dt_filtered_noisy_interp_signal, 
                                                                                        x_dense=x_dense)
    
    if plot_flag:
        # with noise of magnitude = mag
        plot_subplots_compare_known_interp(d_dt_signal, d_dt_filtered_noisy_interp_signal, 
                                        list(zip(filtered_noisy_x_samples, filtered_noisy_y_samples)), 
                                        x_dense, sampling_rate, 
                                        (string_title_known_to_interp + 
                                        f", with Filtering White Gaussian Noise, mag={mag}, filter_window={filter_window}"))
        
        plot_subplots_compare_known_interp_error(known_signal=d_dt_signal, 
                                    interp_signal=d_dt_filtered_noisy_interp_signal,
                                    error_signal=error_analysis_dict_noisy_filtered["error_signal"], 
                                    deviation_signal=error_analysis_dict_noisy_filtered["deviation_signal"],
                                    x_dense=x_dense, sampling_rate=sampling_rate,
                                    string_title=(string_title_error + 
                                    f", with Filtering White Gaussian Noise, mag={mag}, filter_window={filter_window}"))

    print(f"\nTest 4.2 results for sampling rate {sampling_rate}" + 
          f"samples/sec:\nFiltering White Gaussian Noise, mag={mag}, Order 4 Derivative, filter_window={filter_window}\n")
    print(f"Average Error: {error_analysis_dict_noisy_filtered["avg_error"]}")
    print(f"Max Error: {error_analysis_dict_noisy_filtered["max_error"]}")
    print(f"Max Error Index: {error_analysis_dict_noisy_filtered["max_error_index"]}")
    print(f"Max Error Index is at Time = {error_analysis_dict_noisy_filtered["max_error_time"]} seconds")
    print(f"Max Deviation: {error_analysis_dict_noisy_filtered["max_deviation"]}")
    print(f"Max Deviation Index: {error_analysis_dict_noisy_filtered["max_deviation_index"]}")
    print(f"Max Deviation Index is at Time = {error_analysis_dict_noisy_filtered["max_deviation_time"]} seconds\n")

    temp_dict_noisy_filtered = {}
    temp_dict_noisy_filtered = error_analysis_dict_noisy_filtered.copy()
    temp_dict_noisy_filtered["sampling_rate"] = sampling_rate
    temp_dict_noisy_filtered["frequency"] = frequency
    temp_dict_noisy_filtered["x_samples"] = x_samples

    if filter_sweep_flag:
        stored_data_noisy_filtered[filter_window] = temp_dict_noisy_filtered
        return stored_data_noisy_filtered[filter_window]
    else:
        stored_data_noisy_filtered[mag] = temp_dict_noisy_filtered
        return stored_data_noisy_filtered[mag]