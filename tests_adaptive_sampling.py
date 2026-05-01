# tests_adaptive_sampling.py

import numpy as np
from cubic_spline import Cubic_Spline
from plotting import plot_subplots_compare_known_interp, plot_subplots_compare_known_interp_error
from methods import linear_interp, error_analysis, adaptive_sampling_weighting


def run_test1(interp_type=None, alpha=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None):

    """
    run_test1(interp_type=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None)

    alpha is the weighting strength between uniform and fully adaptive sampling 

    supports magnitude or slope-informed or curvature-informed adaptive sampling strategy 

    M is scaling of the signal frequency

    plotting will happen unless disabled 

    returns stored_data dict
    """
    if any(v is None for v in [interp_type, alpha, M, plot_flag, string_title_known_to_interp, string_title_error, informed_type]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if M < 2:
        raise ValueError("\nM must be >= 2.\n")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("\nError, alpha must be between 0 and 1.\n")
    plot_flag = bool(plot_flag)
    
    stored_data = {}
    stored_data[alpha] = None
    
    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)
    d_dt_signal = lambda t: 1/3 * (2*np.pi * frequency) * np.cos(2*np.pi * frequency * t)
    d2_dt2_signal = lambda t: -(1/3) * (2*np.pi * frequency)**2 * np.sin(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1
    
    adaptive_x_samples = None

    if informed_type == "mag":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "slope":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d_dt_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "curvature":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d2_dt2_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    
    adaptive_y_samples = signal(adaptive_x_samples)

    interp_signal = None
    if interp_type == "linear":
        interp_signal = linear_interp(list(zip(adaptive_x_samples, adaptive_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(adaptive_x_samples, adaptive_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        interp_signal = cubic.cubic_spline_interpolation(coeff)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    if plot_flag:
        plot_subplots_compare_known_interp(signal, interp_signal, list(zip(adaptive_x_samples, adaptive_y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + f", alpha={alpha}"))

    error_analysis_dict = {}
    error_analysis_dict = error_analysis(known_signal=signal, interp_signal=interp_signal, 
                                                                            x_dense=x_dense) 

    if plot_flag:
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=interp_signal,
                                                error_signal=error_analysis_dict["error_signal"], 
                                                deviation_signal=error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error + f", alpha={alpha}"))
    
    print(f"\nAdaptive Test 1 {informed_type}-informed results for alpha={alpha}, "+
          f"sampling rate={sampling_rate} samples/sec:\n")
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
    temp_dict["x_samples"] = adaptive_x_samples
    stored_data[alpha] = temp_dict

    return stored_data[alpha]


def run_test1_1(interp_type=None, alpha=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None):

    """
    run_test1_1(interp_type=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None)

    alpha is the weighting strength between uniform and fully adaptive sampling 

    supports magnitude or slope-informed or curvature-informed adaptive sampling strategy 

    M is scaling of the signal frequency

    plotting will happen unless disabled 

    returns stored_data dict
    """
    if any(v is None for v in [interp_type, alpha, M, plot_flag, string_title_known_to_interp, string_title_error, informed_type]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if M < 2:
        raise ValueError("\nM must be >= 2.\n")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("\nError, alpha must be between 0 and 1.\n")
    plot_flag = bool(plot_flag)
    
    stored_data = {}
    stored_data[alpha] = None
    
    frequency = 10  # base frequency in Hz

    signal = lambda t: (0.45*np.sin(2*np.pi * frequency * t)
                        + 0.25*np.sin(2*np.pi * 3*frequency * t)
                        + 0.15*np.sin(2*np.pi * 7*frequency * t))

    d_dt_signal = lambda t: (0.45*(2*np.pi * frequency)*np.cos(2*np.pi * frequency * t)
                            + 0.25*(2*np.pi * 3*frequency)*np.cos(2*np.pi * 3*frequency * t)
                            + 0.15*(2*np.pi * 7*frequency)*np.cos(2*np.pi * 7*frequency * t))

    d2_dt2_signal = lambda t: (-0.45*(2*np.pi * frequency)**2*np.sin(2*np.pi * frequency * t)
                                -0.25*(2*np.pi * 3*frequency)**2*np.sin(2*np.pi * 3*frequency * t)
                                -0.15*(2*np.pi * 7*frequency)**2*np.sin(2*np.pi * 7*frequency * t))

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1
    
    adaptive_x_samples = None
    
    if informed_type == "mag":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "slope":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d_dt_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "curvature":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d2_dt2_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    
    adaptive_y_samples = signal(adaptive_x_samples)

    interp_signal = None
    if interp_type == "linear":
        interp_signal = linear_interp(list(zip(adaptive_x_samples, adaptive_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(adaptive_x_samples, adaptive_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        interp_signal = cubic.cubic_spline_interpolation(coeff)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    if plot_flag:
        plot_subplots_compare_known_interp(signal, interp_signal, list(zip(adaptive_x_samples, adaptive_y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + f", alpha={alpha}"))

    error_analysis_dict = {}
    error_analysis_dict = error_analysis(known_signal=signal, interp_signal=interp_signal, 
                                                                            x_dense=x_dense) 

    if plot_flag:
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=interp_signal,
                                                error_signal=error_analysis_dict["error_signal"], 
                                                deviation_signal=error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error + f", alpha={alpha}"))
    
    print(f"\nAdaptive Test 1.1 {informed_type}-informed results for alpha={alpha}, "+
          f"sampling rate={sampling_rate} samples/sec:\n")
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
    temp_dict["x_samples"] = adaptive_x_samples
    stored_data[alpha] = temp_dict

    return stored_data[alpha]


def run_test1_2(interp_type=None, alpha=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None):

    """
    run_test1_2(interp_type=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None)

    alpha is the weighting strength between uniform and fully adaptive sampling 

    supports magnitude or slope-informed or curvature-informed adaptive sampling strategy 

    M is scaling of the signal frequency

    plotting will happen unless disabled 

    returns stored_data dict
    """
    if any(v is None for v in [interp_type, alpha, M, plot_flag, string_title_known_to_interp, string_title_error, informed_type]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if M < 2:
        raise ValueError("\nM must be >= 2.\n")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("\nError, alpha must be between 0 and 1.\n")
    plot_flag = bool(plot_flag)
    
    stored_data = {}
    stored_data[alpha] = None

    frequency = 10  # base frequency in Hz

    signal = lambda t: (0.45*np.sin(2*np.pi * 7*frequency * t)*np.sin(np.pi * frequency * t)**2)

    d_dt_signal = lambda t: (0.45*((2*np.pi * 7*frequency)*np.cos(2*np.pi * 7*frequency * t)*np.sin(np.pi * frequency * t)**2
                            + np.sin(2*np.pi * 7*frequency * t)*(np.pi * frequency)*np.sin(2*np.pi * frequency * t)))

    d2_dt2_signal = lambda t: (0.45*(-(2*np.pi * 7*frequency)**2*np.sin(2*np.pi * 7*frequency * t)*np.sin(np.pi * frequency * t)**2
                                + 2*(2*np.pi * 7*frequency)*np.cos(2*np.pi * 7*frequency * t)*(np.pi * frequency)*np.sin(2*np.pi * frequency * t)
                                + np.sin(2*np.pi * 7*frequency * t)*(2*(np.pi * frequency)**2)*np.cos(2*np.pi * frequency * t)))

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1
    
    adaptive_x_samples = None
    
    if informed_type == "mag":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "slope":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d_dt_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "curvature":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d2_dt2_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    
    adaptive_y_samples = signal(adaptive_x_samples)

    interp_signal = None
    if interp_type == "linear":
        interp_signal = linear_interp(list(zip(adaptive_x_samples, adaptive_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(adaptive_x_samples, adaptive_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        interp_signal = cubic.cubic_spline_interpolation(coeff)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    if plot_flag:
        plot_subplots_compare_known_interp(signal, interp_signal, list(zip(adaptive_x_samples, adaptive_y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + f", alpha={alpha}"))

    error_analysis_dict = {}
    error_analysis_dict = error_analysis(known_signal=signal, interp_signal=interp_signal, 
                                                                            x_dense=x_dense) 

    if plot_flag:
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=interp_signal,
                                                error_signal=error_analysis_dict["error_signal"], 
                                                deviation_signal=error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error + f", alpha={alpha}"))
    
    print(f"\nAdaptive Test 1.2 {informed_type}-informed results for alpha={alpha}, "+
          f"sampling rate={sampling_rate} samples/sec:\n")
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
    temp_dict["x_samples"] = adaptive_x_samples
    stored_data[alpha] = temp_dict

    return stored_data[alpha]


def run_test2(interp_type=None, alpha=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None):

    """
    run_test2(interp_type=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None)

    alpha is the weighting strength between uniform and fully adaptive sampling 

    supports magnitude or slope-informed or curvature-informed adaptive sampling strategy 

    M is scaling of the signal frequency

    plotting will happen unless disabled 

    returns stored_data dict
    """
    if any(v is None for v in [interp_type, alpha, M, plot_flag, string_title_known_to_interp, string_title_error, informed_type]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if M < 2:
        raise ValueError("\nM must be >= 2.\n")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("\nError, alpha must be between 0 and 1.\n")
    plot_flag = bool(plot_flag)
    
    uniform_data = {} 
    adaptive_data = {}

    frequency = 10 # Hz
    signal = lambda t: 1/3 * np.sin(2*np.pi * frequency * t)
    d_dt_signal = lambda t: 1/3 * (2*np.pi * frequency) * np.cos(2*np.pi * frequency * t)
    d2_dt2_signal = lambda t: -(1/3) * (2*np.pi * frequency)**2 * np.sin(2*np.pi * frequency * t)

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    uniform_interp_signal = None
    if interp_type == "linear":
        uniform_interp_signal = linear_interp(list(zip(x_samples, y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(x_samples, y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        uniform_interp_signal = cubic.cubic_spline_interpolation(coeff)
    
    adaptive_x_samples = None
    
    if informed_type == "mag":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "slope":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d_dt_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "curvature":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d2_dt2_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    
    adaptive_y_samples = signal(adaptive_x_samples)

    adaptive_interp_signal = None
    if interp_type == "linear":
        adaptive_interp_signal = linear_interp(list(zip(adaptive_x_samples, adaptive_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(adaptive_x_samples, adaptive_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        adaptive_interp_signal = cubic.cubic_spline_interpolation(coeff)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    if plot_flag:
        plot_subplots_compare_known_interp(signal, adaptive_interp_signal, list(zip(adaptive_x_samples, adaptive_y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + 
                                                                    f", {informed_type}-informed adaptive sampling") +
                                                                    f", alpha={alpha}, samples={samples}")
        
        plot_subplots_compare_known_interp(signal, uniform_interp_signal, list(zip(x_samples, y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + 
                                                                    f", uniform sampling") +
                                                                    f", samples={samples}")

    uniform_error_analysis_dict = {}
    adaptive_error_analysis_dict = {}

    uniform_error_analysis_dict = error_analysis(known_signal=signal, interp_signal=uniform_interp_signal, 
                                                                                            x_dense=x_dense) 
    adaptive_error_analysis_dict = error_analysis(known_signal=signal, interp_signal=adaptive_interp_signal, 
                                                                                            x_dense=x_dense) 

    if plot_flag:
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=adaptive_interp_signal,
                                                error_signal=adaptive_error_analysis_dict["error_signal"], 
                                                deviation_signal=adaptive_error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error +
                                                f", {informed_type}-informed adaptive sampling") + 
                                                f", alpha={alpha}, samples={samples}")
        
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=uniform_interp_signal,
                                                error_signal=uniform_error_analysis_dict["error_signal"], 
                                                deviation_signal=uniform_error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error +
                                                f", uniform sampling") + 
                                                f", samples={samples}")
    

    print(f"\nTest 2 Uniform vs Adaptive results for M={M}, alpha={alpha}, "
        + f"{informed_type}-informed, sampling rate={sampling_rate} samples/sec:\n")

    print("Uniform Sampling:")
    print(f"Average Error: {uniform_error_analysis_dict['avg_error']}")
    print(f"Max Error: {uniform_error_analysis_dict['max_error']}")
    print(f"Max Error Index: {uniform_error_analysis_dict['max_error_index']}")
    print(f"Max Error Index is at Time = {uniform_error_analysis_dict['max_error_time']} seconds")
    print(f"Max Deviation: {uniform_error_analysis_dict['max_deviation']}")
    print(f"Max Deviation Index: {uniform_error_analysis_dict['max_deviation_index']}")
    print(f"Max Deviation Index is at Time = {uniform_error_analysis_dict['max_deviation_time']} seconds\n")

    print(f"Adaptive Sampling, {informed_type}-informed:")
    print(f"Average Error: {adaptive_error_analysis_dict['avg_error']}")
    print(f"Max Error: {adaptive_error_analysis_dict['max_error']}")
    print(f"Max Error Index: {adaptive_error_analysis_dict['max_error_index']}")
    print(f"Max Error Index is at Time = {adaptive_error_analysis_dict['max_error_time']} seconds")
    print(f"Max Deviation: {adaptive_error_analysis_dict['max_deviation']}")
    print(f"Max Deviation Index: {adaptive_error_analysis_dict['max_deviation_index']}")
    print(f"Max Deviation Index is at Time = {adaptive_error_analysis_dict['max_deviation_time']} seconds\n")

    temp_dict_1 = {}
    temp_dict_1 = uniform_error_analysis_dict.copy()
    temp_dict_1["sampling_rate"] = sampling_rate
    temp_dict_1["frequency"] = frequency
    temp_dict_1["x_samples"] = x_samples
    uniform_data[M] = temp_dict_1
    
    temp_dict_2 = {}
    temp_dict_2 = adaptive_error_analysis_dict.copy()
    temp_dict_2["sampling_rate"] = sampling_rate
    temp_dict_2["frequency"] = frequency
    temp_dict_2["x_samples"] = adaptive_x_samples
    adaptive_data[M] = temp_dict_2

    return uniform_data[M], adaptive_data[M]


def run_test2_1(interp_type=None, alpha=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None):

    """
    run_test2_1(interp_type=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None)

    alpha is the weighting strength between uniform and fully adaptive sampling 

    supports magnitude or slope-informed or curvature-informed adaptive sampling strategy 

    M is scaling of the signal frequency

    plotting will happen unless disabled 

    returns stored_data dict
    """
    if any(v is None for v in [interp_type, alpha, M, plot_flag, string_title_known_to_interp, string_title_error, informed_type]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if M < 2:
        raise ValueError("\nM must be >= 2.\n")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("\nError, alpha must be between 0 and 1.\n")
    plot_flag = bool(plot_flag)
    
    uniform_data = {} 
    adaptive_data = {}

    frequency = 10  # base frequency in Hz

    signal = lambda t: (0.45*np.sin(2*np.pi * frequency * t)
                        + 0.25*np.sin(2*np.pi * 3*frequency * t)
                        + 0.15*np.sin(2*np.pi * 7*frequency * t))

    d_dt_signal = lambda t: (0.45*(2*np.pi * frequency)*np.cos(2*np.pi * frequency * t)
                            + 0.25*(2*np.pi * 3*frequency)*np.cos(2*np.pi * 3*frequency * t)
                            + 0.15*(2*np.pi * 7*frequency)*np.cos(2*np.pi * 7*frequency * t))

    d2_dt2_signal = lambda t: (-0.45*(2*np.pi * frequency)**2*np.sin(2*np.pi * frequency * t)
                                -0.25*(2*np.pi * 3*frequency)**2*np.sin(2*np.pi * 3*frequency * t)
                                -0.15*(2*np.pi * 7*frequency)**2*np.sin(2*np.pi * 7*frequency * t))

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    uniform_interp_signal = None
    if interp_type == "linear":
        uniform_interp_signal = linear_interp(list(zip(x_samples, y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(x_samples, y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        uniform_interp_signal = cubic.cubic_spline_interpolation(coeff)
    
    adaptive_x_samples = None
    
    if informed_type == "mag":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "slope":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d_dt_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "curvature":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d2_dt2_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    
    adaptive_y_samples = signal(adaptive_x_samples)

    adaptive_interp_signal = None
    if interp_type == "linear":
        adaptive_interp_signal = linear_interp(list(zip(adaptive_x_samples, adaptive_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(adaptive_x_samples, adaptive_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        adaptive_interp_signal = cubic.cubic_spline_interpolation(coeff)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    if plot_flag:
        plot_subplots_compare_known_interp(signal, adaptive_interp_signal, list(zip(adaptive_x_samples, adaptive_y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + 
                                                                    f", {informed_type}-informed adaptive sampling") +
                                                                    f", alpha={alpha}, samples={samples}")
        
        plot_subplots_compare_known_interp(signal, uniform_interp_signal, list(zip(x_samples, y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + 
                                                                    f", uniform sampling") +
                                                                    f", samples={samples}")

    uniform_error_analysis_dict = {}
    adaptive_error_analysis_dict = {}

    uniform_error_analysis_dict = error_analysis(known_signal=signal, interp_signal=uniform_interp_signal, 
                                                                                            x_dense=x_dense) 
    adaptive_error_analysis_dict = error_analysis(known_signal=signal, interp_signal=adaptive_interp_signal, 
                                                                                            x_dense=x_dense) 

    if plot_flag:
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=adaptive_interp_signal,
                                                error_signal=adaptive_error_analysis_dict["error_signal"], 
                                                deviation_signal=adaptive_error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error +
                                                f", {informed_type}-informed adaptive sampling") + 
                                                f", alpha={alpha}, samples={samples}")
        
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=uniform_interp_signal,
                                                error_signal=uniform_error_analysis_dict["error_signal"], 
                                                deviation_signal=uniform_error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error +
                                                f", uniform sampling") + 
                                                f", samples={samples}")
    

    print(f"\nTest 2.1 Uniform vs Adaptive results for M={M}, alpha={alpha}, "
        + f"{informed_type}-informed, sampling rate={sampling_rate} samples/sec:\n")

    print("Uniform Sampling:")
    print(f"Average Error: {uniform_error_analysis_dict['avg_error']}")
    print(f"Max Error: {uniform_error_analysis_dict['max_error']}")
    print(f"Max Error Index: {uniform_error_analysis_dict['max_error_index']}")
    print(f"Max Error Index is at Time = {uniform_error_analysis_dict['max_error_time']} seconds")
    print(f"Max Deviation: {uniform_error_analysis_dict['max_deviation']}")
    print(f"Max Deviation Index: {uniform_error_analysis_dict['max_deviation_index']}")
    print(f"Max Deviation Index is at Time = {uniform_error_analysis_dict['max_deviation_time']} seconds\n")

    print(f"Adaptive Sampling, {informed_type}-informed:")
    print(f"Average Error: {adaptive_error_analysis_dict['avg_error']}")
    print(f"Max Error: {adaptive_error_analysis_dict['max_error']}")
    print(f"Max Error Index: {adaptive_error_analysis_dict['max_error_index']}")
    print(f"Max Error Index is at Time = {adaptive_error_analysis_dict['max_error_time']} seconds")
    print(f"Max Deviation: {adaptive_error_analysis_dict['max_deviation']}")
    print(f"Max Deviation Index: {adaptive_error_analysis_dict['max_deviation_index']}")
    print(f"Max Deviation Index is at Time = {adaptive_error_analysis_dict['max_deviation_time']} seconds\n")

    temp_dict_1 = {}
    temp_dict_1 = uniform_error_analysis_dict.copy()
    temp_dict_1["sampling_rate"] = sampling_rate
    temp_dict_1["frequency"] = frequency
    temp_dict_1["x_samples"] = x_samples
    uniform_data[M] = temp_dict_1
    
    temp_dict_2 = {}
    temp_dict_2 = adaptive_error_analysis_dict.copy()
    temp_dict_2["sampling_rate"] = sampling_rate
    temp_dict_2["frequency"] = frequency
    temp_dict_2["x_samples"] = adaptive_x_samples
    adaptive_data[M] = temp_dict_2

    return uniform_data[M], adaptive_data[M]


def run_test2_2(interp_type=None, alpha=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None):

    """
    run_test2_1(interp_type=None, M=None, plot_flag=True, 
              string_title_known_to_interp=None, string_title_error=None, informed_type=None)

    alpha is the weighting strength between uniform and fully adaptive sampling 

    supports magnitude or slope-informed or curvature-informed adaptive sampling strategy 

    M is scaling of the signal frequency

    plotting will happen unless disabled 

    returns stored_data dict
    """
    if any(v is None for v in [interp_type, alpha, M, plot_flag, string_title_known_to_interp, string_title_error, informed_type]):
        raise ValueError("\nall parameters must be provided.\n")
    if not isinstance(M, (float, int)):
        raise TypeError("\nM must be float or int type.\n")
    if M < 2:
        raise ValueError("\nM must be >= 2.\n")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("\nError, alpha must be between 0 and 1.\n")
    plot_flag = bool(plot_flag)
    
    uniform_data = {} 
    adaptive_data = {}

    frequency = 10  # base frequency in Hz

    signal = lambda t: (0.45*np.sin(2*np.pi * 7*frequency * t)*np.sin(np.pi * frequency * t)**2)

    d_dt_signal = lambda t: (0.45*((2*np.pi * 7*frequency)*np.cos(2*np.pi * 7*frequency * t)*np.sin(np.pi * frequency * t)**2
                            + np.sin(2*np.pi * 7*frequency * t)*(np.pi * frequency)*np.sin(2*np.pi * frequency * t)))

    d2_dt2_signal = lambda t: (0.45*(-(2*np.pi * 7*frequency)**2*np.sin(2*np.pi * 7*frequency * t)*np.sin(np.pi * frequency * t)**2
                                + 2*(2*np.pi * 7*frequency)*np.cos(2*np.pi * 7*frequency * t)*(np.pi * frequency)*np.sin(2*np.pi * frequency * t)
                                + np.sin(2*np.pi * 7*frequency * t)*(2*(np.pi * frequency)**2)*np.cos(2*np.pi * frequency * t)))

    sampling_rate = M * frequency # min sampling rate is 2xfreq
    period = 1/frequency
    number_periods = 2
    x0 = 0
    x_end = number_periods * period 
    samples = int((x_end - x0) * sampling_rate) + 1

    # evenly spaced sampling
    x_samples = np.linspace(x0, x_end, samples)
    y_samples = signal(x_samples)

    uniform_interp_signal = None
    if interp_type == "linear":
        uniform_interp_signal = linear_interp(list(zip(x_samples, y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(x_samples, y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        uniform_interp_signal = cubic.cubic_spline_interpolation(coeff)
    
    adaptive_x_samples = None
    
    if informed_type == "mag":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "slope":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d_dt_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    if informed_type == "curvature":
        adaptive_x_samples = adaptive_sampling_weighting(weight_func=d2_dt2_signal, x_start=x0, x_end=x_end,
                                    samples=samples, alpha=alpha, dense=2000, tol=1e-12, threshold=1e-12)
    
    adaptive_y_samples = signal(adaptive_x_samples)

    adaptive_interp_signal = None
    if interp_type == "linear":
        adaptive_interp_signal = linear_interp(list(zip(adaptive_x_samples, adaptive_y_samples)))
    if interp_type == "cubic":
        cubic = Cubic_Spline(list(zip(adaptive_x_samples, adaptive_y_samples)))
        coeff = cubic.cubic_spline_natural_coeff()
        adaptive_interp_signal = cubic.cubic_spline_interpolation(coeff)

    dense = 2000
    x_dense = np.linspace(x0, x_end, dense)

    if plot_flag:
        plot_subplots_compare_known_interp(signal, adaptive_interp_signal, list(zip(adaptive_x_samples, adaptive_y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + 
                                                                    f", {informed_type}-informed adaptive sampling") +
                                                                    f", alpha={alpha}, samples={samples}")
        
        plot_subplots_compare_known_interp(signal, uniform_interp_signal, list(zip(x_samples, y_samples)), 
                                            x_dense, sampling_rate, (string_title_known_to_interp + 
                                                                    f", uniform sampling") +
                                                                    f", samples={samples}")

    uniform_error_analysis_dict = {}
    adaptive_error_analysis_dict = {}

    uniform_error_analysis_dict = error_analysis(known_signal=signal, interp_signal=uniform_interp_signal, 
                                                                                            x_dense=x_dense) 
    adaptive_error_analysis_dict = error_analysis(known_signal=signal, interp_signal=adaptive_interp_signal, 
                                                                                            x_dense=x_dense) 

    if plot_flag:
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=adaptive_interp_signal,
                                                error_signal=adaptive_error_analysis_dict["error_signal"], 
                                                deviation_signal=adaptive_error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error +
                                                f", {informed_type}-informed adaptive sampling") + 
                                                f", alpha={alpha}, samples={samples}")
        
        plot_subplots_compare_known_interp_error(known_signal=signal, interp_signal=uniform_interp_signal,
                                                error_signal=uniform_error_analysis_dict["error_signal"], 
                                                deviation_signal=uniform_error_analysis_dict["deviation_signal"],
                                                x_dense=x_dense, sampling_rate=sampling_rate,
                                                string_title=(string_title_error +
                                                f", uniform sampling") + 
                                                f", samples={samples}")
    

    print(f"\nTest 2.2 Uniform vs Adaptive results for M={M}, alpha={alpha}, "
        + f"{informed_type}-informed, sampling rate={sampling_rate} samples/sec:\n")

    print("Uniform Sampling:")
    print(f"Average Error: {uniform_error_analysis_dict['avg_error']}")
    print(f"Max Error: {uniform_error_analysis_dict['max_error']}")
    print(f"Max Error Index: {uniform_error_analysis_dict['max_error_index']}")
    print(f"Max Error Index is at Time = {uniform_error_analysis_dict['max_error_time']} seconds")
    print(f"Max Deviation: {uniform_error_analysis_dict['max_deviation']}")
    print(f"Max Deviation Index: {uniform_error_analysis_dict['max_deviation_index']}")
    print(f"Max Deviation Index is at Time = {uniform_error_analysis_dict['max_deviation_time']} seconds\n")

    print(f"Adaptive Sampling, {informed_type}-informed:")
    print(f"Average Error: {adaptive_error_analysis_dict['avg_error']}")
    print(f"Max Error: {adaptive_error_analysis_dict['max_error']}")
    print(f"Max Error Index: {adaptive_error_analysis_dict['max_error_index']}")
    print(f"Max Error Index is at Time = {adaptive_error_analysis_dict['max_error_time']} seconds")
    print(f"Max Deviation: {adaptive_error_analysis_dict['max_deviation']}")
    print(f"Max Deviation Index: {adaptive_error_analysis_dict['max_deviation_index']}")
    print(f"Max Deviation Index is at Time = {adaptive_error_analysis_dict['max_deviation_time']} seconds\n")

    temp_dict_1 = {}
    temp_dict_1 = uniform_error_analysis_dict.copy()
    temp_dict_1["sampling_rate"] = sampling_rate
    temp_dict_1["frequency"] = frequency
    temp_dict_1["x_samples"] = x_samples
    uniform_data[M] = temp_dict_1
    
    temp_dict_2 = {}
    temp_dict_2 = adaptive_error_analysis_dict.copy()
    temp_dict_2["sampling_rate"] = sampling_rate
    temp_dict_2["frequency"] = frequency
    temp_dict_2["x_samples"] = adaptive_x_samples
    adaptive_data[M] = temp_dict_2

    return uniform_data[M], adaptive_data[M]

