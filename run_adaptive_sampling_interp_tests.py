# run_adaptive_sampling_interp_tests.py

from tests_adaptive_sampling import (run_test1, run_test1_1, run_test1_2, run_test2, run_test2_1, run_test2_2)
from plotting import plot_error_per_rate
from methods import find_min_for_tolerance
import numpy as np


def run_adaptive_sampling_interp_tests(test=None, plot_flag=True, interp_type=None, informed_type=None):

    """
    run_adaptive_sampling_interp_tests(test=None, plot_flag=True, interp_type=None, informed_type=None):

        test is test number
        plot flag is flag for iterative plotting
        interp type is string for specifying linear vs cubic interp
        informed_type is string for specifying adaptive sampling strategy to be
            magnitude-informed (f) or slope-informed (f') or curvature-informed (f'')

    the function runs adaptive sampling interpolation tests

    tests:
        interpolation accuracy vs adaptive sampling weighting, supports magnitude or 1st or 2nd derivative informed, at fixed sampling frequency 

            basic sine wave function
            multi-frequency sine wave function
            burst/modulated sine wave function

        interpolation accuracy, comparing uniform vs adaptive sampling, sweeping number samples used to interpolate 

            basic sine wave function
            multi-frequency sine wave function
            burst/modulated sine wave function

    for each test, compare against the known signal and interpolated signal:
        - avg error
        - max error and index
        - max deviation and index
    """
    if test is None or interp_type is None or informed_type is None:
        raise ValueError("\ntest, interp_type, informed_type cannot be None.\n")
    if not isinstance(test, (float, int)):
        raise TypeError("\ntest number must be float or int type.\n")
    if not isinstance(interp_type, (str)):
        raise TypeError("\ninterp_type must be str type.\n")
    if interp_type.lower() not in ["linear", "cubic"]:
        raise ValueError("\nYou must enter valid interpolation type (linear or cubic)\n")
    if not isinstance(informed_type, (str)):
        raise TypeError("\ninformed_type must be str type.\n")
    if informed_type.lower() not in ["slope", "curvature", "mag"]:
        raise ValueError("\nYou must enter valid information type (slope, curvature, or mag)\n")
    interp_type = interp_type.lower()
    informed_type = informed_type.lower()
    plot_flag = bool(plot_flag)


# TEST 1: interpolation accuracy vs adaptive sampling weighting, supports magnitude or 1st or 2nd derivative informed, at fixed sampling frequency 

# basic sine function

    stored_data = {}
    fixed_M = 75
    avg_err_tol = 1e-3
    max_dev_tol = 1e-3

    string_title_known_to_interp = "Known Signal to Interpolated Signal"
    string_title_error = "Known Signal to Interpolated Signal and Error"

    # use if you want plots
    alpha_test_list = [0, 0.15, 0.25, 0.5, 0.75, 0.9, 1]
    # use dense list for finding min. variable needed for error < tol
    alpha_dense_test_list = np.linspace(0, 1, 15)

    if test == 1:
        if plot_flag:
            for a in alpha_test_list:
                stored_data[a] = run_test1(interp_type=interp_type, alpha=a, M=fixed_M, plot_flag=plot_flag, 
                                        string_title_known_to_interp=(string_title_known_to_interp +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        string_title_error=(string_title_error +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        informed_type=informed_type)
        else:
            for a in alpha_dense_test_list:
                stored_data[a] = run_test1(interp_type=interp_type, alpha=a, M=fixed_M, plot_flag=plot_flag, 
                                        string_title_known_to_interp=(string_title_known_to_interp +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        string_title_error=(string_title_error +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        informed_type=informed_type)

            find_min_for_tolerance(stored_data=stored_data, test_list=list(reversed(alpha_dense_test_list)),
                                            avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol, string_type="alpha")
            
        if plot_flag:
            plot_error_per_rate(stored_data, alpha_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"), 
                                              string_type="alpha")
        else:
            plot_error_per_rate(stored_data, alpha_dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"),
                                              string_type="alpha")
            

# TEST 1.1: interpolation accuracy vs adaptive sampling weighting, supports magnitude or 1st or 2nd derivative informed, at fixed sampling frequency 

# multi-frequncy sine function

    if test == 1.1:
        if plot_flag:
            for a in alpha_test_list:
                stored_data[a] = run_test1_1(interp_type=interp_type, alpha=a, M=fixed_M, plot_flag=plot_flag, 
                                        string_title_known_to_interp=(string_title_known_to_interp +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        string_title_error=(string_title_error +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        informed_type=informed_type)
        else:
            for a in alpha_dense_test_list:
                stored_data[a] = run_test1_1(interp_type=interp_type, alpha=a, M=fixed_M, plot_flag=plot_flag, 
                                        string_title_known_to_interp=(string_title_known_to_interp +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        string_title_error=(string_title_error +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        informed_type=informed_type)

            find_min_for_tolerance(stored_data=stored_data, test_list=list(reversed(alpha_dense_test_list)),
                                            avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol, string_type="alpha")
            
        if plot_flag:
            plot_error_per_rate(stored_data, alpha_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"), 
                                              string_type="alpha")
        else:
            plot_error_per_rate(stored_data, alpha_dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"),
                                              string_type="alpha")
            

# TEST 1.2: interpolation accuracy vs adaptive sampling weighting, supports magnitude or 1st or 2nd derivative informed, at fixed sampling frequency 

# burst/modulated sine function

    if test == 1.2:
        if plot_flag:
            for a in alpha_test_list:
                stored_data[a] = run_test1_2(interp_type=interp_type, alpha=a, M=fixed_M, plot_flag=plot_flag, 
                                        string_title_known_to_interp=(string_title_known_to_interp +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        string_title_error=(string_title_error +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        informed_type=informed_type)
        else:
            for a in alpha_dense_test_list:
                stored_data[a] = run_test1_2(interp_type=interp_type, alpha=a, M=fixed_M, plot_flag=plot_flag, 
                                        string_title_known_to_interp=(string_title_known_to_interp +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        string_title_error=(string_title_error +
                                                            f", {informed_type}-informed adaptive sampling"),
                                        informed_type=informed_type)

            find_min_for_tolerance(stored_data=stored_data, test_list=list(reversed(alpha_dense_test_list)),
                                            avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol, string_type="alpha")
            
        if plot_flag:
            plot_error_per_rate(stored_data, alpha_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"), 
                                              string_type="alpha")
        else:
            plot_error_per_rate(stored_data, alpha_dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"),
                                              string_type="alpha")
            

# TEST 2: comparing uniform vs adaptive sampling, sweeping number samples used to interpolate  

# basic sine wave function

    # use if you want plots
    test_list = [3, 4, 8, 16, 32, 64]
    # use dense list for finding min. data samples needed for error < tol
    dense_test_list = np.linspace(10, 280, 13)
    fixed_alpha = 0.35
    stored_data_uniform = {}
    stored_data_adaptive = {}

    if test == 2:
        if plot_flag:
            for M in test_list:
                stored_data_uniform[M], stored_data_adaptive[M] = run_test2(interp_type=interp_type, alpha=fixed_alpha, 
                                                                            M=M, plot_flag=plot_flag, 
                                                                            string_title_known_to_interp=(string_title_known_to_interp),
                                                                            string_title_error=(string_title_error),
                                                                            informed_type=informed_type)
        else:
            for M in dense_test_list:
                stored_data_uniform[M], stored_data_adaptive[M] = run_test2(interp_type=interp_type, alpha=fixed_alpha, 
                                                                            M=M, plot_flag=plot_flag, 
                                                                            string_title_known_to_interp=(string_title_known_to_interp),
                                                                            string_title_error=(string_title_error),
                                                                            informed_type=informed_type)
            
            print("\nUniform Sampling, Find Min. Samples for Tolerance\n")
            
            find_min_for_tolerance(stored_data=stored_data_uniform, test_list=dense_test_list,
                                    avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol)
            
            print(f"\nAdaptive Sampling, {informed_type}-informed, Find Min. Samples for Tolerance\n")
    
            find_min_for_tolerance(stored_data=stored_data_adaptive, test_list=dense_test_list,
                                    avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol)
            
        if plot_flag:

            plot_error_per_rate(stored_data_uniform, test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", uniform sampling"))

            plot_error_per_rate(stored_data_adaptive, test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"))
            
        else:
            plot_error_per_rate(stored_data_uniform, dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", uniform sampling"))

            plot_error_per_rate(stored_data_adaptive, dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"))
            

# TEST 2.1: comparing uniform vs adaptive sampling, sweeping number samples used to interpolate  

# multi-frequency sine wave function

    test_list = [14, 16, 18, 20, 25, 30, 40, 64]
    dense_test_list = np.linspace(14, 280, 15)

    if test == 2.1:
        if plot_flag:
            for M in test_list:
                stored_data_uniform[M], stored_data_adaptive[M] = run_test2_1(interp_type=interp_type, alpha=fixed_alpha, 
                                                                            M=M, plot_flag=plot_flag, 
                                                                            string_title_known_to_interp=(string_title_known_to_interp),
                                                                            string_title_error=(string_title_error),
                                                                            informed_type=informed_type)
        else:
            for M in dense_test_list:
                stored_data_uniform[M], stored_data_adaptive[M] = run_test2_1(interp_type=interp_type, alpha=fixed_alpha, 
                                                                            M=M, plot_flag=plot_flag, 
                                                                            string_title_known_to_interp=(string_title_known_to_interp),
                                                                            string_title_error=(string_title_error),
                                                                            informed_type=informed_type)
            
            print("\nUniform Sampling, Find Min. Samples for Tolerance\n")
            
            find_min_for_tolerance(stored_data=stored_data_uniform, test_list=dense_test_list,
                                    avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol)
            
            print(f"\nAdaptive Sampling, {informed_type}-informed, Find Min. Samples for Tolerance\n")
    
            find_min_for_tolerance(stored_data=stored_data_adaptive, test_list=dense_test_list,
                                    avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol)
            
        if plot_flag:

            plot_error_per_rate(stored_data_uniform, test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", uniform sampling"))

            plot_error_per_rate(stored_data_adaptive, test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"))
            
        else:
            plot_error_per_rate(stored_data_uniform, dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", uniform sampling"))

            plot_error_per_rate(stored_data_adaptive, dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"))


# TEST 2.2: comparing uniform vs adaptive sampling, sweeping number samples used to interpolate  

# burst/modulated sine wave function

    if test == 2.2:
        if plot_flag:
            for M in test_list:
                stored_data_uniform[M], stored_data_adaptive[M] = run_test2_2(interp_type=interp_type, alpha=fixed_alpha, 
                                                                            M=M, plot_flag=plot_flag, 
                                                                            string_title_known_to_interp=(string_title_known_to_interp),
                                                                            string_title_error=(string_title_error),
                                                                            informed_type=informed_type)
        else:
            for M in dense_test_list:
                stored_data_uniform[M], stored_data_adaptive[M] = run_test2_2(interp_type=interp_type, alpha=fixed_alpha, 
                                                                            M=M, plot_flag=plot_flag, 
                                                                            string_title_known_to_interp=(string_title_known_to_interp),
                                                                            string_title_error=(string_title_error),
                                                                            informed_type=informed_type)
            
            print("\nUniform Sampling, Find Min. Samples for Tolerance\n")
            
            find_min_for_tolerance(stored_data=stored_data_uniform, test_list=dense_test_list,
                                    avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol)
            
            print(f"\nAdaptive Sampling, {informed_type}-informed, Find Min. Samples for Tolerance\n")
    
            find_min_for_tolerance(stored_data=stored_data_adaptive, test_list=dense_test_list,
                                    avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol)
            
        if plot_flag:

            plot_error_per_rate(stored_data_uniform, test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", uniform sampling"))

            plot_error_per_rate(stored_data_adaptive, test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"))
            
        else:
            plot_error_per_rate(stored_data_uniform, dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", uniform sampling"))

            plot_error_per_rate(stored_data_adaptive, dense_test_list, 
                                string_title=(string_title_known_to_interp + 
                                              f", {informed_type}-informed adaptive sampling"))