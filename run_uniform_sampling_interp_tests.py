# run_uniform_sampling_interp_tests.py

from tests_uniform_sampling import (run_test1, run_test2_1, run_test2_2, 
                                    run_test3_1, run_test3_2, run_test4_1, run_test4_2)
from plotting import plot_error_per_rate
from methods import find_min_for_tolerance
import numpy as np


def run_uniform_sampling_interp_tests(test=None, plot_flag=True, interp_type=None):

    """
    run_uniform_sampling_interp_tests(test=None, plot_flag=True, interp_type=None):

        test is test number
        plot flag is flag for iterative plotting
        interp type is string for specifying linear vs cubic interp

    the function runs uniform sampling interpolation tests

    tests:
        interpolation accuracy vs sampling rate

        derivative testing will compare 2nd and 4th order methods:

            interpolation derivative accuracy vs sampling rate at fixed h step
            interpolation derivative accuracy vs h step for fixed sampling rate

        interpolation accuracy vs noise amplitude at fixed sampling rate
        interpolation derivative accuracy vs noise amplitude at fixed sampling rate and h step

        filtered interpolation accuracy vs noise amplitude at fixed sampling rate
        filtered interpolation derivative accuracy vs noise amplitude at fixed sampling rate and h step

        filtered interpolation accuracy vs window size for moving avg filter at fixed 
                                                        noise amplitude and fixed sampling rate
        filtered interpolation derivative accuracy vs window size for moving avg filter at fixed
                                                                noise amplitude, sampling rate, and h step

    for each test, compare against the known signal and interpolated signal:
        - avg error
        - max error and index
        - max deviation and index
    """
    if test is None or interp_type is None:
        raise ValueError("\ntest, interp_type cannot be None.\n")
    if not isinstance(test, (float, int)):
        raise TypeError("\ntest number must be float or int type.\n")
    if not isinstance(interp_type, (str)):
        raise TypeError("\ninterp_type must be str type.\n")
    if interp_type.lower() not in ["linear", "cubic"]:
        raise ValueError("\nYou must enter valid interpolation type (linear or cubic)\n")
    interp_type = interp_type.lower()

    # 1ST TEST: interpolation accuracy vs sampling rate

    stored_data = {}
    plot_flag = bool(plot_flag)
    avg_err_tol = 1e-3
    max_dev_tol = 1e-3

    string_title_known_to_interp = "Known Signal to Interpolated Signal"
    string_title_error = "Known Signal to Interpolated Signal and Error"

    # use if you want plots
    test_list = [3, 4, 8, 16, 32, 64]
    # use dense list for finding min. data samples needed for error < tol
    dense_test_list = np.linspace(10, 280, 13)

    if test == 1:
        if plot_flag:
            for M in test_list:
                stored_data[M] = run_test1(interp_type=interp_type, M=M, plot_flag=plot_flag, 
                                           string_title_known_to_interp=string_title_known_to_interp,
                                                                string_title_error=string_title_error)
        else:
            for M in dense_test_list:
                stored_data[M] = run_test1(interp_type=interp_type, M=M, plot_flag=plot_flag, 
                                           string_title_known_to_interp=string_title_known_to_interp,
                                                                string_title_error=string_title_error)

            find_min_for_tolerance(stored_data=stored_data, test_list=dense_test_list,
                                            avg_error_tol=avg_err_tol, max_dev_tol=max_dev_tol)
            
        if plot_flag:
            plot_error_per_rate(stored_data, test_list, string_title=string_title_known_to_interp)
        else:
            plot_error_per_rate(stored_data, dense_test_list, string_title=string_title_known_to_interp)


    # 2ND TEST: derivative testing will compare 2nd and 4th order methods

    # 2.1 TEST: interpolation derivative accuracy vs sampling rate at fixed h step 

    stored_data_order2 = {}
    stored_data_order4 = {}
    dt_avg_err_tol = 1e-2
    dt_max_dev_tol = 1e-2
    string_title_known_to_interp_derivative = "Known Signal Derivative to Interpolated Signal Derivative"
    string_title_error_derivative = "Known Signal Derivative to Interpolated Signal Derivative and Error"

    if test == 2.1:
        if plot_flag:
            for M in test_list:
                stored_data_order2[M], stored_data_order4[M] = run_test2_1(interp_type=interp_type, 
                                                                            M=M, plot_flag=plot_flag,
                                          string_title_known_to_interp=string_title_known_to_interp_derivative,
                                                              string_title_error=string_title_error_derivative)

        else:
            for M in dense_test_list:
                stored_data_order2[M], stored_data_order4[M] = run_test2_1(interp_type=interp_type, 
                                                                           M=M, plot_flag=plot_flag,
                                          string_title_known_to_interp=string_title_known_to_interp_derivative,
                                                              string_title_error=string_title_error_derivative)

            print("\nOrder 2 Derivative, Find Min. Samples for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_order2, test_list=dense_test_list,
                                            avg_error_tol=dt_avg_err_tol, max_dev_tol=dt_max_dev_tol)

            print("\nOrder 4 Derivative, Find Min. Samples for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_order4, test_list=dense_test_list,
                                            avg_error_tol=dt_avg_err_tol, max_dev_tol=dt_max_dev_tol)
        if plot_flag:    
            plot_error_per_rate(stored_data_order2, test_list, 
                                        string_title=(string_title_known_to_interp_derivative+" Order 2"))
            plot_error_per_rate(stored_data_order4, test_list, 
                                        string_title=(string_title_known_to_interp_derivative+" Order 4"))
        else:
            plot_error_per_rate(stored_data_order2, dense_test_list, 
                                        string_title=(string_title_known_to_interp_derivative+" Order 2"))
            plot_error_per_rate(stored_data_order4, dense_test_list, 
                                        string_title=(string_title_known_to_interp_derivative+" Order 4"))


    # 2.2 TEST: interpolation derivative accuracy vs h step for fixed sampling rate

    h_test_list = 1/5* np.array([0.01, 0.005, 0.0025, 0.001, 0.0005])
    h_max = 1/5 * (0.01)
    h_min = h_max * 1e-2
    dense_h_test_list = np.linspace(h_max, h_min, 20)
    fixed_M = 77.5 * 5 

    if test == 2.2:
        if plot_flag:
            for h in h_test_list:
                stored_data_order2[h], stored_data_order4[h] = run_test2_2(interp_type=interp_type, h=h, 
                                                                           M=fixed_M, plot_flag=plot_flag,
                                            string_title_known_to_interp=string_title_known_to_interp_derivative,
                                                                string_title_error=string_title_error_derivative)

        else:
            for h in dense_h_test_list:
                stored_data_order2[h], stored_data_order4[h] = run_test2_2(interp_type=interp_type, h=h, 
                                                                           M=fixed_M, plot_flag=plot_flag,
                                          string_title_known_to_interp=string_title_known_to_interp_derivative,
                                                              string_title_error=string_title_error_derivative)

            print("\nOrder 2 Derivative, Find Min. h Step for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_order2, test_list=dense_h_test_list,
                            avg_error_tol=dt_avg_err_tol, max_dev_tol=dt_max_dev_tol, string_type='h Step')

            print("\nOrder 4 Derivative, Find Min. h Step for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_order4, test_list=dense_h_test_list,
                            avg_error_tol=dt_avg_err_tol, max_dev_tol=dt_max_dev_tol, string_type='h Step')

        if plot_flag:    
            plot_error_per_rate(stored_data_order2, h_test_list, 
                                string_title=(string_title_known_to_interp_derivative+" Order 2"), string_type="h Step")
            plot_error_per_rate(stored_data_order4, h_test_list, 
                                string_title=(string_title_known_to_interp_derivative+" Order 4"), string_type="h Step")
        else:
            plot_error_per_rate(stored_data_order2, dense_h_test_list, 
                                string_title=(string_title_known_to_interp_derivative+" Order 2"), string_type="h Step")
            plot_error_per_rate(stored_data_order4, dense_h_test_list, 
                                string_title=(string_title_known_to_interp_derivative+" Order 4"), string_type="h Step")
            

    # TEST 3: WGN WHITE GAUSSIAN NOISE TESTS 
           
    # TEST 3.1: interpolation accuracy vs noise amplitude at fixed sampling rate

    noise_mag = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 2.5, 5]
    dense_noise_mag = np.linspace(0.0005, 10, 25)
    stored_data_noisy = {}
    avg_err_tol_noisy = 0.1
    max_dev_tol_noisy = 0.1

    if test == 3.1:
        if plot_flag:
            for m in reversed(noise_mag):
                stored_data_noisy[m] = run_test3_1(interp_type=interp_type, mag=m, M=fixed_M, 
                                                                        plot_flag=plot_flag, 
                                string_title_known_to_interp=string_title_known_to_interp,
                                                    string_title_error=string_title_error)
        else:
            for m in reversed(dense_noise_mag):
                stored_data_noisy[m] = run_test3_1(interp_type=interp_type, mag=m, M=fixed_M, 
                                                                        plot_flag=plot_flag, 
                                string_title_known_to_interp=string_title_known_to_interp,
                                                    string_title_error=string_title_error)
                
            print("\nWhite Gaussian Noise Case, Find Min. mag for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_noisy, test_list=list(reversed(dense_noise_mag)),
                                            avg_error_tol=avg_err_tol_noisy, max_dev_tol=max_dev_tol_noisy, 
                                                                                    string_type="noise mag")
        if plot_flag:    
            plot_error_per_rate(stored_data_noisy, noise_mag, 
                                string_title=(string_title_known_to_interp + ", with White Gaussian Noise"), 
                                                                                    string_type="noise mag")
        else:
            plot_error_per_rate(stored_data_noisy, dense_noise_mag, 
                                string_title=(string_title_known_to_interp + ", with White Gaussian Noise"), 
                                                                                    string_type="noise mag")


    # TEST 3.2: interpolation derivative accuracy vs noise amplitude at fixed sampling rate and h step
    
    fixed_h = 0.002

    if test == 3.2:
        if plot_flag:
            for m in reversed(noise_mag):
                stored_data_noisy[m] = run_test3_2(interp_type=interp_type, mag=m, h=fixed_h, M=fixed_M, 
                                                                                    plot_flag=plot_flag, 
                        string_title_known_to_interp=(string_title_known_to_interp_derivative + ", Order 4"),
                                            string_title_error=(string_title_error_derivative+ ", Order 4"))
        else:
            for m in reversed(dense_noise_mag):
                stored_data_noisy[m] = run_test3_2(interp_type=interp_type, mag=m, h=fixed_h, M=fixed_M, 
                                                                                    plot_flag=plot_flag, 
                        string_title_known_to_interp=(string_title_known_to_interp_derivative + ", Order 4"),
                                            string_title_error=(string_title_error_derivative+ ", Order 4"))
                
            print("\nOrder 4 Derivative Tests With White Gaussian Noise, Find Min. mag for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_noisy, test_list=list(reversed(dense_noise_mag)),
                                            avg_error_tol=avg_err_tol_noisy, max_dev_tol=max_dev_tol_noisy, 
                                                                                    string_type="noise mag")
        if plot_flag:    
            plot_error_per_rate(stored_data_noisy, noise_mag, 
                                string_title=(string_title_known_to_interp_derivative + 
                                              ", with White Gaussian Noise, Order 4"), 
                                                                string_type="noise mag")
        else:
            plot_error_per_rate(stored_data_noisy, dense_noise_mag, 
                                string_title=(string_title_known_to_interp_derivative + 
                                              ", with White Gaussian Noise, Order 4"), 
                                                                string_type="noise mag")
            

    # TEST 4: FILTERING WGN WHITE GAUSSIAN NOISE TESTS 
           
    # TEST 4.1: filtered interpolation accuracy vs noise amplitude at fixed sampling rate
    stored_data_noisy_filtered = {}

    if test == 4.1:
        if plot_flag:
            for m in reversed(noise_mag):
                stored_data_noisy_filtered[m] = run_test4_1(interp_type=interp_type, mag=m, 
                                                            M=fixed_M, plot_flag=plot_flag, 
                                string_title_known_to_interp=string_title_known_to_interp,
                                                    string_title_error=string_title_error)
        else:
            for m in reversed(dense_noise_mag):
                stored_data_noisy_filtered[m] = run_test4_1(interp_type=interp_type, mag=m, 
                                                            M=fixed_M, plot_flag=plot_flag, 
                                string_title_known_to_interp=string_title_known_to_interp,
                                                    string_title_error=string_title_error)
                
            print("\nFiltering White Gaussian Noise Case, Find Min. mag for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_noisy_filtered, test_list=list(reversed(dense_noise_mag)),
                                            avg_error_tol=avg_err_tol_noisy, max_dev_tol=max_dev_tol_noisy, 
                                                                            string_type="filtered noise mag")
        if plot_flag:    
            plot_error_per_rate(stored_data_noisy_filtered, noise_mag, 
                                string_title=(string_title_known_to_interp + 
                                              ", with Filtering White Gaussian Noise"), 
                                                        string_type="filtered noise mag")
        else:
            plot_error_per_rate(stored_data_noisy_filtered, dense_noise_mag, 
                                string_title=(string_title_known_to_interp + 
                                              ", with Filtering White Gaussian Noise"), 
                                                        string_type="filtered noise mag")


    # TEST 4.2: filtered interpolation derivative accuracy vs noise amplitude 
    # at fixed sampling rate and h step

    if test == 4.2:
        if plot_flag:
            for m in reversed(noise_mag):
                stored_data_noisy_filtered[m] = run_test4_2(interp_type=interp_type, mag=m, h=fixed_h, 
                                                                        M=fixed_M, plot_flag=plot_flag, 
                        string_title_known_to_interp=(string_title_known_to_interp_derivative + ", Order 4"),
                                            string_title_error=(string_title_error_derivative+ ", Order 4"))
        else:
            for m in reversed(dense_noise_mag):
                stored_data_noisy_filtered[m] = run_test4_2(interp_type=interp_type, mag=m, h=fixed_h, M=fixed_M, 
                                                                                            plot_flag=plot_flag, 
                            string_title_known_to_interp=(string_title_known_to_interp_derivative + ", Order 4"),
                                            string_title_error=(string_title_error_derivative+ ", Order 4"))
                
            print("\nOrder 4 Derivative Tests With Filtering White Gaussian Noise, Find Min. mag for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_noisy_filtered, test_list=list(reversed(dense_noise_mag)),
                                            avg_error_tol=avg_err_tol_noisy, max_dev_tol=max_dev_tol_noisy, 
                                                                            string_type="filtered noise mag")
        if plot_flag:    
            plot_error_per_rate(stored_data_noisy_filtered, noise_mag, 
                                string_title=(string_title_known_to_interp_derivative + 
                                              ", with Filtering White Gaussian Noise, Order 4"), 
                                                                    string_type="filtered noise mag")
        else:
            plot_error_per_rate(stored_data_noisy_filtered, dense_noise_mag, 
                                string_title=(string_title_known_to_interp_derivative + 
                                              ", with Filtered White Gaussian Noise, Order 4"), 
                                                                    string_type="filtered noise mag")
            
    # TEST 5: FILTERING WGN, SWEEPING WINDOW FOR MOVING AVERAGE FILTER

    # TEST 5.1: filtered interpolation accuracy vs window size for moving avg filter at fixed 
                                                            # noise amplitude and fixed sampling rate
    fixed_mag = 1/3 * 0.1
    window_size = [7, 9, 11, 13, 15, 17]
    dense_window_size = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 45, 75, 93]

    if test == 5.1:
        if plot_flag:
            for w in window_size:
                stored_data_noisy_filtered[w] = run_test4_1(interp_type=interp_type, filter_window=w, 
                                                mag=fixed_mag, M=fixed_M, plot_flag=plot_flag, 
                                                string_title_known_to_interp=string_title_known_to_interp,
                                                string_title_error=string_title_error, filter_sweep_flag=True)
        else:
            for w in dense_window_size:
                stored_data_noisy_filtered[w] = run_test4_1(interp_type=interp_type, filter_window=w, 
                                                mag=fixed_mag, M=fixed_M, 
                                                plot_flag=plot_flag, filter_sweep_flag=True,
                                                string_title_known_to_interp=string_title_known_to_interp,
                                                string_title_error=string_title_error)
                
            print("\nFiltering White Gaussian Noise Case, Find Min. window filter size for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_noisy_filtered, test_list=dense_window_size,
                                            avg_error_tol=avg_err_tol_noisy, max_dev_tol=max_dev_tol_noisy, 
                                                                            string_type="filter window size")
        if plot_flag:    
            plot_error_per_rate(stored_data_noisy_filtered, window_size, 
                                string_title=(string_title_known_to_interp + 
                                              ", with Filtering White Gaussian Noise"), 
                                                        string_type="filter window size")
        else:
            plot_error_per_rate(stored_data_noisy_filtered, dense_window_size, 
                                string_title=(string_title_known_to_interp + 
                                              ", with Filtering White Gaussian Noise"), 
                                                        string_type="filter window size")
            
    
    # TEST 5.2: filtered interpolation derivative accuracy vs window size for moving avg filter 
                                                # at fixed noise amplitude, sampling rate, and h step

    if test == 5.2:
        if plot_flag:
            for w in window_size:
                stored_data_noisy_filtered[w] = run_test4_2(interp_type=interp_type, h=fixed_h, filter_window=w, 
                                mag=fixed_mag, M=fixed_M, plot_flag=plot_flag, filter_sweep_flag=True, 
                                string_title_known_to_interp=(string_title_known_to_interp_derivative + ", Order 4"),
                                string_title_error=(string_title_error_derivative+ ", Order 4"))
        else:
            for w in dense_window_size:
                stored_data_noisy_filtered[w] = run_test4_2(interp_type=interp_type, h=fixed_h, filter_window=w, 
                                mag=fixed_mag, M=fixed_M, plot_flag=plot_flag, filter_sweep_flag=True,
                                string_title_known_to_interp=(string_title_known_to_interp_derivative + ", Order 4"),
                                string_title_error=(string_title_error_derivative + ", Order 4"))
                
            print("\n Order 4 Derivative Tests With Filtering White Gaussian Noise Case, " + 
                f"Find Min. window filter size for Tolerance\n")
            find_min_for_tolerance(stored_data=stored_data_noisy_filtered, test_list=dense_window_size,
                                            avg_error_tol=avg_err_tol_noisy, max_dev_tol=max_dev_tol_noisy, 
                                                                            string_type="filter window size")
        if plot_flag:    
            plot_error_per_rate(stored_data_noisy_filtered, window_size, 
                                string_title=(string_title_known_to_interp + 
                                              ", with Filtering White Gaussian Noise, Order 4"), 
                                                        string_type="filter window size")
        else:
            plot_error_per_rate(stored_data_noisy_filtered, dense_window_size, 
                                string_title=(string_title_known_to_interp + 
                                              ", with Filtering White Gaussian Noise, Order 4"), 
                                                        string_type="filter window size")
