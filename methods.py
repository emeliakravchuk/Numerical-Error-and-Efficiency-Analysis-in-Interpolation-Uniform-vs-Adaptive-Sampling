# methods.py

import numpy as np

# Both derivatives implementations reused from previous lab/hw and based on course textbook.
def centered_difference_4th_order(f=None, h=1):

    """
    centered_difference_4th_order(f, h)
    has error O(h^4)
    returns df(x) as a lambda
    """
    if f is None or not callable(f):
        raise TypeError(f"f must be callable.\n")
    if not isinstance(h, (int, float)):
        raise TypeError("h must be an int or float\n")
    if h <= 0:
        raise ValueError("h must be > 0\n")
    
    df = lambda x: ( -f(x + 2*h)  + 8 * f(x + h)  - 8 * f(x - h)  + f(x - 2*h) ) / (12*h)
    return df


# Both derivatives implementations reused from previous lab/hw and based on course textbook.
def centered_difference_2nd_order(f=None, h=1):

    """
    centered_difference_2nd_order(f, h)
    has error O(h^2)
    returns df(x) as a lambda
    """
    if f is None or not callable(f):
        raise TypeError("f must be callable.\n")
    if not isinstance(h, (int, float)):
        raise TypeError("h must be an int or float\n")
    if h <= 0:
        raise ValueError("h must be > 0\n")

    df = lambda x: (f(x + h) - f(x - h)) / (2*h)
    return df


# Based on random signals course assignments.
def white_guassian_noise(samples=None):

    """
    white_guassian_noise(samples=None)

    input: samples for the noise signal

    noise is the normal guassian distribution 
    with mean=0, variance=1

    returns noise
    """
    if samples is None:
        raise ValueError("\nNumber of samples must be provided.\n")
    if samples <= 0:
        raise ValueError("\nNumber of samples must be a positive integer.\n")
    if not isinstance(samples, int):
        raise TypeError("\nSamples must be int type.\n")
    
    # Generate White Gaussian Noise
    noise = np.random.normal(0, 1, samples)

    return noise 


def piecewise_ordering(Li=None, x=None, y=None):

    """
    piecewise_ordering(Li=None, x=None, y=None)

    input: Li is all lines from paired data, but it is not ordered / piecewise joined

    this function is going to join the Li segments 

    returns f(xx) as a function handle 
    """
    if Li is None or not callable(Li):
        raise ValueError("\nLi must be provided and callable.\n")
    if x is None or y is None:
        raise ValueError("\nx and y must be provided.\n")
    
    def f(xx):
        xx = np.asarray(xx)
        if np.any(xx < x[0]) or np.any(xx > x[-1]):
            print(f"\nWarning, some value xx is outside the interpolation range [{x[0]}, {x[-1]}].\n")

        if xx.ndim == 0: # scalar input
            vals = Li(xx)
            i = min(np.searchsorted(x, xx, side="right") - 1, len(x) - 2)
            return vals[i]
        
        else:
            y_out = np.zeros(len(xx), dtype=float)

            for k in range(len(xx)):
                vals = Li(xx[k])
                i = min(np.searchsorted(x, xx[k], side="right") - 1, len(x) - 2)
                y_out[k] = vals[i]

            return y_out

    return f


def linear_interp(paired=None):

    """
    linear_interp(paired=None)

    takes paired values of (x,y)
    x should be increasing monotonically 

    calls piecewise_ordering()

    Li is all lines from paired data, but it is not ordered / piecewise joined

    returns f(xx) as a function handle 
    """

    if paired is None:
        raise ValueError("\nOrdered paires (x, y) must be provided.\n")
    if len(paired) < 2:
        raise ValueError("\nAt least two data points are required.\n")
    
    paired.sort()
    x, y = zip(*paired) # unzip
    x = np.array(x, dtype=float) # to float array
    y = np.array(y, dtype=float)

    if np.any(np.diff(x) <= 0):
        raise ValueError("\nThe provided x values must be strictly increasing.\n")

    Li = lambda xx: y[0:-1] + (y[1:] - y[0:-1]) / (x[1:] - x[0:-1]) * (xx - x[0:-1])

    f = piecewise_ordering(Li, x, y)
    
    return f


def averaging_noise_filter(M=None, paired=None):

    """
    averaging_noise_filter(M=None, paired=None)

    M is number of points in the moving average 
    paired is the (x,y) noisy data points used to construct the noisy signal
    the function will filter the noisy data points

    returns x_filtered, y_filtered
    """
    if M is None:
        raise ValueError("\nM cannot be None.\n")
    if not isinstance(M, int):
        raise TypeError("\nM must be int type.\n")
    if M < 1:
        raise ValueError("\nM must be >= 1.\n")
    if M % 2 == 0:
        raise ValueError("\nwindow must be odd for symmetric averaging.\n")
    if paired is None:
        raise ValueError("\nOrdered paires (x, y) must be provided.\n")
    if len(paired) < 2:
        raise ValueError("\nAt least two data points are required.\n")
    
    paired.sort()
    x, y = zip(*paired) # unzip
    x = np.array(x, dtype=float) # to float array
    y = np.array(y, dtype=float)

    if np.any(np.diff(x) <= 0):
        raise ValueError("\nThe provided x values must be strictly increasing.\n")
    if M > len(x):
        raise ValueError("\nwindow cannot be larger than signal length.\n")
    
    k = M // 2
    y_filt = np.zeros(len(y), dtype=float)
    
    # symmetric for center nodes
    i = np.arange(k, len(y) - k)
    y_filt[i] = sum(y[i-k+j] for j in range(M)) / M
    # forward looking at beginning nodes
    i_begin = np.arange(k)
    y_filt[i_begin] = sum(y[i_begin+j] for j in range(M)) / M
    # backward looking for ending nodes
    i_end = np.arange(len(y) - k, len(y))
    y_filt[i_end] = sum(y[i_end-M+1+j] for j in range(M)) / M

    x_filt = x

    return x_filt, y_filt


def error_analysis(known_signal=None, interp_signal=None, x_dense=None):

    """
    error_analysis(known_signal=None, interp_signal=None, x_dense=None)

    takes known signal, interpolated signal, dense x grid

    returns dict:
            return {
                "avg_error": avg_error,
                "max_error": max_error,
                "max_error_index": max_error_index,
                "max_error_time": max_error_time,
                "max_deviation": max_deviation,
                "max_deviation_index": max_deviation_index,
                "max_deviation_time": max_deviation_time,
                "error_signal": error_signal,
                "deviation_signal": deviation_signal,
                "interp_signal": y_interp,
                "known_signal": y_known
                }
    """
    if known_signal is None or interp_signal is None or x_dense is None:
        raise ValueError("\nAll function parameters have to be provided.\n")
    
    y_known = known_signal(x_dense)
    y_interp = interp_signal(x_dense)
    
    deviation_signal = y_interp - y_known
    abs_dev = np.abs(deviation_signal)
    epsilon = 1e-12
    tol = 0.1 * np.max(np.abs(y_known))
    # another way to try: error_signal = np.abs(deviation_signal / ( y_known + epsilon ) )
    error_signal = np.where(
            np.abs(y_known) < tol, 
            abs_dev,
            abs_dev / np.abs(y_known + epsilon))

    max_deviation_index = np.abs(deviation_signal).argmax()
    max_deviation_time = x_dense[max_deviation_index]
    max_deviation = np.abs(deviation_signal[max_deviation_index]) # changed to be abs deviation instead of signed dev

    max_error_index = error_signal.argmax()
    max_error_time = x_dense[max_error_index]
    max_error = np.max(error_signal)
    avg_error = np.mean(error_signal)

    return {
        "avg_error": avg_error,
        "max_error": max_error,
        "max_error_index": max_error_index,
        "max_error_time": max_error_time,
        "max_deviation": max_deviation,
        "max_deviation_index": max_deviation_index,
        "max_deviation_time": max_deviation_time,
        "error_signal": error_signal,
        "deviation_signal": deviation_signal,
        "interp_signal": y_interp,
        "known_signal": y_known
        }


def find_min_for_tolerance(stored_data=None, test_list=None,
                avg_error_tol=1e-3, max_dev_tol=1e-3, string_type="M"):
    
    """
    find_min_for_tolerance(stored_data=None, test_list=None,
                                   avg_error_tol=None, max_dev_tol=None):

    stored_data is a dict of the form:
        stored_data[M] = {
            "avg_error": ...,
            "max_error": ...,
            "sampling_rate": ...,
            "samples": ...,
            }

    where M is the scaling of the signal frequency
    test_list is the list of M values

    function finds min number of samples needed to achieve some error tolerance 

    return ok_avg_err, ok_max_dev
        where ok_avg_err, ok_max_dev are like = [error found under tol, x_samples, M]
    """
    if stored_data is None or test_list is None or avg_error_tol is None or max_dev_tol is None:
        raise ValueError("\nAll function parameters have to be provided.\n")
    if not isinstance(avg_error_tol, (int, float)):
        raise TypeError("\navg_err_tol tolerance must be positive real number\n")
    if not isinstance(max_dev_tol, (int, float)):
        raise TypeError("\nmax_dev_tol tolerance must be positive real number\n")
    if float(max_dev_tol) <= 0:
        print("\nWarning, max_dev_tol must be positive real number. Using default.\n")
        max_dev_tol = 1e-3
    max_dev_tol = abs(float(max_dev_tol))
    if float(avg_error_tol) <= 0:
        print("\nWarning, avg_error_tol must be positive real number. Using default.\n")
        avg_error_tol = 1e-3
    avg_error_tol = abs(float(avg_error_tol))

    M_vals = test_list
    avg_err = np.zeros(len(test_list), dtype=float)
    max_dev = np.zeros(len(test_list), dtype=float)
    ok_avg_err = [None] * 3
    ok_max_dev = [None] * 3
    temp_dict = {}
    flag_avg_error = False
    flag_max_dev = False

    for i in range(len(M_vals)):
        temp_dict = stored_data[M_vals[i]]
        avg_err[i] = temp_dict["avg_error"]
        max_dev[i] = temp_dict["max_deviation"]

        # now the avg_err and max_dev <= tol do not have to occur at same time
        if (not flag_avg_error) and avg_err[i] <= avg_error_tol:
            ok_avg_err[0] = avg_err[i]
            ok_avg_err[1] = len(temp_dict["x_samples"])
            ok_avg_err[2] = M_vals[i]
            flag_avg_error = True

        if (not flag_max_dev) and abs(max_dev[i]) <= max_dev_tol:
            ok_max_dev[0] = max_dev[i]
            ok_max_dev[1] = len(temp_dict["x_samples"])
            ok_max_dev[2] = M_vals[i]
            flag_max_dev = True

        if flag_max_dev == True and flag_avg_error == True:
            break
            
    print("\nMinimum samples meeting tolerances:")

    if ok_avg_err[0] is None:
        print(f"No case satisfied average error tolerance {avg_error_tol}\n")
    else:
        print(
            f"Average error <= {avg_error_tol}: \n"
            f"avg_error = {ok_avg_err[0]}, samples = {ok_avg_err[1]}, {string_type} = {ok_avg_err[2]}"
        )

    if ok_max_dev[0] is None:
        print(f"No case satisfied max deviation tolerance {max_dev_tol}\n")
    else:
        print(
            f"Max deviation <= {max_dev_tol}: \n"
            f"max_deviation = {ok_max_dev[0]}, samples = {ok_max_dev[1]}, {string_type} = {ok_max_dev[2]}"
        )



def adaptive_sampling_weighting(weight_func=None, x_start=None, x_end=None,
                                samples=None, alpha=1.0, dense=2000, tol=1e-12, threshold=1e-12):
    
    """
    adaptive_sampling_weighting(weight_func=None, x_start=None, x_end=None,
                                samples=None, alpha=1.0, dense=2000, tol=1e-12, threshold=1e-12)

    weight_func:
        function used to decide where samples should cluster
            ex: weight_func = f' gives slope-based adaptive sampling
            ex: weight_func = f'' gives curvature-based adaptive sampling

    alpha:
        controls transition from uniform to adaptive sampling
            alpha = 0 --> uniform sampling
            alpha = 1 --> fully adaptive sampling

    weights where to sample based on weight_func magnitude
    such that the variable alpha controls the weighting distribution
    between uniform and adaptive sampling:

        density = (1 - alpha) * uniform_density + alpha * adaptive_density

    returns adaptive_x_samples
    """

    if weight_func is None or not callable(weight_func):
        raise TypeError("\nweight_func must be callable.\n")
    if x_start is None or x_end is None or samples is None:
        raise ValueError("\nx_start, x_end, and samples must be provided.\n")
    if any(not isinstance(v, (int, float)) for v in [x_start, x_end, alpha, tol, threshold]):
        raise TypeError("\nError, variables [x_start, x_end, alpha, tol, threshold] must be int or float type.\n")
    if any(not isinstance(v, (int)) for v in [samples, dense]):
        raise TypeError("\nError, variables [samples, dense] must be int type.\n")
    if x_start > x_end:
        x_start, x_end = x_end, x_start
    if x_start == x_end:
        raise ValueError("\nx_start and x_end cannot be equal.\n")
    if samples < 2:
        raise ValueError("\nsamples must be >= 2.\n")
    if dense < samples:
        raise ValueError("\ndense must be >= samples.\n")
    if threshold <= 0:
        raise ValueError("\nthreshold must be some positive real number.\n")
    if tol <= 0:
        raise ValueError("\ntol must be some positive real number.\n")
    if not (0 <= alpha <= 1):
        raise ValueError("\nalpha must satisfy 0 <= alpha <= 1.\n")

    x_dense = np.linspace(x_start, x_end, dense)
    raw_density = np.abs(weight_func(x_dense)) # abs value signal
    mean_density = np.mean(raw_density) # mean value
    std_density = np.std(raw_density) # standard deviation from mean

    # check for if the mag. is ~ zero
    if mean_density <= tol:
        print("\nWarning: weight function is basically zero everywhere. Returning uniform samples.\n")
        return np.linspace(x_start, x_end, samples)
    
    # check for if the variantion / deviation in mag is ~ zero
    if std_density <= threshold:
        print("\nWarning: weight function is nearly constant. Returning uniform samples.\n")
        return np.linspace(x_start, x_end, samples)

    # normalize to be able to weight from uniform to fully adaptive
    adaptive_density = raw_density / mean_density
    uniform_density = np.ones_like(adaptive_density)

    # adaptive weighting, ranges from uniform sampling to fully adaptive sampling
    density = (1 - alpha) * uniform_density + alpha * adaptive_density
    # cdf --> culmative distribution function 
    cumulative = np.cumsum(density)
    # normalize cdf so it ranges from 0 to 1
    cdf = cumulative / cumulative[-1]
    # evenly spaced intervals in probability/density space
    u = np.linspace(0, 1, samples)
    # mapping density to x location 
    adaptive_x_samples = np.interp(u, cdf, x_dense)
    # making sure correct start and end points 
    adaptive_x_samples[0] = x_start
    adaptive_x_samples[-1] = x_end

    return adaptive_x_samples