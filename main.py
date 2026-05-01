# main.py

# README --> go to main and uncomment which test you want to try out

import numpy as np
from run_uniform_sampling_interp_tests import run_uniform_sampling_interp_tests
from run_adaptive_sampling_interp_tests import run_adaptive_sampling_interp_tests


def main():

    """
    main()

    calls:

        run_uniform_sampling_interp_tests(test=None, plot_flag=True, interp_type=None):

        test is test num (1, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2)
        plot flag is flag for iteritive plotting
        interp type is like "linear" or "cubic"

    plot_error_per_rate() will always plot:

        avg error and max deviation vs sweept input

    setting plot flag to 0 / False enables:
    
        find_min_for_tolerance() with a denser sweept grid input

    setting the plot flag to 1 / True enables iterative plotting: 

        side by side of known signal and interpolated signal
        subplots of known, interpolated, deviation, and error signals

        iterative plotting: 

            sweeping through sampling rates, h steps, and noise magnitudes 


    calls:

        run_adaptive_sampling_interp_tests(test=None, plot_flag=True, interp_type=None, informed_type=None):

        test is test num (1, 1.1, 1.2, 2, 2.1, 2.2)
        plot flag is flag for iterative plotting
        interp type is string for specifying linear vs cubic interp
            valid input: "linear" or "cubic"

        informed_type is string for specifying adaptive sampling strategy to be 
        magnitude-informed (f) or slope-informed (f') or curvature-informed (f'')
            valid input: "mag" or "slope" or "curvature"

    plotting works same as for uniform sampling

        iterative plotting: 

        sweeping through alpha weighting, sweeping through samples used to interpolate 

    """

    try: 
        # -----------------------------
        # FINAL PROJECT TEST CALLS
        # Uncomment one test at a time
        #
        # Function formats:
        # run_uniform_sampling_interp_tests(test, plot_flag, interp_type)
        # run_adaptive_sampling_interp_tests(test, plot_flag, interp_type, informed_type)
        #
        # plot_flag: 0 = summary sweep plots only, 1 = iterative plots
        # interp_type: "linear" or "cubic"
        # informed_type: "mag", "slope", or "curvature"
        #
        # change inputs as desired
        # -----------------------------


    # -----------------------------------
    # UNIFORM TESTS
    # all uniform sampling tests use the basic sine wave function
    # derivatives tests compare 2nd-order and 4th-order centered differences
    # other tests using derivatives use the 4th-order centered differences
    # -----------------------------------


        # -----------------------------
        # TEST 1: Uniform sampling interpolation accuracy vs sampling rate
        # -----------------------------

        # run_uniform_sampling_interp_tests(1, 0, "cubic")

        # -----------------------------
        # TEST 2.1: Numerical derivative accuracy vs sampling rate
        # -----------------------------

        # run_uniform_sampling_interp_tests(2.1, 0, "linear")

        # -----------------------------
        # TEST 2.2: Numerical derivative accuracy vs h step
        # -----------------------------

        # run_uniform_sampling_interp_tests(2.2, 1, "linear")

        # -----------------------------
        # TEST 3.1: Noisy interpolation accuracy vs noise magnitude
        # -----------------------------

        # run_uniform_sampling_interp_tests(3.1, 1, "cubic")

        # -----------------------------
        # TEST 3.2: Noisy derivative accuracy vs noise magnitude
        # -----------------------------

        # run_uniform_sampling_interp_tests(3.2, 1, "cubic")

        # -----------------------------
        # TEST 4.1: Filtered noisy interpolation accuracy vs noise magnitude
        # -----------------------------

        # run_uniform_sampling_interp_tests(4.1, 1, "cubic")

        # -----------------------------
        # TEST 4.2: Filtered noisy derivative accuracy vs noise magnitude
        # -----------------------------

        # run_uniform_sampling_interp_tests(4.2, 0, "linear")

        # -----------------------------
        # TEST 5.1: Filtered interpolation accuracy vs moving-average window size
        # -----------------------------

        # run_uniform_sampling_interp_tests(5.1, 0, "cubic")

        # -----------------------------
        # TEST 5.2: Filtered derivative accuracy vs moving-average window size
        # -----------------------------

        # run_uniform_sampling_interp_tests(5.2, 1, "linear")


    # -----------------------------------
    # ADAPTIVE TESTS
    # tests use the basic, multi-frequency, 
    # and burst/modulated sine wave functions
    # -----------------------------------


        # -----------------------------
        # ADAPTIVE TEST 1: Alpha sweep, basic sine signal
        # -----------------------------

        # run_adaptive_sampling_interp_tests(1, 0, "linear", "mag")
        # run_adaptive_sampling_interp_tests(1, 0, "cubic", "curvature")

        # -----------------------------
        # ADAPTIVE TEST 1.1: Alpha sweep, multi-frequency sine signal
        # -----------------------------

        # run_adaptive_sampling_interp_tests(1.1, 0, "cubic", "mag")
        # run_adaptive_sampling_interp_tests(1.1, 0, "linear", "curvature")

        # -----------------------------
        # ADAPTIVE TEST 1.2: Alpha sweep, burst/modulated sine signal
        # -----------------------------

        # run_adaptive_sampling_interp_tests(1.2, 0, "linear", "mag")
        # run_adaptive_sampling_interp_tests(1.2, 1, "cubic", "curvature")

        # -----------------------------
        # ADAPTIVE TEST 2: Uniform vs adaptive sample-efficiency, basic sine signal
        # -----------------------------

        # run_adaptive_sampling_interp_tests(2, 0, "cubic", "curvature")
        # run_adaptive_sampling_interp_tests(2, 0, "linear", "curvature")

        # -----------------------------
        # ADAPTIVE TEST 2.1: Uniform vs adaptive sample-efficiency, multi-frequency sine signal
        # -----------------------------

        # run_adaptive_sampling_interp_tests(2.1, 0, "cubic", "curvature")
        run_adaptive_sampling_interp_tests(2.1, 0, "linear", "curvature")

        # -----------------------------
        # ADAPTIVE TEST 2.2: Uniform vs adaptive sample-efficiency, burst/modulated sine signal
        # -----------------------------

        # run_adaptive_sampling_interp_tests(2.2, 0, "linear", "mag")
        # run_adaptive_sampling_interp_tests(2.2, 1, "linear", "curvature")

        # Final showcase case: curvature-informed adaptive sampling 
        # reduced samples needed to reach the desired tolerance
        # compared to uniform sampling for linear interpolation 
        # for the tested multi-frequency sine and burst/modulated sine signals


    except Exception as e:
        print(f"\nError: {e}\n")
        raise



if __name__ == "__main__":
    """ main """
    main()