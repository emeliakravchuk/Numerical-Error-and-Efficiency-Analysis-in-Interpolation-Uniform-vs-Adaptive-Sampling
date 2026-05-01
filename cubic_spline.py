# cubic_spline.py file 

# The cubic spline implementation was modified from previous lab and based on course textbook.

import numpy as np

class Cubic_Spline:

    """
    Cubic_Splines class
    takes list of paired_data (x,y) input

    """

    def __init__(self, paired_data=None):

        """
        initialize the class
        takes list paired_data that's like [(x0, y0), (x1, y1), (x2, y2)]
        """
        if paired_data is None:
            raise ValueError("\npaired_data must contain x and y.\n")
        if len(paired_data) < 2:
            raise ValueError("\nAt least two data points are required.\n")
        
        paired_data.sort()
        x, y = zip(*paired_data) # unzip

        if not all(isinstance(v, (float, int)) for v in x):
            raise TypeError("\nx has to be int or float type.\n")

        if not all(isinstance(v, (float, int)) for v in y):
            raise TypeError("\ny has to be int or float type.\n")

        self.x_vec = np.asarray(x, dtype=float)
        self.y_vec = np.asarray(y, dtype=float)

        if len(self.x_vec) != len(self.y_vec):
            raise ValueError("\nx and y must have the same length.\n")
        if len(self.x_vec) < 2:
            raise ValueError("\nAt least two data points are required.\n")
        
        if np.any(np.diff(x) <= 0):
            raise ValueError("\nThe provided x values must be strictly increasing.\n")

        self.length = len(self.x_vec)

    
    def cubic_spline_natural_coeff(self):

        """
        cubic_spline_natural_coeff(self)
        compute natural cubic spline coefficients
        returns 
            [ a_i, b_i, c_i, d_i ]
            for each S_i (x)
        """
        # NATURAL CUBIC SPLINE
        # S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3
        # Derivatives:
        # S'_i(x)  = b_i + 2*c_i*(x - x_i) + 3*d_i*(x - x_i)^2
        # S''_i(x) = 2*c_i + 6*d_i*(x - x_i)

        # At node x_i:
        # S''(x_i) = 2*c_i  →  c_i = (1/2)*second derivative

        # a_i = y_i = function value = f_i   (on interval [x_i, x_i+1])
        # b_i = slope   (on interval [x_i, x_i+1])
        # c_i = curvature (2nd derivative) * 1/2   (at node x_i)
        # d_i = rate of change of curvature (3rd derivative) * 1/6   (on interval [x_i, x_i+1])

        # h_i = x_i+1  -  x_i
        # f_i+1  =  f_i  +  b_i h_i  +  c_i h_i^2  +  d_i h_i^3

        # endpoints of 2nd derivative = 0
        # c_0 = 0
        # c_n-1 = 0

        # system equations
        # h_i-1 * c_i-1   +   2 (h_i-1 + h_i) c_i   +   h_i * c_i+1   = 3 [ (y_i+1 - y_i) / h_i   -   (y_i - y_i-1) / h_i-1 ]

        # b_i  =  (y_i+1 - y_i) / h_i   -   (h_i / 3) (2*c_i + c_i+1)
        # d_i  =  (c_i+1 - c_i) / (3 h_i)

        # n = len(inputted vector); number of points
        # c_i is at nodes; i = 0 to n-1
        # h_i is total intervals; i from 0 to n-2
        # a_i, b_i, d_i are at intervals;  i = 0 to n-2
        # system is solving for c_i except for end points so: i = 1 to n-2

        x = self.x_vec
        y = self.y_vec
        n = self.length
        intervals = n - 1
        
        h = np.diff(x)
        a = y[:-1].copy()
        b = np.zeros((n-1), dtype=float)
        d = np.zeros((n-1), dtype=float)
        c = np.zeros(n, dtype=float)

        # only one interval
        if n == 2:
            # S_0(x) = a_0 + b_0 (x - x_0)
            # natural spline with 2 points is just a line, c = [0,0]
            b[0] = (y[1] - y[0]) / h[0]
            d[0] = 0.0
            return [ a, b, c, d ]
        
        # build tridiagonal system based on book
        A = np.zeros((n, n), dtype=float)
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        # diagonals
        np.fill_diagonal(A[1:-1, 1:-1], 2 * (h[:-1] + h[1:]))  # main diagonal
        np.fill_diagonal(A[1:-1, :-2],   h[:-1])                # lower diagonal
        np.fill_diagonal(A[1:-1, 2:],    h[1:])                 # upper diagonal
        # RHS
        rhs_full = np.zeros(n, dtype=float)
        # right hand side of the systems of equations for interrior nodes
        rhs = 3 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
        # RHS of system of equaitons
        rhs_full[1:-1] = rhs
        # solve for all c values
        c = np.linalg.solve(A, rhs_full)
        # solve for b and d values
        # 1 to n-1, 0 to n-2, 0 to n-2        0 to n-2, 1 to n-1
        b = (y[1:] - y[:-1]) / h  -   ( h * (2 * c[:-1] + c[1:]) ) / 3
        # 1 to n-1, 0 to n-2,    0 to n-2
        d = (c[1:] - c[:-1]) / (3 * h)

        return [a, b, c, d]
    

    def piecewise_ordering(self, Si=None, x=None, y=None):

        """
        piecewise_ordering(self, Si=None, x=None, y=None)

        input: Si is all lines from paired data, but it is not ordered / piecewise joined

        this function is going to join the Si segments 

        returns f(xx) as a function handle 
        """
        if Si is None or not callable(Si):
            raise ValueError("\nLi must be provided and callable.\n")
        if x is None or y is None:
            raise ValueError("\nx and y must be provided.\n")
        
        def f(xx):
            xx = np.asarray(xx)
            if np.any(xx < x[0]) or np.any(xx > x[-1]):
                print(f"\nWarning, some value xx is outside the interpolation range [{x[0]}, {x[-1]}].\n")

            if xx.ndim == 0: # scalar input
                vals = Si(xx)
                i = min(np.searchsorted(x, xx, side="right") - 1, len(x) - 2)
                return vals[i]
            
            else:
                y_out = np.zeros(len(xx), dtype=float)

                for k in range(len(xx)):
                    vals = Si(xx[k])
                    i = min(np.searchsorted(x, xx[k], side="right") - 1, len(x) - 2)
                    y_out[k] = vals[i]

                return y_out

        return f


    def cubic_spline_interpolation(self, coeff_vec=None):

        """
        cubic_spline_interpolation(self, coeff_vec)
        compute natural cubic spline interpolation

        takes coeff_vec which is
            [ a_i, b_i, c_i, d_i ]
            for each S_i (x) 

        returns interpolated function handle 
        """
        # S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3

        if coeff_vec is None or len(coeff_vec) < 4:
            raise ValueError("\nCoeffecient values must be provided.\n")
        
        a = coeff_vec[0]
        b = coeff_vec[1]
        c = coeff_vec[2]
        d = coeff_vec[3]

        x = self.x_vec
        y = self.y_vec
        
        Si = lambda xx: (
            a
            + b * (xx - x[0:-1])
            + c[0:-1] * (xx - x[0:-1])**2
            + d * (xx - x[0:-1])**3
        )

        f = self.piecewise_ordering(Si, x, y)
        
        return f