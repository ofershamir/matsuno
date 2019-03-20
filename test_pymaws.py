#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:42:01 2019
Some general rules of testing:
    # A testing unit should focus on one tiny bit of functionality and prove it
      correct.
    # Each test unit must be fully independent.
      Each test must be able to run alone, and also within the test suite,
      regardless of the order that they are called.
    # Try hard to make tests that run fast.
    # Learn your tools and learn how to run a single test or a test case.
    # Always run the full test suite before a coding session, and run it again
      after.
    # It is a good idea to implement a hook that runs all tests before pushing
      code to a shared repository.
    # If you are in the middle of a development session and have to interrupt
      your work, it is a good idea to write a broken unit test about what you
      want to develop next.
    # The first step when you are debugging your code is to write a new test
      pinpointing the bug. While it is not always possible to do,
      those bug catching tests are among the most valuable pieces of code in
      your project.
    # Use long and descriptive names for testing functions.
    # Another use of the testing code is as an introduction to new developers.

@author: shlomi
"""
import unittest


def f(n):
    """pre_factor for scipy.special.eval_hermite function for comparison with
    pymaws _eval_hermite_polynomial function.
    input : n integer and scaler"""
    import numpy as np
    return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**-0.5


class Test_pymaws(unittest.TestCase):
    def test_hermite(self):
        """Testing _eval_hermite_polynomial in pymaws vs.
            scipy.special.eval_hermite with prefactor correction:
            1)Random sample from x, and choose 1 to 10 n's.
            2)Run _eval_hermite_polynomial(x, n).
            3)Run eval_hermite(n, x) from spipy.special multiplied by f(n).
            f(n)=(2**n * np.math.factorial(n) * np.sqrt(np.pi))**-0.5.
            4)The error is the vector norm of the difference between the two
              results.
            5)The error for each n should be lower than the threshold (1e-6).
            The error ~ n because of the factorial in f(n)."""
        import numpy as np
        from pymaws import _eval_hermite_polynomial
        from scipy.special import eval_hermite
        print('Testing pymaws._eval_hermite_polynomial: ')
        thresh = 1e-6
        ns = np.arange(1, 11)
        x = 20.0 * np.random.rand(100) - 10.0
        print('Using ' + str(len(ns)) + ' n`s from ' + str(min(ns)) + ' to ' +
              str(max(ns)) + ':')
        print('Using ' + str(len(x)) + ' random x`s from ' +
              str(round(min(x), 2)) + ' to ' + str(round(max(x), 2)) + ':')
        for n in ns:
            # calculate with pymaws function:
            totest = _eval_hermite_polynomial(x, n)
            # calculate with scipy function:
            test = f(n) * eval_hermite(n, x)
            # error is the norm of the x vector for each n:
            error = np.linalg.norm(test - totest)
            self.assertLessEqual(error, abs(thresh),
                                 'Error should be' +
                                 ' smaller than ' + str(thresh))
        print('_eval_hermite_polynomial test complete with threshold of ' +
              str(thresh))

    def test_omega(self):
        """Testing _eval_omega in pymaws vs. numpy.roots:
           1)Randomly sample 25 k's and n's from 1-50.
           2)For each wave-type(Rossby, EIG, WIG) run _eval_omega(k, n)
             for Earth's parametes.
           3)Run np.roots with the coeffs of the cubic eq.#2 in:
             https://doi.org/10.5194/gmd-2018-260
             (Note: we scale k = k/A only for np.roots run.
           4)The error is the sum of the difference of steps 2,3.
           5)The error should be lower than the threshold of 1e-18.
           6)The sum of all three wave-types in step 2 should be lower than the
            threshold of 1e-18."""
        import numpy as np
        from pymaws import _eval_omega
        from pymaws import Earth
        from pymaws import _unpack_parameters
        print('Testing pymaws._eval_omega: ')
        # unpack dictionary into vars:
        OMEGA = _unpack_parameters(Earth, 'angular_frequency')
        G = _unpack_parameters(Earth, 'gravitational_acceleration')
        A = _unpack_parameters(Earth, 'mean_radius')
        H0 = _unpack_parameters(Earth, 'layer_mean_depth')
        ns = np.random.randint(1, 51, 25)
        ks = np.random.randint(1, 51, 25)
        print('Using ' + str(len(ns)) + ' random n`s from ' + str(min(ns)) +
              ' to ' + str(max(ns)) + ':')
        print('Using ' + str(len(ks)) + ' random k`s from ' + str(min(ks)) +
              ' to ' + str(max(ks)) + ':')
        test = {'Rossby': 0, 'WIG': 0, 'EIG': 0}
        totest = test.copy()
        thresh = 1e-18
        for k in ks:
            for n in ns:
                coeffs = [-1, 0, (G * H0) * (k/A)**2 + 2 * OMEGA * (2 * n + 1)
                          * (G * H0)**0.5 / A, (2 * OMEGA * G * H0 * k / A)
                          / A]
                roots = np.roots(coeffs)
                test.update(Rossby=-min(abs(roots)), WIG=min(roots),
                            EIG=max(roots))
                totest.update(_eval_omega(k, n, Earth))
                error = {k: test[k] - totest[k] for k in test.keys()}
                totest_sum = sum(test.values())
                self.assertLessEqual(sum([abs(x) for x in error.values()]),
                                     thresh, 'Error should be' +
                                     ' smaller than ' + str(thresh))
                self.assertLessEqual(totest_sum, thresh, 'Sum of all 3 roots '
                                     + 'should be smaller than ' + str(thresh))
        print('Test complete with threshold of ' + str(thresh))

    def test_v(self):
        """Testing _eval_meridional_velocity in pymaws using
            np.polynomial.hermite.hermgauss:
           1)Compute the Lamb parameter for Earth.
           2)"""
        from numpy.polynomial.hermite import hermgauss
        from pymaws import _eval_meridional_velocity
        from pymaws import Earth
        from pymaws import _unpack_parameters
        print('Testing eval_meridional_velocity: ')
        # unpack dictionary into vars:
        OMEGA = _unpack_parameters(Earth, 'angular_frequency')
        G = _unpack_parameters(Earth, 'gravitational_acceleration')
        A = _unpack_parameters(Earth, 'mean_radius')
        H0 = _unpack_parameters(Earth, 'layer_mean_depth')
        Lamb = (2. * OMEGA * A)**2 / (G * H0)
        x, w = hermgauss(10)
        v_10 = _eval_meridional_velocity(x, Lamb, n=10, amp=1e2)
        v_5 = _eval_meridional_velocity(x, Lamb, n=5, amp=1e2)
        print(sum(v_10*v_5*w))
        print(sum(v_10*v_10*w))

    def test_eval_field_amplitudes(self):
        """Testing _eval_field_amplitudes in pymaws using np.roots:
           1)
        """
        from pymaws import _eval_field_amplitudes
        from pymaws import _eval_omega
        from pymaws import _unpack_parameters
        from pymaws import Earth
        import numpy as np
        OMEGA = _unpack_parameters(Earth, 'angular_frequency')
        A = _unpack_parameters(Earth, 'mean_radius')
        ns = np.random.randint(1, 51, 25)
        ks = np.random.randint(1, 51, 25)
        print('Using ' + str(len(ns)) + ' random n`s from ' + str(min(ns)) +
              ' to ' + str(max(ns)) + ':')
        print('Using ' + str(len(ks)) + ' random k`s from ' + str(min(ks)) +
              ' to ' + str(max(ks)) + ':')
        lats = np.deg2rad(np.linspace(-90, 90))
        for k in ks:
            for n in ns:
                for lat in lats:
                    omega = _eval_omega(k, n, Earth)
                    v_hat = _eval_field_amplitudes(lat, k, n, 'v', 'Rossby',
                                                   Earth)
                    phi_hat = _eval_field_amplitudes(lat, k, n, 'phi',
                                                     'Rossby', Earth)
                    u_hat = _eval_field_amplitudes(lat, k, n, 'u', 'Rossby',
                                                   Earth)
                    
if __name__ == '__main__':
    unittest.main()
