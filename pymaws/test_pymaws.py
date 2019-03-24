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
        print('')

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
                coeffs = [-1, 0, (G * H0) * (k / A)**2 + 2 * OMEGA *
                          (2 * n + 1) * (G * H0)**0.5 / A, (2 * OMEGA * G *
                          H0 * k / A) / A]
                roots = np.roots(coeffs)
                test.update(Rossby=-min(abs(roots)), WIG=min(roots),
                            EIG=max(roots))
                totest.update(_eval_omega(k, n, Earth))
                error = {k: test[k] - totest[k] for k in test.keys()}
                totest_sum = sum(test.values())
                self.assertLessEqual(sum([abs(x) for x in error.values()]),
                                     thresh, 'Error should be' +
                                     ' smaller than ' + str(thresh))
                self.assertLessEqual(totest_sum, thresh,
                                     'Sum of all 3 roots ' +
                                     'should be smaller than ' + str(thresh))
        print('_eval_omega test complete with threshold of ' + str(thresh))
        print('')

    def test_v(self):
        """Testing _eval_meridional_velocity in pymaws using
            np.trapz, basically testing for orthonormallity of the meridional
            velocity functions:
           1)Compute the Lamb parameter for Earth.
           2)Assaign x to be latitude range from -80 to 80 in radians.
           3)Sample an amplitude from 1e-15 to 1e15 uniformly.
           4)Run _eval_meridional_velocity with n=10, x ,Lamb, amp
           5)Run _eval_meridional_velocity with n=5, x ,Lamb, amp
           6)The error is an integral of step 4 times step 5 times a scaling
            constant.
           7)The term that should equal one is the integral of step 4 times
             step 4 (or step 5 times step 5) times the same scaling factor
             from step 6."""
        from pymaws import _eval_meridional_velocity
        from pymaws import Earth
        from pymaws import _unpack_parameters
        import numpy as np
        import random
        print('Testing _eval_meridional_velocity: ')
        # unpack dictionary into vars:
        OMEGA = _unpack_parameters(Earth, 'angular_frequency')
        G = _unpack_parameters(Earth, 'gravitational_acceleration')
        A = _unpack_parameters(Earth, 'mean_radius')
        H0 = _unpack_parameters(Earth, 'layer_mean_depth')
        Lamb = (2. * OMEGA * A)**2 / (G * H0)
        thresh = 1e-15
        lats = np.deg2rad(np.linspace(-80., 80., 100))
        print('Using ' + str(len(lats)) + ' lats from ' +
              str(round(min(lats), 2)) + ' to ' +
              str(round(max(lats), 2)) + ':')
        amp = random.uniform(np.log(1e-15), np.log(1e15))
        amp = np.exp(amp)
        print('Using amplitude: {0:.2g}'.format(amp))
        v_10 = _eval_meridional_velocity(lats, Lamb, n=10, amp=amp)
        v_5 = _eval_meridional_velocity(lats, Lamb, n=5, amp=amp)
        error = np.trapz(v_10 * v_5, lats) * Lamb**0.25 / (amp**2)
        one = np.trapz(v_10 * v_10, lats) * Lamb**0.25 / (amp**2)
        self.assertLessEqual(error, thresh, 'Error should be' +
                             ' smaller than ' + str(thresh))
        self.assertAlmostEqual(one, 1.0, 7)
        print('_eval_meridional_velocity test complete with threshold of ' +
              str(thresh))
        print('')

    def test_eval_field_amplitudes(self):
        """Testing _eval_field_amplitudes in pymaws:
           1)Sample random n's and k's from 1-15 and from 1-50 respectively.
           2)Sample 10 latitudes from -80 to 80 in radians.
           3)Sample 1 radnom amplitude from 1e-15 to 1e15 uniformly.
           4)For each wave_type and n and k, compute the omega using
            _eval_omega and v_hat, u_hat, phi_hat
            the error term is (-1j * omega * u_hat - 2 * OMEGA * lats * v_hat +
                             1j * k / (A) * phi_hat) / amp
           5)The final error term is sum(abs(np.real(error from step 4)))
        """
        from pymaws import _eval_field_amplitudes
        from pymaws import _eval_omega
        from pymaws import _unpack_parameters
        from pymaws import Earth
        import random
        import numpy as np
        print('Testing pymaws._eval_field_amplitudes: ')
        OMEGA = _unpack_parameters(Earth, 'angular_frequency')
        A = _unpack_parameters(Earth, 'mean_radius')
        # ns = np.arange(1, 11)
        ns = np.random.randint(1, 16, 10)
        ks = np.random.randint(1, 51, 25)
        print('Using ' + str(len(ns)) + ' random n`s from ' + str(min(ns)) +
              ' to ' + str(max(ns)) + ':')
        print('Using ' + str(len(ks)) + ' random k`s from ' + str(min(ks)) +
              ' to ' + str(max(ks)) + ':')
        lats = np.deg2rad(80.0 * np.random.rand(10) - 40.0)
        print('Using ' + str(len(lats)) + ' random lats from ' +
              str(round(min(lats), 2)) + ' to ' + str(round(max(lats), 2)) +
              ':')
        waves = ['Rossby', 'EIG', 'WIG']
        thresh = 1e-15
        for wave in waves:
            print('Testing ' + wave + ' wave_type:')
            amp = random.uniform(np.log(1e-15), np.log(1e15))
            amp = np.exp(amp)
            # print("Sammy ate {0:.3f} percent of a pizza!".format(75.765367))
            print('Using amplitude: {0:.2g}'.format(amp))
            for k in ks:
                for n in ns:
                    omega = _eval_omega(k, n, Earth)[wave]
                    v_hat = _eval_field_amplitudes(lats, k, n, amp, 'v', wave,
                                                   Earth)
                    phi_hat = _eval_field_amplitudes(lats, k, n, amp, 'phi',
                                                     wave, Earth)
                    u_hat = _eval_field_amplitudes(lats, k, n, amp, 'u', wave,
                                                   Earth)
                    error = (-1j * omega * u_hat - 2 * OMEGA * lats * v_hat +
                             1j * k / (A) * phi_hat) / amp
                    summed_error = sum(abs(np.real(error)))
                    self.assertLessEqual(summed_error,
                                         thresh, 'Error should be' +
                                         ' smaller than ' + str(thresh))
            print(' Test for ' + wave +
                  ' done with threshold of ' + str(thresh))
        print('_eval_meridional_velocity test for all wave types done' +
              ' with threshold of ' + str(thresh))
        print('')

    def test_eval_field(self):
        """Testing eval_field in pymaws:
           1)Sample 10 latitudes from -80 to 80 in radians.
           2)Sample 1 radnom amplitude from 1e-15 to 1e15 uniformly.
           3)pick lon1, lon2, time1 and time2 to be 1, 2, 1, 1.01 respectivly.
           4)Run eval_field for v, u(two times) and phi(two lons).
           5)The error term is ((u2 - u1) / (time2 - time1) - 2 * OMEGA * lats
             * v1 + 1.0 / (A) * (phi2 - phi1) / (lon2 - lon1)) / amp
           6)The summed error term is sum(abs(error from step 5))
            """
        from pymaws import eval_field
        from pymaws import Earth
        from pymaws import _unpack_parameters
        import numpy as np
        import random
        print('Testing pymaws.eval_field: ')
        thresh = 1e-5
        OMEGA = _unpack_parameters(Earth, 'angular_frequency')
        A = _unpack_parameters(Earth, 'mean_radius')
        lats = np.deg2rad(80.0 * np.random.rand(10) - 40.0)
        print('Using ' + str(len(lats)) + ' random lats from ' +
              str(round(min(lats), 2)) + ' to ' + str(round(max(lats), 2)) +
              ':')
        lon1 = np.deg2rad(1.0)
        lon2 = np.deg2rad(2.0)
        time1 = 1.0
        time2 = 1.01
        amp = random.uniform(np.log(1e-15), np.log(1e15))
        amp = np.exp(amp)
        print('Using amplitude: {0:.2g}'.format(amp))
        nj = lats.shape[0]
        v1 = np.zeros((nj))
        u1 = np.zeros((nj))
        u2 = np.zeros((nj))
        phi1 = np.zeros((nj))
        phi2 = np.zeros((nj))
        for j in range(nj):
            v1[j] = eval_field(lats[j], lon1, time1, field='v', amp=amp)
            u1[j] = eval_field(lats[j], lon1, time1, field='u', amp=amp)
            u2[j] = eval_field(lats[j], lon1, time2, field='u', amp=amp)
            phi1[j] = eval_field(lats[j], lon1, time1, field='phi', amp=amp)
            phi2[j] = eval_field(lats[j], lon2, time1, field='phi', amp=amp)
        error = ((u2 - u1) / (time2 - time1) - 2 * OMEGA * lats *
                 v1 + 1.0 / (A) * (phi2 - phi1) / (lon2 - lon1)) / amp
        summed_error = sum(abs(error))
        self.assertLessEqual(summed_error,
                             thresh, 'Error should be' +
                             ' smaller than ' + str(thresh))
        print('')


if __name__ == '__main__':
    unittest.main()
