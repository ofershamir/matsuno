#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:42:01 2019

@author: shlomi
"""
import unittest


def omega(w, k, n, parameters):
    OMEGA = parameters.get('angular_frequency', 'missing angular_frequency key\
                           from parameters dictionary...')
    G = parameters.get('gravitational_acceleration', 'missing \
                        gravitational_acceleration key from parameters\
                        dictionary...')
    A = parameters.get('mean_radius', 'missing mean_radius key\
                        from parameters dictionary...')
    H0 = parameters.get('layer_mean_depth', 'missing layer_mean_depth key\
                        from parameters dictionary...')
    omega = -w**3.0 + w * ((G * H0) * k**2 + 2 * OMEGA * (2 * n + 1)
                           * (G * H0)**0.5 / A) + (2 * OMEGA * G * H0 * k) / A
    return omega


def f(n):
    """pre_factor for scipy.special.eval_hermite function for comparison with
    pymaws eval_hermite_polynomial function.
    input : n integer and scaler"""
    import numpy as np
    return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**-0.5


class Test_pymaws(unittest.TestCase):
    def test_hermite(self):
        """Testing eval_hermite_polynomial in pymaws vs.
            scipy.special.eval_hermite with prefactor correction"""
        import numpy as np
        from pymaws import eval_hermite_polynomial
        from scipy.special import eval_hermite
        print('Testing pymaws.eval_hermite_polynomial...')
        thresh = 1e-6  
        ns = np.arange(1, 11)
        x = 20.0 * np.random.rand(100) - 10.0
        print('Using n`s from ' + str(min(ns)) + ' to ' + str(max(ns)) + ':')
        print('Using random x`s from ' + str(min(x)) + ' to ' + str(max(x)) +
              ':')
        for n in ns:
            # calculate with pymaws function:
            totest = eval_hermite_polynomial(x, n)
            # calculate with scipy function:
            test = f(n) * eval_hermite(n, x)
            # error is the norm of the x vector for each n:
            error = np.linalg.norm(test - totest)
            self.assertLessEqual(error, abs(thresh),
                                 'Relative error should be' +
                                 ' smaller than ' + str(thresh))
        print('Test complete with threshold of ' + str(thresh))

    def test_omega(self):
        """Testing eval_omega in pymaws vs. numpy.roots"""
        import numpy as np
        from pymaws import eval_omega
        from pymaws import Earth
        # unpack dictionary into vars:
        OMEGA = Earth.get('angular_frequency', 'missing angular_frequency key\
                               from parameters dictionary...')
        G = Earth.get('gravitational_acceleration', 'missing \
                            gravitational_acceleration key from parameters\
                            dictionary...')
        A = Earth.get('mean_radius', 'missing mean_radius key\
                            from parameters dictionary...')
        H0 = Earth.get('layer_mean_depth', 'missing layer_mean_depth key\
                            from parameters dictionary...')
        ns = np.random.randint(1, 51, 25)
        ks = np.random.randint(1, 51, 25)
        test = {'Rossby': 0, 'WIG': 0, 'EIG': 0}
        totest = test.copy()
        for k in ks:
            for n in ns:
                coeffs = [-1, 0, (G * H0) * k ** 2 + 2 * OMEGA * (2 * n + 1) *
                          (G * H0)**0.5 / A, (2 * OMEGA * G * H0 * k) / A]
                roots = np.roots(coeffs)
                test.update(Rossby=-min(abs(roots)))
                test.update(WIG=min(roots))
                test.update(EIG=max(roots))
                totest.update(Rossby=eval_omega(k, n, 'Rossby', Earth))
                totest.update(WIG=eval_omega(k, n, 'WIG', Earth))
                totest.update(EIG=eval_omega(k, n, 'EIG', Earth))
                error = {k: test[k] - totest[k] for k in test.keys()}
                print(test,totest)
                break
                # print(str(n), str(k), error)
                
if __name__ == '__main__':
    unittest.main()
