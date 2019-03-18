#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:42:01 2019

@author: shlomi
"""
import unittest


def f(n):
    """pre_factor for scipy.special.eval_hermite function for comparison with
    pymaws eval_hermite_polynomial function.
    input : n integer and scaler"""
    import numpy as np
    return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**-0.5


class Test_pymaws(unittest.TestCase):
    def test_hermite(self):
        import numpy as np
        from pymaws import eval_hermite_polynomial
        from scipy.special import eval_hermite
        eps = 1e-4
        ns = np.arange(1, 10)
        xs = np.linspace(-12, 12)  # typical numbers for Earth
        for n in ns:
            for x in xs:
                totest = eval_hermite_polynomial(x, int(n))
                test = f(n) * eval_hermite(n, x)
                dif = abs(test - totest)
                # print('n:' + str(n), 'x:' + str(x))
                # print('test:' + str(test))
                # print('to_test:' + str(totest))
                # print('difference:' + str(dif))
                self.assertLessEqual(dif, eps,
                                     'absolute of the difference should be' +
                                     ' smaller than ' + str(eps))


if __name__ == '__main__':
    unittest.main()
