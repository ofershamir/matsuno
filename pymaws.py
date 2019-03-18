#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:03:39 2019
# make sure the low-level input is scaler and integer - Done.
# for validation we need a simple estimation of the fields
# need citation of the equations
# validation of hermite polynomial - Done :
    # our_hermite_func(x, n) = scipy.special.eval_hermite(n, x) * prefactor(n)
    # prefactor(n) = (2**n*n!*pi**0.5)**-0.5
# what to do if someone puts degrees instead of radians ? docs ?
@author: shlomi
"""
Earth = {
        'angular_frequency': 7.29212e-5,
        'gravitational_acceleration': 9.80616,
        'mean_radius': 6371220.,
        'layer_mean_depth': 30.
        }


def eval_omega(k, n, wave_type='Rossby', parameters=Earth):
    """
    Evaluates the wave-frequency for a given wave number and wave mode.

    Parameters
    ----------
    k : Integer, scalar
        spherical wave-number (dimensionless). must be >= 1.
    n : Integer, scaler
        wave-mode (dimensionless). must be >=1.
    wave_type: str
        choose Rossby waves or WIG waves or EIG waves.
        Defualt: Rossby
    parameters: dict
        planetary parameters dict with keys:
            angular_frequency: float, (rad/sec)
            gravitational_acceleration: float, (m/sec^2)
            mean_radius: float, (m)
            layer_mean_depth: float, (m)
        Defualt: Earth's parameters defined above

    Returns
    -------
    omega : Float, scalar
            wave frequency in 1/sec

    Notes
    -----
    This function supports k>=1 and n>=1 inputs only.
    Special treatments are required for k=0 and n=-1,0/-.

    """
    import numpy as np
    # make sure k and n are scalers:
    if not np.isscalar(n):
        raise TypeError('n must be scalar')
    if not np.isscalar(k):
        raise TypeError('k must be scalar')
    # make sure input is integer:
    if not isinstance(k, int):
        raise TypeError('k should be integer, i.e., ' + int(k))
        k = int(k)
    if not isinstance(n, int):
        raise TypeError('n should be integer, i.e., ' + int(n))
        k = int(k)
    # check for k>=1 and n>=1
    if k < 1:
        raise ValueError('pymaws supports only k>=1 for now...')
    if n < 1:
        raise ValueError('pymaws supports only n>=1 for now...')
    # unpack dictionary into vars:
    OMEGA = parameters.get('angular_frequency', 'missing angular_frequency key\
                           from parameters dictionary...')
    G = parameters.get('gravitational_acceleration', 'missing \
                        gravitational_acceleration key from parameters\
                        dictionary...')
    A = parameters.get('mean_radius', 'missing mean_radius key\
                        from parameters dictionary...')
    H0 = parameters.get('layer_mean_depth', 'missing layer_mean_depth key\
                        from parameters dictionary...')
    # cubic root estimation ?
    omegaj = np.zeros((1, 3))
    delta0 = 3. * (G * H0 * (k / A)**2 + 2. * OMEGA *
                   (G * H0)**0.5 / A * (2 * n + 1))
    delta4 = -54. * OMEGA * G * H0 * k / A**2
    # estimate cubic root o
    for j in range(1, 4):
        deltaj = (delta4**2 - 4. * delta0**3 + 0. * 1j)**0.5
        deltaj = (0.5 * (delta4 + deltaj))**(1. / 3.)
        deltaj = deltaj * np.exp(2. * np.pi * 1j * j / 3.)
        omegaj[0, j - 1] = np.real(-1. / 3. * (deltaj + delta0 / deltaj))
    # choose wave-type:
    if wave_type == 'Rossby':
        omega = -np.min(np.abs(omegaj))
    elif wave_type == 'WIG':
        omega = np.min(omegaj)
    elif wave_type == 'EIG':
        omega = np.max(omegaj)

    return omega


def eval_hermite_polynomial(x, n):
    """
    Evaluates the normalized Hermite polynomial of degree n at point/s x
    using the three-term recurrence relation.

    Parameters
    ----------
    x : Float, array_like or scalar
        list or array of points where the evalution takes place.
    n : Integer, scaler
        degree of the Hermite polynomial.

    Returns
    -------
    H_n : Float, array_like or scalar
            Evaluation of the normalized Hermite polynomail.

    Notes
    -----

    """
    import numpy as np
    # make sure n is integer and scaler:
    if not np.isscalar(n):
        raise TypeError('n must be scalar')
    if not isinstance(n, int):
        raise TypeError('n should be integer, i.e., ' + int(n))
    # make sure x is an array(or scalar):
    x = np.asarray(x)
    # main evaluation:
    if n < 0:
        H_n = np.zeros(x.shape)
    elif n == 0:
        H_n = np.ones(x.shape) / np.pi**0.25
    elif n == 1:
        H_n = (4.0 / np.pi)**0.25 * x
    elif n >= 2:
        H_n = ((2.0 / n)**0.5 * x * eval_hermite_polynomial(x, n - 1) -
               ((n - 1) / n)**0.5 * eval_hermite_polynomial(x, n - 2))
    return H_n


def eval_meridional_velocity(lat, n, amp, EPSILON):
    """
    Evaluates the meridional velocity amplitude at a given latitude point and
    a given wave-amplitude.

    Parameters
    ----------
    lat : Float, array_like or scalar
          latitude(radians)
    n : Integer, scaler
        wave-mode (dimensionless)
    amp : Float, scalar
          wave amplitude(m/sec)
    EPSILON: Float, scalar
            Lamb's parameter.
    Returns
    -------
    vel : Float, array_like or scalar
          Evaluation of the meridional velocity.

    Notes
    -----
    This function supports n>=1 inputs only.
    Special treatments are required for n=-1,0/-.

    """
    import numpy as np
    if not np.isscalar(amp):
        raise TypeError('amp must be scalar')
    # re-scale latitude
    y = EPSILON**0.25 * lat

    # Gaussian envelope
    ex = np.exp(-0.5 * y**2)

    vel = amp * ex * eval_hermite_polynomial(y, n)

    return vel


def eval_field_amplitudes(lat, k, n, amp, field='phi', wave_type='Rossby',
                          parameters=Earth):
    """
    Evaluates the latitude dependent amplitudes at a given latitude point.

    Parameters
    ----------
    lat : Float, array_like or scalar
          latitude(radians)
    k : Integer, scalar
    n : Integer, scaler
        wave-mode (dimensionless)
    amp : Float, scalar
          wave amplitude(m/sec)
    field : str
            pick 'phi' for geopotential height,
            'u' for zonal velocity and 'v' for meridional velocity
            Defualt : 'phi'
    wave_type: str
        choose Rossby waves or WIG waves or EIG waves.
        Defualt: Rossby
    parameters: dict
        planetary parameters dict with keys:
            angular_frequency: float, (rad/sec)
            gravitational_acceleration: float, (m/sec^2)
            mean_radius: float, (m)
            layer_mean_depth: float, (m)
        Defualt: Earth's parameters defined above
    Returns
    -------
    Either u_hat(m/sec), v_hat(m/sec) or p_hat(m^2/sec^2) : Float, array_like
    or scalar Evaluation of the amplitudes for the zonal velocity,
    or meridional velocity or the geopotential height respectivly.

    Notes
    -----
    This function supports k>=1 and n>=1 inputs only.
    Special treatments are required for k=0 and n=-1,0/-.

    """
    # unpack dictionary into vars:
    OMEGA = parameters.get('angular_frequency', 'missing angular_frequency key\
                           from parameters dictionary...')
    G = parameters.get('gravitational_acceleration', 'missing \
                        gravitational_acceleration key from parameters\
                        dictionary...')
    A = parameters.get('mean_radius', 'missing mean_radius key\
                        from parameters dictionary...')
    H0 = parameters.get('layer_mean_depth', 'missing layer_mean_depth key\
                        from parameters dictionary...')
    # Lamb's parameter:
    EPSILON = (2. * OMEGA * A)**2 / (G * H0)
    # evaluate wave frequency:
    omega = eval_omega(k, n, wave_type, parameters)
    # evaluate the meridional velocity amp first:
    v_hat = eval_meridional_velocity(lat, n, amp)
    # evaluate functions for u and phi:
    v_hat_plus_1 = eval_meridional_velocity(lat, n + 1, amp)
    v_hat_minus_1 = eval_meridional_velocity(lat, n - 1, amp)
    if field == 'v':
        return v_hat
    elif field == 'u':
        u_hat = (- ((n + 1) / 2.0)**0.5 * (omega / (G * H0)**0.5 + k / A) * v_hat_plus_1 -
                 ((n) / 2.0)**0.5 * (omega / (G * H0)**0.5 - k / A) * v_hat_minus_1)
        # pre-factors
        u_hat = G * H0 * EPSILON**0.25 / \
            (1j * A * (omega**2 - G * H0 * (k / A)**2)) * u_hat
        return u_hat
    elif field == 'phi':
        p_hat = (- ((n + 1) / 2.0)**0.5 * (omega + (G * H0)**0.5 * k / A) * v_hat_plus_1
                 + ((n) / 2.0)**0.5 * (omega - (G * H0)**0.5 * k / A) * v_hat_minus_1)
        p_hat = G * H0 * EPSILON**0.25 / \
            (1j * A * (omega**2 - G * H0 * (k / A)**2)) * p_hat
        return p_hat
    else:
        raise KeyError('field must be u, v or phi')


def eval_field(lat, lon, time, k, n, amp, field='phi', wave_type='Rossby',
               parameters=Earth):
    """
    Evaluates the analytic solutions of either the zonal or meridional velocity
    or the geopotential height on an on arbitrary lat x lon grids at times
    time.

    Parameters
    ----------
    lat : Float, array_like or scalar
          latitude(radians)
    lon : Float, array_like or scalar
          longitude(radians)
    time : Float, array_like or scaler
           time(sec), should be scalar and =0 if one wants only initial
           conditions.
    k : Integer, scalar
    n : Integer, scaler
        wave-mode (dimensionless)
    amp : Float, scalar
          wave amplitude(m/sec)
    field : str
            pick 'phi' for geopotential height,
            'u' for zonal velocity and v for meridional velocity
            Defualt : 'phi'
    wave_type: str
        choose Rossby waves or WIG waves or EIG waves.
        Defualt: Rossby
    parameters: dict
        planetary parameters dict with keys:
            angular_frequency: float, (rad/sec)
            gravitational_acceleration: float, (m/sec^2)
            mean_radius: float, (m)
            layer_mean_depth: float, (m)
        Defualt: Earth's parameters defined above
    Returns
    -------
    f : Float, 3D array (time, lon, lat)
        Evaluation of the the zonal velocity(m/sec),
        or meridional velocity(m/sec) or the geopotential height(m^2/sec^2)
        respectivly.

    Notes
    -----
    This function supports k>=1 and n>=1 inputs only.
    Special treatments are required for k=0 and n=-1,0/-.

    """
    import numpy as np
    # check for ndim :
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    time = np.atleast_1d(time)

    if lat.ndim > 1:
        raise ValueError('lat must be 1D array or scalar only!')
    if lon.ndim > 1:
        raise ValueError('lon must be 1D array or scalar only!')
    if time.ndim > 1:
        raise ValueError('time must be 1D array or scalar only!')
    # number of grid points
    nj = lat.shape[0]
    ni = lon.shape[0]
    nt = time.shape[0]

    # preallocate
    f = np.zeros((nt, ni, nj))

    # frequency
    omega = eval_omega(k, n, wave_type, parameters)

    # latitude-dependent amplitudes
    if field == 'phi':
        f_hat = eval_field_amplitudes(lat, k, n, amp, 'phi', wave_type,
                                      parameters)
    elif field == 'u':
        f_hat = eval_field_amplitudes(lat, k, n, amp, 'u', wave_type,
                                      parameters)
    elif field == 'v':
        f_hat = eval_field_amplitudes(lat, k, n, amp, 'v', wave_type,
                                      parameters)

    # adding time and longitude dependence
    for t in range(nt):
        f[t, :, :] = np.real(
            np.outer(np.exp(1j * (k * lon - omega * time[t])), f_hat))
    return f