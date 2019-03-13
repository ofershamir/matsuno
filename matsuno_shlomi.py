"""
get_fields(lat, lon, time, k, n, amp, wave_type) - evaluates the analytic
    solutions for the proposed test case on arbitrary lat x lon grids.

Inputs:
    lat        - 1D array of desired latitudes (radians)
    lon        - 1D array of desired longitudes (radians)
    time       - 1D array of desired times (sec)(array even for a single time)
    k          - spherical wave-number (dimensionless)
    n          - wave-mode (dimensionless)
    amp        - wave-amplitude (m/sec)
    wave_type  - string of wave-type: 'WIG', 'Rossby', or 'EIG'

Outouts:
    u          - 3D array (time, lon, lat): zonal velocity (m/sec)
    v          - 3D array (time, lon, lat): meridional velocity (m/sec)
    h          - 3D array (time, lon, lat): free-surface height anomaly (m)
    Phi        - 3D array (time, lon, lat): geopotential height (m^2/sec^2)

get_omega(k, n, wave_type) - evaluates the wave-frequency.

Inputs:
    k          - spherical wave-number (dimensionless)
    n          - wave-mode (dimensionless)
    wave_type  - string of wave-type: 'WIG', 'Rossby', or 'EIG'

Outouts:
    omega      - wave-frequency (rad/sec)

*This code is only valid for wave-numbers k>=1 and wave-modes n>=1.
Special treatments are required for k=0 and n=-1,0/-.*
"""

import numpy as np

# parameters
OMEGA = 7.29212e-5                # angular frequency (rad/sec)
G = 9.80616                       # gravitational acceleration (m/sec^2)
A = 6371220.                      # mean radius (m)
H0 = 0.5                          # Layer's mean depth (m)
EPSILON = (2. * OMEGA * A)**2 / (G * H0)  # Lamb's parameter


def get_omega(k, n, wave_type):
    """Eevaluates the wave-frequency."""

    omegaj = np.zeros((1, 3))
    delta0 = 3. * (G * H0 * (k / A)**2 + 2. * OMEGA *
                   (G * H0)**0.5 / A * (2 * n + 1))
    delta4 = -54. * OMEGA * G * H0 * k / A**2

    for j in range(1, 4):
        deltaj = (delta4**2 - 4. * delta0**3 + 0. * 1j)**0.5
        deltaj = (0.5 * (delta4 + deltaj))**(1. / 3.)
        deltaj = deltaj * np.exp(2. * np.pi * 1j * j / 3.)
        omegaj[0, j - 1] = np.real(-1. / 3. * (deltaj + delta0 / deltaj))

    if wave_type == 'Rossby':
        omega = -np.min(np.abs(omegaj))
    elif wave_type == 'WIG':
        omega = np.min(omegaj)
    elif wave_type == 'EIG':
        omega = np.max(omegaj)

    return omega


def get_hermite_polynomial(x, n):
    """Evaluates the normalized Hermite polynomial of degree n using the
       three-term recurrence relation."""

    if n < 0:
        H_n = np.zeros(x.shape)
    elif n == 0:
        H_n = np.ones(x.shape) / np.pi**0.25
    elif n == 1:
        H_n = (4.0 / np.pi)**0.25 * x
    elif n >= 2:
        H_n = ((2.0 / n)**0.5 * x * get_hermite_polynomial(x, n - 1) -
               ((n - 1) / n)**0.5 * get_hermite_polynomial(x, n - 2))

    return H_n


def get_psi(lat, n, amp):
    """Evaluates the eigenfunction psi."""

    # re-scale latitude
    y = EPSILON**0.25 * lat

    # Gaussian envelope
    ex = np.exp(-0.5 * y**2)

    psi_n = amp * ex * get_hermite_polynomial(y, n)

    return psi_n


def get_amplitudes(lat, k, n, amp, wave_type):
    """Evaluates the latitude dependent amplitudes."""

    omega = get_omega(k, n, wave_type)

    psi_n = get_psi(lat, n, amp)
    psi_n_plus_1 = get_psi(lat, n + 1, amp)
    psi_n_minus_1 = get_psi(lat, n - 1, amp)

    v_hat = psi_n

    u_hat = (- ((n + 1) / 2.0)**0.5 * (omega / (G * H0)**0.5 + k / A) * psi_n_plus_1
             - ((n) / 2.0)**0.5 * (omega / (G * H0)**0.5 - k / A) * psi_n_minus_1)

    p_hat = (- ((n + 1) / 2.0)**0.5 * (omega + (G * H0)**0.5 * k / A) * psi_n_plus_1
             + ((n) / 2.0)**0.5 * (omega - (G * H0)**0.5 * k / A) * psi_n_minus_1)

    # pre-factors
    u_hat = G * H0 * EPSILON**0.25 / \
        (1j * A * (omega**2 - G * H0 * (k / A)**2)) * u_hat
    p_hat = G * H0 * EPSILON**0.25 / \
        (1j * A * (omega**2 - G * H0 * (k / A)**2)) * p_hat

    return u_hat, v_hat, p_hat


def get_fields(lat, lon, time, k, n, amp, wave_type):
    """Evaluates the fields."""

    # number of grid points
    nj = lat.shape[0]
    ni = lon.shape[0]
    nt = time.shape[0]

    # preallocate
    u = np.zeros((nt, ni, nj))
    v = np.zeros((nt, ni, nj))
    Phi = np.zeros((nt, ni, nj))

    # frequency
    omega = get_omega(k, n, wave_type)

    # latitude-dependent amplitudes
    u_hat, v_hat, p_hat = get_amplitudes(lat, k, n, amp, wave_type)

    # adding time and longitude dependence
    for t in range(nt):
        u[t, :, :] = np.real(
            np.outer(np.exp(1j * (k * lon - omega * time[t])), u_hat))
        v[t, :, :] = np.real(
            np.outer(np.exp(1j * (k * lon - omega * time[t])), v_hat))
        Phi[t, :, :] = np.real(
            np.outer(np.exp(1j * (k * lon - omega * time[t])), p_hat))

    # transform to free-surface height anomaly
    h = Phi / G

    return u, v, h, Phi
