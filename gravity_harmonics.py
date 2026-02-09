#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:48:00 2023

@author: linzifan

An extension of the simple mass-radius model assuming non-rotating, spherically symmetric planet.
Assumes uniform rotation and calculates shape and gravitational moments J_2n self-consistently following
the CMS method (Hubbard 2013; Wisdom & Hubbard 2016; Militzer et al. 2019).
    
All units are in SI or in dimensionless planetary unit.
"""

import math
import numpy as np
from scipy.special import legendre
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import datetime
from tqdm import tqdm
import EOS as eos
import simpleMR as smr
import sys
np.set_printoptions(threshold=sys.maxsize) # print the full numpy array without truncation


# ====== Helper functions handy to have ====================================================================
def list_rindex(li, x):
    for i in reversed(range(len(li))):
        if li[i] == x:
            return i
    raise ValueError("{} is not in list".format(x))


def legendre_mu(n, mu):
    """
    Given degree n of Legendre polynomial and mu = cos(theta), calculate and return the Pn(mu) value.
    """
    legendre_poly = legendre(n) # this contains polynomial prefactors
    return legendre_poly(mu)
    

# ====== CMS ================================================================================================
# ====== Gravitational harmonics functions - implement equation 16 to 23 in Militzer et al. (2019) ==========
def J2n_gravity_harmonics(lambdai_arr, Jin_arr, n):
    """
    Calculates zonal gravity harmonics observable at the surface of the planet. Implements Eq. 23 in 
    Militzer et al. (2019).
    
    Inputs:
    - lambdai_arr is an array with N_L elements. i ranges from 0 to N_L - 1
    - Jin_arr is an array of interior gravity harmonics J_in; each element of Jin_arr is also an array representing
    gravitational harmonics (J0, J2, J4, ...) at layer i
    - n is an even number not greater than (len(Jin_arr[0])-1) * 2
    """
    assert len(lambdai_arr) == len(Jin_arr)
    assert n % 2 == 0 # n is an even integer
    assert n <= (len(Jin_arr[0])-1) * 2
    
    Jn_arr = Jin_arr[:, n] # each element of Jin_arr contains (J0, J2, J4, ...), this step extract all Jn from all i layers as an array
    return np.sum(lambdai_arr**n * Jn_arr) # Eq. 23


def Jin_interior_harmonics(n, M, delta_i, lambda_i, zeta_im_arr, mu_m_arr, quadw_arr):
    """
    Implements Eq. 16 in Militzer et al. (2019).
    
    Inputs:
    - n is layer index between 0 (outermost spheriod) and N_L - 1 (innermost spheriod)
    - M is the total mass of the planet given by Eq. 20
    - delta_i is the density difference between two layers
    - lambda_i is r_i(0)/r_0(0), the ratio of equatorial radius
    - zeta_im_arr is an array of zeta at level i evaluated at mu_m angles, such that zeta_i,m = zeta_im_arr[m];
    its length is N_m
    - mu_m_arr is an array of N_m mu angles; the integral of dmu is approximated with a summation of delta mu; note
    that mu_m_arr ranges between 0 and +1 (hemisphere only due to symmetry), and in the sum we need to extend 
    mu_m_arr to -1 to +1
    - quadw_arr is the array of quadrature weights
    
    Returns: normalized interior harmonics J_i,n
    """
    assert len(zeta_im_arr) == len(mu_m_arr)
    assert n % 2 == 0 # n is an even integer
    assert len(mu_m_arr) == len(quadw_arr)
    
    Jin = -1.0/(n+3.0)
    Jin *= 2.0 * np.pi / M 
    Jin *= delta_i * lambda_i**3.0
    
    # now we have everything to calculate the summation
    summation = quadw_arr[0] * legendre_mu(n, mu_m_arr[0]) * zeta_im_arr[0]**(n+3) # special case when mu=0, i.e., at the equator
    for m in range(1, len(mu_m_arr)): # sum from m=1 to m=N_M-1; multiply by 2 due to north-south symmetry
        summation += 2 * quadw_arr[m] * legendre_mu(n, mu_m_arr[m]) * zeta_im_arr[m]**(n+3)
    Jin *= summation
    # func = legendre_mu(n, mu_m_arr) * zeta_im_arr**(n+3)
    # integral = 2 * np.trapz(func, x=mu_m_arr)
    # Jin *= integral
    
    # print('84', type(Jin), Jin)
    return Jin
    
    
def Jin_p_exterior_harmonics(n, M, delta_i, lambda_i, zeta_im_arr, mu_m_arr, quadw_arr):
    """
    Implements Eq. 17 in Militzer et al. (2019).
    
    Inputs: The same inputs as Jin_interior_harmonics()
    
    Returns: Normalized exterior harmonics J'_i,n. Note that there is a special case for n=2 (Eq. 18).
    """
    assert len(zeta_im_arr) == len(mu_m_arr)
    assert n % 2 == 0 # n is an even integer
    assert len(mu_m_arr) == len(quadw_arr)
    
    if n == 2: # use Eq. 18 for n=2
        Jinp = -2 * np.pi / M 
        Jinp *= delta_i * lambda_i**3.0
        
        summation = quadw_arr[0] * legendre_mu(n, mu_m_arr[0]) * np.log(zeta_im_arr[0]) # special case when mu=0
        for m in range(1, len(mu_m_arr)): # sum from m=1 to m=N_M-1; multiply by 2 due to north-south symmetry
            summation += 2 * quadw_arr[m] * legendre_mu(n, mu_m_arr[m]) * np.log(zeta_im_arr[m])
        Jinp *= summation
        # func = legendre_mu(n, mu_m_arr) * np.log(zeta_im_arr)
        # integral = 2 * np.trapz(func, x=mu_m_arr)
        # Jinp *= integral 
    else: # use Eq. 17 for n != 2
        Jinp = -1/(2-n)
        Jinp *= 2 * np.pi / M 
        Jinp *= delta_i * lambda_i**3.0
        
        summation = quadw_arr[0] * legendre_mu(n, mu_m_arr[0]) * zeta_im_arr[0]**(2-n) # special case when mu=0
        for m in range(1, len(mu_m_arr)): # sum from m=1 to m=N_M-1; multiply by 2 due to north-south symmetry
            summation += 2 * quadw_arr[m] * legendre_mu(n, mu_m_arr[m]) * zeta_im_arr[m]**(2-n)
        Jinp *= summation
        # func = legendre_mu(n, mu_m_arr) * zeta_im_arr**(2-n)
        # integral = 2 * np.trapz(func, x=mu_m_arr)
        # Jinp *= integral
        
    return Jinp


def Ji0_pp(delta_i, M, Req=1.0):
    """
    Implements Eq. 19 in Militzer et al. (2019).
    
    Inputs:
    - the same delta_i and M as in Jin_interior_harmonics()
    - M is the (dimensionless) mass of the planet
    - Req is the equitorial radius of the planet (normalized to 1)
    
    Returns: Normalized exterior harmonics J'_i,n. Note that there is a special case for n=2 (Eq. 18).
    """
    return (2.0 * np.pi * delta_i * Req**3.0) / (3.0 * M)


def M_planet_mass(delta_i, lambda_i, zeta_2darr, mu_arr, quadw_arr):
    """
    Implements Eq. 20 in Militzer et al. (2019).
    
    Inputs:
    - delta_i is the density difference between two layers
    - lambda_i is r_i(0)/r_0(0), the ratio of equatorial radius
    - zeta_2darr is a 2D array with N_L * N_m elements; zeta_im is represented by element zeta_2darr[i, m]
    - mu_arr is a 1D array of angles mu_m; note that for each layer i, the angles mu_m should be the same, so there
    is no need for mu_i,m
    
    Returns: Total mass of the planet M (unitless, given that delta_i and lambda_i are unitless density and length).
    """
    assert len(delta_i) == len(lambda_i)
    assert len(delta_i) == len(zeta_2darr)
    assert len(mu_arr) == len(quadw_arr)
    
    M = 2 * np.pi / 3.0
    
    sum_i = 0.0
    for i in range(len(delta_i)): # sum from 0 to N_L-1
        sum_m = quadw_arr[0] * zeta_2darr[i, 0]**3.0
        for m in range(1, len(mu_arr)): # sum from 1 to N_M-1
            sum_m += 2 * quadw_arr[m] * zeta_2darr[i, m]**3.0
        # func = zeta_2darr[i]**3.0
        # integral = np.trapz(func, x=mu_arr) # integral of dmu zeta(mu)**3.0
        sum_i += delta_i[i] * lambda_i[i]**3.0 * sum_m
    M *= sum_i
    
    # print(type(M), M)
    return M


def Vim_gravity_potential(zeta_2darr, mu_arr, lambda_arr, delta_arr, Jin, Jinp, Jinpp, i, m, N_L, n_max=33,
                          zeta_im=None):
    """
    Implements Eq. 21 in Militzer et al. (2019). Calculates the expansion of the exact expression for gravitational
    potential V_i on spheriod i and angle mu.
    
    Inputs:
    - zeta_2darr is a 2D array with N_L * N_m elements; zeta_im is represented by element zeta_2darr[i, m]
    - mu_arr is a 1D array such that mu_i,m is represented by mu_arr[m]; the angle mu grid is the same for all layers
    - lambda_arr is a 1D array such that lambda_i = lambda_arr[i]
    - delta_arr is a 1D array of density differences between layers, such that delta_i = delta_arr[i]
    - Jin, Jinp, and Jinpp are interior and exterior gravity harmonics calculated from Eq. 16-19; they should be
    2D array with length N_L and each element should have length n_max/2 + 1, such that J_i,n = Jin[i, n];
    note that J''_{i,0} only works for mu=0, so is a 1D array
    - i is the layer index
    - m is the angle index
    - M is the planetary mass
    - N_L is the total number of layers; i ranges from 0 to N_L - 1
    - n_max is the maximum polynomial order, typically 16 or 32 (set n_max=33 so that J32 is also included)
    - zeta_im: instead of inputting an array, one can choose to input zeta_im value as a float directly. This works
    better for scipy.optimize.fsolve.
    
    Returns: Gravitational potential Vi at a point on surface i (unitless).
    
    Note that there is a special case for i=0, zeta_i=1, and mu=0 (potential on the equator of the outermost
    spheroid).
    """
    # all should have length = N_L
    assert len(zeta_2darr) == len(lambda_arr)
    assert len(zeta_2darr) == len(delta_arr)
    assert len(zeta_2darr) == len(Jin)
    assert len(zeta_2darr) == len(Jinp)
    assert len(zeta_2darr) == len(Jinpp)
    assert len(zeta_2darr) == N_L
    
    if zeta_im == None:
        zeta_im = zeta_2darr[i, m]
    
    if i == 0 and m == 0: # special case for V_im = V_00, the gravitational potential on the equator at the surface
        v00 = 0.0
        # calculate v00 from the sum in equation 22
        for n in range(n_max):
            if n % 2 == 0: # only sum even Jn harmonics
                # calculate J_n using equation 23
                # Jn_step = 0.0
                # for j in range(N_L):
                #     Jn_step += lambda_arr[j]**n * Jin[j, n]
                Jn_step = J2n_gravity_harmonics(lambda_arr, Jin, n)
                v00 += legendre_mu(n, 0.0) * Jn_step
        return -v00
    
    else: # all other cases except for V(1, 0)
        a = -1 / (zeta_im * lambda_arr[i])
        
        b = 0.0
        c = 0.0
        d = 0.0
        # the sum for J_{i,n} goes from j=i to j=N_L-1
        for j in range(i, N_L):
            for n in range(n_max):
                if n % 2 == 0: # only sum even Jn harmonics
                    # calculate first sum involving J_{i,n}
                    b += Jin[j, n] * (lambda_arr[j] / (lambda_arr[i]*zeta_im))**n * legendre_mu(n, mu_arr[m])
        
        # the sum for J'_{i,n} and J''_{i,0} go from j=0 to j=i-1
        for j in range(0, i):
            for n in range(n_max):
                if n % 2 == 0: # only sum even Jn harmonics
                    # calculate second sum involving J'_{i,n}
                    c += Jinp[j, n] * ((lambda_arr[i]*zeta_im)/lambda_arr[j])**(n+1) * legendre_mu(n, mu_arr[m])
                    
            # calculate third sum involving J''_{i,0}
            d += Jinpp[j] * lambda_arr[i]**3.0 * zeta_im**3.0
        
        # print(type(a), type(b), type(c), type(d))
        # print("249", i, m, a, b, c, d)
        return a * (b + c + d)


def Qim_centrifugal_potential(r_im, mu_m, omega, Req_si, M_si, r_unit):
    """
    Calculate Qi potential assuming uniform rotation.
    $Q_{im} = \frac{1}{2} r_i^2(\mu_m) \omega^2 (1-\mu_{im}^2)$
    """
    if r_unit == "SI": # SI units
        Q = 0.5 * r_im**2.0 * omega**2.0 * (1 - mu_m**2.0) # 1/2 r^2 omega^2 sin^2(theta)
        # print(type(Q), Q)
        return Q * (Req_si / (smr.G * M_si)) # convert potential Q in m2 s-2 into dimensionless Q
    
    elif r_unit == "PU": # planetary units
        Q = 0.5 * r_im**2.0 * omega**2.0 * (1 - mu_m**2.0) # 1/2 r^2 omega^2 sin^2(theta)
        return Q * (Req_si**3.0 / (smr.G * M_si)) # convert into dimensionless Q; r_pu introduces another 1/Req**2.0 factor


def Qim_centrifugal_potential_zeta(zeta_im, r_i0, mu_m, omega, Req_si, M_si, r_unit):
    """ The same as above, but uses zeta_im as the free variable. 
    Note that zeta_i(mu) = r_i(mu) / r_i(0).
    """
    r_im = zeta_im * r_i0

    if r_unit == "SI": # SI units
        Q = 0.5 * r_im**2.0 * omega**2.0 * (1 - mu_m**2.0) # 1/2 r^2 omega^2 sin^2(theta)
        # print(type(Q), Q)
        return Q * (Req_si / (smr.G * M_si)) # convert potential Q in m2 s-2 into dimensionless Q
    
    elif r_unit == "PU": # planetary units
        Q = 0.5 * r_im**2.0 * omega**2.0 * (1 - mu_m**2.0) # 1/2 r^2 omega^2 sin^2(theta)
        return Q * (Req_si**3.0 / (smr.G * M_si)) # convert into dimensionless Q; r_pu introduces another 1/Req**2.0 factor


def f_im_derivative(zeta_2darr, mu_arr, lambda_arr, delta_arr, Jin, Jinp, Jinpp, i, m, M, N_L, 
                    arr_r_i0, omega, n_max=33):
    """
    Analytically calculate f'im(zeta_im) = d f_im(zeta_im) / d zeta_im, which is the derivative of the potential
    difference Ui(zeta_im, mu_m) - Ui(1, 0) (Eq. 7 in Militzer et al. 2019).
    
    Inputs:
    - most of the inputs are the same as in Vim_gravity_potential()
    - r_2darr is a 2D array such that r_{i,m} = r_2darr[i, m]
    - omega is the angular velocity of the planet's rotation
    
    The derivative has two parts, dQ/dzeta and dV/dzeta.
    """
    # all should have length = N_L
    assert len(zeta_2darr) == len(lambda_arr)
    assert len(zeta_2darr) == len(delta_arr)
    assert len(zeta_2darr) == len(Jin)
    assert len(zeta_2darr) == len(Jinp)
    assert len(zeta_2darr) == len(Jinpp)
    assert len(zeta_2darr) == len(arr_r_i0)
    assert len(zeta_2darr) == N_L
    
    def dVim(zeta_2darr, mu_arr, lambda_arr, delta_arr, Jin, Jinp, Jinpp, i, m, M, N_L, n_max):
        a = 1 / (zeta_2darr[i, m]**2.0 * lambda_arr[i])
        
        b = 0.0
        c = 0.0
        d = 0.0
        # the sum for J_{i,n} (interior harmonics) goes from j=i to j=N_L-1
        for j in range(i, N_L):
            for n in range(n_max):
                if n % 2 == 0: # only sum even Jn harmonics
                    # calculate first sum involving J_{i,n}
                    b += (n+1) * Jin[j, n] * (lambda_arr[j] / (lambda_arr[i]*zeta_2darr[i, m]))**n * legendre_mu(n, mu_arr[m])
        
        # the sum for J'_{i,n} and J''_{i,0} (exterior harmonics) go from j=0 to j=i-1
        for j in range(0, i):
            for n in range(n_max):
                if n % 2 == 0: # only sum even Jn harmonics
                    # calculate second sum involving J'_{i,n}
                    c += -n * Jinp[j, n] * ((lambda_arr[i]*zeta_2darr[i, m])/lambda_arr[j])**(n+1) * legendre_mu(n, mu_arr[m])
                    
            # calculate third sum involving J''_{i,0}
            d += -2 * Jinpp[j] * lambda_arr[i]**3.0 * zeta_2darr[i, m]**3.0
            
        return a * (b + c + d)
    
    
    def dQim(r_i0, zeta_2darr, mu_arr, omega, i, m):
        return r_i0**2.0 * omega**2.0 * (1 - mu_arr[m]**2.0) * zeta_2darr[i, m]
    
    
    dV_dzeta = dVim(zeta_2darr, mu_arr, lambda_arr, delta_arr, Jin, Jinp, Jinpp, i, m, M, N_L, n_max)
    dQ_dzeta = dQim(arr_r_i0[i], zeta_2darr, mu_arr, omega, i, m)
    # if i == 10:
    #     print("325", dV_dzeta, dQ_dzeta)
    return dV_dzeta + dQ_dzeta


def cms_eos(p, t, layer_name, rho_p_func=None, use_rho_p_func=False):
    """ Given pressure p (in Pa), temperature t (in K), and layer_name (string, 'core', 'mantle', 'water', or 'H/He'),
    return the density using physical EOS from EOS.py 

    When running rock giant model, because we don't have a tabulated EOS in hand, we need to pass a function 
    rho_p_func(), which takes pressure as input and returns a density, into this function. This function comes
    from e.g., interpolating rho-P profile.

    use_rho_p_func=True will force the function to use rho_p_func, instead of calling EOS functions
    """
    if use_rho_p_func:
        return rho_p_func(p)
    else:
        if layer_name == "core":
            return eos.eos_tabulated(p, "Zeng2021_core")
        elif layer_name == "mantle":
            return eos.eos_tabulated(p, "Zeng2021_mantle")
        elif layer_name == "water":
            return eos.eos_AQUA(p, t)[0]
        elif layer_name == "H/He":
            return eos.eos_cd21_hhe(p, t)[0]
        elif layer_name == "custom_rho_p":
            return rho_p_func(p)
        else:
            raise ValueError("Invalid layer name! Only core, mantle, water, H/He are allowed.")


# ====== Main iteration functions ===========================================================================
def initialize_planet_array(forward_output, layer_boundaries, layers, NL, r_core, omega, mu, arr_quadw,
                            n_max=33):
    """
    Given a text file containing outputs from my forward planetary interior model, set up an initial array
    assuming spherical (zeta=1 everywhere) planet.
    
    The array has length N_L (number of spheriods) with index from i = 0 to N_L - 1. Each subarray arr[i] has
    the following elements:
    - i (integer), index between 0 and N_L - 1
    - mu (array with N_m elements), values of cos(theta); due to N-S symmetry, defining for one hemisphere is enough
    - zeta_i (array with N_m elements), such that zeta_i[m] = zeta_{i, m}; initially, zeta = 1 everywhere
    - r_i0 (float), radius at the equator at spheriod i
    - r_i (array with N_m elements), such that r_i[m] = r_{i, m}; initially, all r_i = r_i0
    - rho_i (float), density at spheriod i
    - P_i (float), pressure at spheriod i
    - T_i (float), temperature at spheriod i; note that rho, P, T are equal everywhere on the spheriod
    - U_i (float), potential at spheriod i, U = V + Q
    - f_i (array with N_m elements), the difference between U_{i, m} and U_{i, 0}, such that f_i[m] = f_{i, m}
    - J_i (array with n_max elements), such that J_i[n] is the J_{i, n} interior harmonic; n_max is usually 16 or 32
    - Jp_i (array with n_max elements), such that Jp_i[n] is the J'_{i, n} exterior harmonic
    - Jpp_i (float), the J''_{i, 0} gravity harmonic at spheriod i
    - lambda_i (float), lambda_i = r_i0 / r_00 is the ratio of the equatorial radius of the ith to the outermost spheroid
    - delta_i (float), the density difference between layer i and layer i-1; special case: delta_0 = rho_0
    - li_layers (string), name of the layer ("core", "mantle", "water", or "H/He"), to decide which EOS to use
    
    The inputs are:
    - forward_output: a string, the filename of forward planetary model file
    - layer_boundaries: list of floats, radius at the layer boundaries (for differentiated models, can be arbitrary for continuous models)
    - layers: list of strings, layers included in this model; should include "core", "mantle", "water", and/or "H/He"
    - NL: number of spheriods
    - NM: number of angles (in each hemisphere); mu[0] corresponds to cos(0) and therefore should be 1 (deleted)
    - r_core: the (adjustable) radius of central core; Militzer et al. (2019) adjusts the mass of the central core
    to achieve balance because at least one free parameter is needed to obtain the correct total mass; here we adjust
    r_core, which should be equivalent to adjusting core mass.
    - mu: an array of cos(theta), given according to Gaussian quadrature tables
    - arr_quadw: Gaussian quadrature weights
    """
    # assert input parameters
    # assert NM % 2 == 1 # Is this necessary though? We are only calculating one hemisphere
    assert len(layer_boundaries) == len(layers)
    assert len(mu) == len(arr_quadw)
    # assert len(mu) == NM
    assert mu[0] == 0.0 # the first angle should be 0 (equator)
    
    # read in data from forward model results file
    forward_data = np.genfromtxt(forward_output, usecols=(0, 1, 2, 3, 4))
    # print(np.shape(forward_data))
    forward_data = np.transpose(forward_data)
    # print(np.shape(forward_data))
    forward_m = forward_data[0]#; forward_m.astype(float)
    forward_r = forward_data[1]#; forward_r.astype(float)
    forward_p = forward_data[2]#; forward_p.astype(float)
    forward_rho = forward_data[3]#; forward_rho.astype(float)
    forward_t = forward_data[4]#; forward_t.astype(float)
    forward_layer = np.genfromtxt(forward_output, usecols=(5), dtype="|U6")
    # forward_layer = forward_data[5]
    # print("429", forward_r, forward_layer)
    # print("430", type(forward_r[0]), type(forward_layer[0]))
    rcore = min(forward_r)
    Req_si = max(forward_r)
    M_pmass_si = max(forward_m)

    # Read in the densities at layer boundaries (outermost density of each layer), and make sure some layers
    # are located right at the boundaries.
    # First, find the index of the last occurence of layer name (e.g., "core")
    li_last_layer = []
    rho_boundary = []

    rho_interp = interp1d(forward_r, forward_rho)
    rho_interp_inverse = interp1d(forward_rho, forward_r)
    rho_core = float(rho_interp(r_core))
    
    if len(layers) > 1: # only run this part when there are 2 or more layers
        for i in range(len(layers)):
            last_index = list_rindex(forward_layer, layers[i])
            li_last_layer.append(last_index) # append index of last layer to li_last_layer
            if i < len(layers)-1:
                rho_boundary.append(forward_rho[last_index]*(1.0+1.0e-6)) # increase the density by a tiny bit, so when interpolating r(rho) for li_layers, we get the desired (inner) layer, not the layer just above it
            else:
                rho_boundary.append(forward_rho[last_index]) # for the outermost layer, don't change rho so that ri0 == 1.0
        rho_boundary.reverse()
        # now rho_boundary contains densities at the outer edge of each layer; pick layers between such that
        # the total number of layers is at least the same as NL; now pick layers from surface to core, excluding 
        # the surface layer, because it is in rho_boundary already
        arr_rho_si = np.ndarray.tolist(np.logspace(np.log10(rho_boundary[0]), np.log10(rho_core), NL-len(rho_boundary))[1:]) 
        arr_rho_si += rho_boundary
        arr_rho_si.sort()
    # else, there is no boundary within the planet, simply use np.logspace() to define the layers
    else:
        arr_rho_si = np.ndarray.tolist(np.logspace(np.log10(forward_rho[-1]*(1.0+1.0e-6)), np.log10(rho_core), NL)) # increase the density by a tiny bit, to avoid interpolation error
        arr_rho_si.sort()

    arr_rho_si = np.asarray(arr_rho_si)
    arr_rho_pu = arr_rho_si * (Req_si**3.0/forward_m[-1]) # scale by a^3 / M to convert to planetary unit

    # initialize indices i
    NL = len(arr_rho_si) # redefine NL in case of length difference
    arr_i = np.arange(0, NL)
    
    # # initialize mu = cos(theta), where theta is the polar angle, so on equator theta = pi/2
    # theta = list(np.linspace(np.pi/2, 0, NM))
    # mu = list(map(lambda ang: np.cos(ang), theta))
    # mu[0] = 0.0 # the first element should be cos(pi/2)=0; explicitly set to 0 to avoid small floats
    
    # initialize shape functions zeta_{i,m}; all zeta_{i,m} should be 1 (spherical) at the beginning
    zeta_each_layer = np.ones(len(mu))
    arr_zeta = np.repeat([zeta_each_layer], len(arr_i), axis=0)
    
    # Initialize r_i0, the radius at the equator, which should remain the same throughout the iteration
    # Note that according to Militzer+ 2019 and Militzer & Hubbard 2023, an ideal r_i0 array should make
    # density rho_{i+1} / rho_i = constant. This will reduce discretization error that is intrinsic to the
    # CMS method, and make convergence faster.

    # First, we need to initialize arr_rho
    # rho_interp = interp1d(forward_r, forward_rho)
    # rho_interp_inverse = interp1d(forward_rho, forward_r)
    # rho_core = rho_interp(r_core)
    # arr_rho_si = np.logspace(np.log10(min(forward_rho)), np.log10(rho_core), NL) # arr_rho[0] is the surface density
    # arr_rho_pu = arr_rho_si * (Req_si**3.0/forward_m[-1]) # scale by a^3 / M to convert to planetary unit

    # print("450", arr_rho_si)
    # print(max(forward_rho), min(forward_rho))
    # print("505", min(forward_rho), max(forward_rho))
    arr_r_i0 = rho_interp_inverse(arr_rho_si)
    # arr_r_i0 = np.linspace(r_core, max(forward_r), NL)
    # arr_r_i0 = np.flip(arr_r_i0) # note that i=0 corresponds to the planet's surface and i=NL to core
    # Req = arr_r_i0[0]
    # note that arr_r_i0 is still in unit of meters at this point
    arr_r_i0_mid = np.ones_like(arr_r_i0) # radius at the midpoints between layer boundaries
    for i in range(len(arr_r_i0) - 1):
        arr_r_i0_mid[i] = 0.5 * (arr_r_i0[i] + arr_r_i0[i+1])
    arr_r_i0_mid[len(arr_r_i0)-1] = rcore # for the innermost layer, midpoint radius is simply rcore
    
    # initialize r_{i,m}; all r_{i', m} should be equal to r_{i', 0} (spherical) at the beginning
    arr_r = np.repeat([np.ones(len(mu))], len(arr_i), axis=0) # initialize array with the right shape
    for i in range(len(arr_i)):
        arr_r[i] = np.ones_like(mu) * arr_r_i0[i] # change each subarray to the correct r_i0 value
        
    # # initialize rho_i; note that density of layer i should be the same everywhere
    # rho_interp = interp1d(forward_r, forward_rho)
    # arr_rho_si = rho_interp(arr_r_i0_mid)
    # arr_rho_pu = arr_rho_si * (Req**3.0/forward_m[-1]) # scale by a^3 / M to convert to planetary unit
    
    # initialize P_i; note that pressure of layer i should be the same everywhere
    # print("527", min(forward_r), max(forward_r), arr_rho_si, arr_r_i0)
    p_interp = interp1d(forward_r, forward_p)
    arr_p_si = p_interp(arr_r_i0)
    arr_p_pu = arr_p_si * (Req_si**4.0/(smr.G * forward_m[-1]**2.0)) # scale by a^4 / G M^2 to convert to planetary unit
    
    # initialize T_i; note that pressure of layer i should be the same everywhere; assume change in shape does not
    # cause change in T_i
    t_interp = interp1d(forward_r, forward_t)
    arr_t = t_interp(arr_r_i0)
    
    # normalize to planetary unit such that r_{1,0} = Req = 1
    # save this till here because the forward_r above is still in unit of meters
    arr_r /= Req_si
    arr_r_i0 /= Req_si 
    arr_r_i0_mid /= Req_si

    # initialize lambda_i = r_i0 / r_00; should start from 1.0 (surface) and decrease as we go towards to core
    arr_lambda = arr_r_i0 / max(arr_r_i0)
    
    # initialize delta_i, the density difference between layer i and layer i-1; special case: delta_0 = rho_0
    arr_delta = np.diff(arr_rho_pu) # delta_i = rho_i - rho_{i-1}
    arr_delta = np.insert(arr_delta, 0, arr_rho_pu[0]) # delta_0 = rho_0
    
    # initialize J_i & J'_i
    pmass_eq20 = M_planet_mass(arr_delta, arr_lambda, arr_zeta, mu, arr_quadw)
    # print("430", pmass_eq20)
    arr_Jin = np.repeat([np.zeros(n_max)], len(arr_i), axis=0) # arr_Jin has shape (NL, n_max)
    arr_Jinp = np.repeat([np.zeros(n_max)], len(arr_i), axis=0) # arr_Jinp has the same shape (NL, n_max)
    for i in range(len(arr_i)):
        for n in range(n_max):
            if n % 2 == 0: # only calculate the even n harmonics
                # Jin_debug = Jin_interior_harmonics(n, pmass_eq20, arr_delta[i], arr_lambda[i], arr_zeta[i], mu)
                # print(type(Jin_debug), Jin_debug)
                arr_Jin[i, n] = Jin_interior_harmonics(n, pmass_eq20, arr_delta[i], arr_lambda[i],
                                                       arr_zeta[i], mu, arr_quadw)
                # Jinp_debug = Jin_p_exterior_harmonics(n, pmass_eq20, arr_delta[i], arr_lambda[i], arr_zeta[i], mu)
                # print(type(Jinp_debug), Jinp_debug)
                arr_Jinp[i, n] = Jin_p_exterior_harmonics(n, pmass_eq20, arr_delta[i], arr_lambda[i],
                                                          arr_zeta[i], mu, arr_quadw)
    # note that Jin and Jinp is initialized as np.zeros, so all the odd harmonics would be 0
    
    # initialize J''_i,0
    arr_Jinpp = Ji0_pp(arr_delta, pmass_eq20, 1.0) # here use dimensionless Req=1
    
    # initialize U_{i, m}; U_{i, m} is not equal on a given spheriod at the beginning, and we need to iterate
    # until it is equal everywhere on all spheriod i
    # also initialize f_{i, m} = U_{i, m} - U_i(1, 0), the potential compared to a reference point
    arr_u = np.ones_like(arr_r)
    arr_f = np.ones_like(arr_r)
    for i in range(len(arr_u)):
        for m in range(len(arr_u[0])):
            vim = Vim_gravity_potential(arr_zeta, mu, arr_lambda, arr_delta,
                                        arr_Jin, arr_Jinp, arr_Jinpp, 
                                        i, m, NL, n_max)
            qim = Qim_centrifugal_potential(arr_r[i, m], mu[m], omega, Req_si, M_pmass_si, "PU")
            # print("459", i, m, vim, qim)
            arr_u[i, m] = vim + qim 
            if m != 0:
                arr_f[i, m] = vim + qim - arr_u[i, 0]
            else:
                arr_f[i, m] = 0.0    
    
    # initialize li_layers
    layer_boundaries = list(map(lambda r: r / Req_si, layer_boundaries)) # scale all to unitless radius
    li_layers = [""] * len(arr_i) # initialize li_layers with empty strings
    # print("520", arr_r_i0_mid, layer_boundaries)
    for i in range(len(arr_i)):
        if arr_r_i0_mid[i] <= layer_boundaries[0]: # below the innermost layer boundary (usually CMB)
            li_layers[i] = layers[0]
        for j in range(len(layer_boundaries)-1): # only consider the first three boundaries (the last boundary is planet surface)
            if arr_r_i0_mid[i] >= layer_boundaries[j] and arr_r_i0_mid[i] < layer_boundaries[j+1]:
                li_layers[i] = layers[j+1]
    # print("527", li_layers)

    # return all the initialized arrays/lists
    # note that arr_r_i0 is in planetary (unitless) units, while arr_r is in meters
    return arr_i, mu, arr_zeta, arr_r_i0, arr_r, arr_rho_si, arr_rho_pu, arr_p_si, arr_p_pu, arr_t, \
        arr_lambda, arr_delta, arr_Jin, arr_Jinp, arr_Jinpp, arr_u, arr_f, arr_quadw, li_layers


def step_planet_array(arr_i, mu, arr_zeta, arr_r_i0, arr_r, arr_rho_si, arr_rho_pu, arr_p_si, arr_p_pu,
                      arr_t, arr_lambda, arr_delta, arr_Jin, arr_Jinp, arr_Jinpp, arr_u, arr_f, arr_quadw,
                      li_layers, omega, Req_si, M_pmass_si, mode, del_zeta=1.0e-3, rho_p_func=None):
    """ Given a set of parameters (arr_i, mu, arr_zeta, ...), calculate one step of the CMS iteration to
    update the shape function, U, f, Jn, etc., and return a new set of parameters (arr_i_new, mu_new, 
    arr_zeta_new, ...). Note that some parameters like index i and angle mu are unchanged.

    del_zeta is the maximum step size in change in shape function zeta. By default, it is 0.1%. Smaller del_zeta
    should be used when closer to convergence.

    Note that this function has two modes, "initial" and "iterate"
    - Under "initial" mode, this should be the first step after initializing the planet (i.e. zeta should be 1
      everywhere when calling this function). For subsequent steps in the iteration, a different function should
      be called, because the first step employs a Newton step to reduce U difference (equation 7), but in later
      steps, fsolve should be used to solve for Vi + Qi = U_i(zeta=1,mu=0).
    - Under "iterate" mode, this should be 2+ steps in an iteration. The only difference is the method for
      calculating zeta. Under this mode, zeta is calculated by solving V + Q = U(1, 0) (see equation 51 in
      Hubbard 2013).
    """
    # assertion
    assert (mode == "initial" or mode == "iterate"), "Invalid mode. Use 'initial' or 'iterate'."

    # derive some useful parameters
    M_pmass = M_planet_mass(arr_delta, arr_lambda, arr_zeta, mu, arr_quadw)
    n_max = np.shape(arr_Jin)[1]
    N_L = len(arr_i)
    Req = arr_r_i0[0] # which is 1.0

    # unchanged parameters
    arr_i_new = arr_i # number of layers (and hence index of each layer) won't change
    mu_new = mu # angles won't change
    arr_quadw_new = arr_quadw # quadrature weights won't change
    arr_t_new = arr_t # the CMS model is temperature independent
    li_layers_new = li_layers # layer names won't change
    arr_r_i0_new = arr_r_i0 # it's assumed that the equatorial radii of every spheroid is fixed
    arr_lambda_new = arr_lambda # hence, lambda_i = r_i,0 / r_0,0 won't change

    # update the shape function zeta
    arr_zeta_new = np.ones_like(arr_zeta)
    arr_r_new = np.ones_like(arr_r)
    if mode == "initial": # first step; run a single Newton step to get new zeta_im array (equation 8)
        # calculate f'_{i, m}, new shape functions, and new r_{i, m}
        for i in range(N_L):
            for m in range(len(mu_new)):
                if m == 0:
                    arr_zeta_new[i][m] = 1.0 # radius at equator is fixed
                else:
                    fp_im = f_im_derivative(arr_zeta, mu_new, arr_lambda_new, arr_delta, arr_Jin, arr_Jinp, arr_Jinpp,
                                            i, m, M_pmass, N_L, arr_r_i0_new, omega, n_max)
                    arr_zeta_new[i][m] = arr_zeta[i][m] - arr_f[i][m] / fp_im # equation 8
                    # if i == 10:
                    #     print("538", i, m, arr_zeta_new[i][m], arr_f[i][m], fp_im, arr_f[i][m]/fp_im)
                    # shape is updated, so we can update radius array using r_i(mu) = zeta_i(mu) * r_i(0)
                arr_r_new[i][m] = arr_zeta_new[i][m] * arr_r_i0[i] # in planetary unit
    elif mode == "iterate":
        for i in range(N_L):
            for m in range(len(mu_new)):
                if m == 0:
                    arr_zeta_new[i][m] = 1.0 # radius at equator is fixed
                else:
                    # define the function we want to minimize to 0, i.e., the potential difference of a layer
                    def func_zeta(zeta_im):
                        """ This is V_i(zeta_im, mu) + Q_i(zeta_im, mu) = U_i(1, 0) and will be used to solve for 
                        the zeta_im that make V_i + Q_i - U_i = 0
                        """
                        vim = Vim_gravity_potential(arr_zeta, mu_new, arr_lambda_new, arr_delta, arr_Jin, arr_Jinp, 
                                                    arr_Jinpp, i, m, N_L, n_max, zeta_im)
                        qim = Qim_centrifugal_potential_zeta(zeta_im, arr_r_i0_new[i], mu[m], omega, Req_si, 
                                                             M_pmass_si, "PU")
                        ui0 = arr_u[i, 0]
                        return np.abs(vim + qim - ui0)
                    

                    # zeta_root = fsolve(func_zeta, [1.0], epsfcn=1.0e-16, maxfev=10000) # the zeta_im that minimizes potential difference
                    # zeta_root = minimize(func_zeta, x0=(1.0,), bounds=((0.5, 1),)) # zeta should always be smaller than 1 (polar radius < equatorial radius)
                    zeta_root = minimize_scalar(func_zeta, bounds=((1.0-del_zeta)*arr_zeta[i][m], (1.0+del_zeta)*arr_zeta[i][m]), # change by at most del_zeta for each step 
                                                method='bounded') # required method to use bounds
                    zeta_root = zeta_root.x
                    # print("602", i, m, zeta_root, func_zeta(zeta_root))
                    
                    # assert that this zeta_root will not lead to layers crossing each other, i.e. 
                    # (1) if i=0 (surface), zeta_root * r_{i=0, mu=0} > zeta_{1, m} * r_{1, 0}, but note that
                    # (2) if i is between 0 and NL-1 (intermediate layer), then
                    #     zeta_{i+1, 0} * r_{i+1, 0} < zeta_root * r_{i, 0} < zeta_{i-1, 0} * r_{i-1, 0}
                    # (3) if i=NL-1 (core), zeta_root * r_{NL-1, 0} < zeta_{NL-2, 0} * r_{NL-2, 0}
                    # if layer crossing happens, adjust zeta_root to avoid it
                    if i == 0:
                        inlayer_r = arr_zeta[i+1][m] * arr_r_i0_new[i+1]
                        if zeta_root * arr_r_i0_new[i] < inlayer_r: # crossing inner layer
                            zeta_root = inlayer_r / arr_r_i0_new[i] 
                    if i > 0 and i < N_L-1:
                        outlayer_r = arr_zeta_new[i-1][m] * arr_r_i0_new[i-1]
                        inlayer_r = arr_zeta[i+1][m] * arr_r_i0_new[i+1]
                        if zeta_root * arr_r_i0_new[i] > outlayer_r: # crossing outer layer
                            zeta_root = outlayer_r / arr_r_i0_new[i]
                        if zeta_root * arr_r_i0_new[i] < inlayer_r: # crossing inner layer
                            zeta_root = inlayer_r / arr_r_i0_new[i]
                    elif i == N_L-1:
                        outlayer_r = arr_zeta_new[i-1][m] * arr_r_i0_new[i-1]
                        if zeta_root * arr_r_i0_new[i] > outlayer_r: # crossing outer layer
                            zeta_root = outlayer_r / arr_r_i0_new[i]

                    arr_zeta_new[i][m] = zeta_root

                arr_r_new[i][m] = arr_zeta_new[i][m] * arr_r_i0_new[i] # in planetary unit

    # update J_i & J'_i, which depend on zeta
    # note that J''_i,0 depend only on M, a, and delta_i (i.e., density), and has no shape dependence, so its
    # update should be saved for later
    arr_Jin_new = np.repeat([np.zeros(n_max)], len(arr_i), axis=0) # arr_Jin has shape (NL, n_max)
    arr_Jinp_new = np.repeat([np.zeros(n_max)], len(arr_i), axis=0) # arr_Jinp has the same shape (NL, n_max)
    for i in range(N_L):
        for n in range(n_max):
            if n % 2 == 0: # only calculate the even n harmonics                
                arr_Jin_new[i, n] = Jin_interior_harmonics(n, M_pmass, arr_delta[i], arr_lambda_new[i],
                                                        arr_zeta_new[i], mu_new, arr_quadw_new)
                arr_Jinp_new[i, n] = Jin_p_exterior_harmonics(n, M_pmass, arr_delta[i], arr_lambda_new[i],
                                                            arr_zeta_new[i], mu_new, arr_quadw_new)
    
    # update U_i and f_i
    arr_u_new = np.ones_like(arr_u)
    arr_f_new = np.ones_like(arr_f)
    for i in range(N_L):
        for m in range(len(mu_new)):
            vim_new = Vim_gravity_potential(arr_zeta_new, mu_new, arr_lambda_new, arr_delta,
                                            arr_Jin_new, arr_Jinp_new, arr_Jinpp, 
                                            i, m, N_L, n_max)
            qim_new = Qim_centrifugal_potential(arr_r_new[i, m], mu_new[m], omega, Req_si, M_pmass_si, "PU")
            arr_u_new[i, m] = vim_new + qim_new 
            if m != 0:
                arr_f_new[i, m] = vim_new + qim_new - arr_u_new[i, 0]
            else:
                arr_f_new[i, m] = 0.0 

    # update pressure (equation 9), in both SI and planetary unit (PU)
    # arr_p_si_new = np.ones_like(arr_p_si)
    # arr_p_si_new[0] = arr_p_si[0]
    arr_p_pu_new = np.ones_like(arr_p_pu)
    arr_p_pu_new[0] = arr_p_pu[0] # P_0 (surface pressure) is keep fixed
    # print("744", arr_p_pu_new[0])
    # print("745, U", arr_u_new)
    # P_1, P_2, ..., P_{N_L - 1} are calculated using equation 9
    # for i in range(1, N_L):
    #     arr_p_si_new[i] = arr_p_si_new[i-1] + arr_rho_si[i-1] * (arr_u_new[i, 0] - arr_u_new[i-1, 0])
    for i in range(1, N_L):
        arr_p_pu_new[i] = arr_p_pu_new[i-1] + arr_rho_pu[i-1] * (arr_u_new[i, 0] - arr_u_new[i-1, 0])
        arr_p_pu_new[i] = abs(arr_p_pu_new[i]) # added to avoid negative pressure when U_i < U_i-1
        # print("750", i, arr_p_pu_new[i], arr_p_pu_new[i-1], arr_rho_pu[i-1], arr_u_new[i, 0], arr_u_new[i-1, 0])
    # arr_p_pu_new = arr_p_si_new * (Req_si**4.0/(smr.G * M_pmass_si**2.0)) # scale to planetary unit
    arr_p_si_new = arr_p_pu_new / (Req_si**4.0/(smr.G * M_pmass_si**2.0)) # scale PU to SI
    # print("753", arr_p_si_new)

    # update density (equation 10) and hence density difference delta, in both SI and PU
    arr_rho_si_new = np.ones_like(arr_rho_si)
    for i in range(N_L-1):
        p_avg = 0.5 * (arr_p_si_new[i+1] + arr_p_si_new[i])
        t_avg = 0.5 * (arr_t_new[i+1] + arr_t_new[i])
        arr_rho_si_new[i] = cms_eos(p_avg, t_avg, li_layers_new[i], rho_p_func)
    arr_rho_si_new[N_L-1] = arr_rho_si[N_L-1] # core density remains unchanged

    arr_rho_pu_new = arr_rho_si_new * (Req_si**3.0/M_pmass_si) # scale by a^3 / M to convert to planetary unit

    arr_delta_new = np.diff(arr_rho_pu_new) # delta_i = rho_i - rho_{i-1}
    arr_delta_new = np.insert(arr_delta_new, 0, arr_rho_pu_new[0]) # delta_0 = rho_0

    # update J''_{i,0}
    arr_Jinpp_new = Ji0_pp(arr_delta_new, M_pmass, 1.0) # here use dimensionless Req=1

    return arr_i_new, mu_new, arr_zeta_new, arr_r_i0_new, arr_r_new, arr_rho_si_new, arr_rho_pu_new, \
        arr_p_si_new, arr_p_pu_new, arr_t_new, arr_lambda_new, arr_delta_new, arr_Jin_new, arr_Jinp_new, \
        arr_Jinpp_new, arr_u_new, arr_f_new, arr_quadw_new, li_layers_new


def save_cms_to_file(cms_results, fn):
    """ cms_results has the same format as args_arr_results in test_rotating_planet.ipynb. """
    
    # save all parameters
    for i in range(len(cms_results)):
        print('Iteration step', i, file=open(fn + '_allout.txt', 'a'))
        print(cms_results[i], file=open(fn + '_allout.txt', 'a'))
        print('', file=open(fn + '_allout.txt', 'a'))
        
    # calculate and save J2n and M
    with open(fn + "_J2n_M.txt", "a") as outfile:
        # write file header
        outfile.write("step\t")
        outfile.write("J0\tJ2\tJ4\tJ6\tJ8\tJ10\tJ12\tJ14\tJ16\tJ18\tJ20\tJ22\tJ24\tJ26\tJ28\tJ30\t")
        outfile.write("M\n")
        
        for i in range(len(cms_results)):
            outfile.write(str(i))
            outfile.write("\t")
            
            lambda_i = cms_results[i][10]
            results_Jn = cms_results[i][12]
            delta_i = cms_results[i][11]
            zeta_2darr = cms_results[i][2]
            mu_arr = cms_results[i][1]
            quadw_arr = cms_results[i][17]

            # calculate zonal gravity harmonics of the observable surface field (following equation 23)
            for k in range(16):
                ji_2k = J2n_gravity_harmonics(lambda_i, results_Jn, int(2*k)) * 1.0e6
                outfile.write(str(ji_2k))
                outfile.write("\t")

            # calculate total mass of the planet M
            pmass = M_planet_mass(delta_i, lambda_i, zeta_2darr, mu_arr, quadw_arr)
            outfile.write(str(pmass))
            outfile.write("\n")


def iterate_planet_array(params, steps, output_name, savefile=True, del_zeta=1.0e-3, rho_p_func=None):
    """ Given an initial set of parameters (either generated by initialize_planet_array() or from a
    previous run), run [steps] steps of iteration, and save all parameters as text files. 
    
    Note that del_zeta can either be a float (all step uses the same del_zeta), or it can be a list of
    floats with the same length as steps. Such that each step can has a different user-defined del_zeta.
    """
    assert len(params) == 22 # total number of parameters
    if type(del_zeta) == float:
        del_zeta = [del_zeta] * steps

    args_arr_results = []

    start_time = datetime.datetime.now()

    for i in tqdm(range(steps)):
        args_arr_new = \
        step_planet_array(params[0], params[1], params[2], params[3], params[4], 
                          params[5], params[6], params[7], params[8], params[9], 
                          params[10], params[11], params[12], params[13],
                          params[14], params[15], params[16], params[17], 
                          params[18], params[19], params[20], params[21], 
                          mode="iterate", del_zeta=del_zeta[i], rho_p_func=rho_p_func)
        args_arr_results.append(args_arr_new)
        args_arr_new += (params[19], params[20], params[21]) # append parameters not included in step_planet_array() outputs
        params = args_arr_new # update initial parameters

    if savefile:
        save_cms_to_file(args_arr_results, output_name)
            # print('Iteration step', i, file=open(output_name + '.txt', 'a'))
            # print(args_arr_new, file=open(output_name + '.txt', 'a'))
            # print('', file=open(output_name + '.txt', 'a'))
    
    print('Ran ' + str(steps) + ' steps.')
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

    return args_arr_results # return results in case need to direct access (i.e., not read from text files)


def initialize_planet_array_precise(forward_output, layer_boundaries, layers, NL, r_core, omega, mu, 
                                    arr_quadw, n_max=33):
    """
    Basically the same as initialize_planet_array(), with the important change that this version avoids using SI
    whenever possible. All quantities are calculated in planetary units (PU) to avoid inaccuracy due to float point
    precision.
    """
    # assert input parameters
    assert len(layer_boundaries) == len(layers)
    assert len(mu) == len(arr_quadw)
    assert mu[0] == 0.0 # the first angle should be 0 (equator)
    
    # read in data from forward model results file, where all data are in SI
    forward_data = np.genfromtxt(forward_output, usecols=(0, 1, 2, 3, 4))
    forward_data = np.transpose(forward_data)
    forward_m = forward_data[0]
    forward_r = forward_data[1]
    forward_p = forward_data[2]
    forward_rho = forward_data[3]
    forward_t = forward_data[4]
    forward_layer = np.genfromtxt(forward_output, usecols=(5), dtype="|U6")
    rcore = min(forward_r)
    Req_si = max(forward_r)
    M_pmass_si = max(forward_m)

    # Read in the densities at layer boundaries (outermost density of each layer), and make sure some layers
    # are located right at the boundaries and on the higher density side.
    # First, find the index of the last occurence of layer name (e.g., "core")
    li_last_layer = []
    rho_boundary = []

    rho_interp = interp1d(forward_r, forward_rho)
    rho_interp_inverse = interp1d(forward_rho, forward_r)
    rho_core = float(rho_interp(r_core))
    
    if len(layers) > 1: # only run this part when there are 2 or more layers
        for i in range(len(layers)):
            last_index = list_rindex(forward_layer, layers[i])
            li_last_layer.append(last_index) # append index of last layer to li_last_layer
            if i < len(layers)-1:
                rho_boundary.append(forward_rho[last_index]*(1.0+1.0e-6)) # increase the density by a tiny bit, so when interpolating r(rho) for li_layers, we get the desired (inner) layer, not the layer just above it
            else:
                rho_boundary.append(forward_rho[last_index]) # for the outermost layer, don't change rho so that ri0 == 1.0
        rho_boundary.reverse()
        # now rho_boundary contains densities at the outer edge of each layer; pick layers between such that
        # the total number of layers is at least the same as NL; now pick layers from surface to core, excluding 
        # the surface layer, because it is in rho_boundary already
        arr_rho_si = np.ndarray.tolist(np.logspace(np.log10(rho_boundary[0]), np.log10(rho_core), NL-len(rho_boundary))[1:]) 
        arr_rho_si += rho_boundary
        arr_rho_si.sort()
    # else, there is no boundary within the planet, simply use np.logspace() to define the layers
    else:
        arr_rho_si = np.ndarray.tolist(np.logspace(np.log10(forward_rho[-1]*(1.0+1.0e-6)), np.log10(rho_core), NL)) # increase the density by a tiny bit, to avoid interpolation error
        arr_rho_si.sort()

    arr_rho_si = np.asarray(arr_rho_si)
    arr_rho_pu = arr_rho_si * (Req_si**3.0/forward_m[-1]) # scale by a^3 / M to convert to planetary unit

    # initialize indices i
    NL = len(arr_rho_si) # redefine NL in case of length difference
    arr_i = np.arange(0, NL)
    
    # initialize shape functions zeta_{i,m}; all zeta_{i,m} should be 1 (spherical) at the beginning
    zeta_each_layer = np.ones(len(mu))
    arr_zeta = np.repeat([zeta_each_layer], len(arr_i), axis=0)
    
    # Initialize r_i0, the radius at the equator, which should remain the same throughout the iteration
    # Note that according to Militzer+ 2019 and Militzer & Hubbard 2023, an ideal r_i0 array should make
    # density rho_{i+1} / rho_i = constant. This will reduce discretization error that is intrinsic to the
    # CMS method, and make convergence faster.
    arr_r_i0 = rho_interp_inverse(arr_rho_si)
    arr_r_i0[0] = Req_si # force the radius of the first layer to be the planet's Req
    # note that arr_r_i0 is still in unit of meters at this point
    arr_r_i0_mid = np.ones_like(arr_r_i0) # radius at the midpoints between layer boundaries
    for i in range(len(arr_r_i0) - 1):
        arr_r_i0_mid[i] = 0.5 * (arr_r_i0[i] + arr_r_i0[i+1])
    arr_r_i0_mid[len(arr_r_i0)-1] = rcore # for the innermost layer, midpoint radius is simply rcore
    
    # initialize r_{i,m}; all r_{i', m} should be equal to r_{i', 0} (spherical) at the beginning
    arr_r = np.repeat([np.ones(len(mu))], len(arr_i), axis=0) # initialize array with the right shape
    for i in range(len(arr_i)):
        arr_r[i] = np.ones_like(mu) * arr_r_i0[i] # change each subarray to the correct r_i0 value

    # initialize P_i; note that pressure of layer i should be the same everywhere
    p_interp = interp1d(forward_r, forward_p)
    arr_p_si = p_interp(arr_r_i0)
    arr_p_pu = arr_p_si * (Req_si**4.0/(smr.G * forward_m[-1]**2.0)) # scale by a^4 / G M^2 to convert to planetary unit
    
    # initialize T_i; note that pressure of layer i should be the same everywhere; assume change in shape does not
    # cause change in T_i
    t_interp = interp1d(forward_r, forward_t)
    arr_t = t_interp(arr_r_i0)
    
    # normalize to planetary unit such that r_{1,0} = Req = 1
    # save this till here because the forward_r above is still in unit of meters
    arr_r /= Req_si
    arr_r_i0 /= Req_si 
    arr_r_i0_mid /= Req_si

    # initialize lambda_i = r_i0 / r_00; should start from 1.0 (surface) and decrease as we go towards to core
    arr_lambda = arr_r_i0 / max(arr_r_i0)
    
    # initialize delta_i, the density difference between layer i and layer i-1; special case: delta_0 = rho_0
    arr_delta = np.diff(arr_rho_pu) # delta_i = rho_i - rho_{i-1}
    arr_delta = np.insert(arr_delta, 0, arr_rho_pu[0]) # delta_0 = rho_0
    
    # initialize J_i & J'_i
    pmass_eq20 = M_planet_mass(arr_delta, arr_lambda, arr_zeta, mu, arr_quadw)
    arr_Jin = np.repeat([np.zeros(n_max)], len(arr_i), axis=0) # arr_Jin has shape (NL, n_max)
    arr_Jinp = np.repeat([np.zeros(n_max)], len(arr_i), axis=0) # arr_Jinp has the same shape (NL, n_max)
    for i in range(len(arr_i)):
        for n in range(n_max):
            if n % 2 == 0: # only calculate the even n harmonics
                arr_Jin[i, n] = Jin_interior_harmonics(n, pmass_eq20, arr_delta[i], arr_lambda[i],
                                                       arr_zeta[i], mu, arr_quadw)
                arr_Jinp[i, n] = Jin_p_exterior_harmonics(n, pmass_eq20, arr_delta[i], arr_lambda[i],
                                                          arr_zeta[i], mu, arr_quadw)
    # note that Jin and Jinp is initialized as np.zeros, so all the odd harmonics would be 0
    
    # initialize J''_i,0
    arr_Jinpp = Ji0_pp(arr_delta, pmass_eq20, 1.0) # here use dimensionless Req=1
    
    # initialize U_{i, m}; U_{i, m} is not equal on a given spheriod at the beginning, and we need to iterate
    # until it is equal everywhere on all spheriod i
    # also initialize f_{i, m} = U_{i, m} - U_i(1, 0), the potential compared to a reference point
    arr_u = np.ones_like(arr_r)
    arr_f = np.ones_like(arr_r)
    for i in range(len(arr_u)):
        for m in range(len(arr_u[0])):
            vim = Vim_gravity_potential(arr_zeta, mu, arr_lambda, arr_delta,
                                        arr_Jin, arr_Jinp, arr_Jinpp, 
                                        i, m, NL, n_max)
            qim = Qim_centrifugal_potential(arr_r[i, m], mu[m], omega, Req_si, M_pmass_si, "PU")
            arr_u[i, m] = vim + qim 
            if m != 0:
                arr_f[i, m] = vim + qim - arr_u[i, 0]
            else:
                arr_f[i, m] = 0.0    
    
    # initialize li_layers
    layer_boundaries = list(map(lambda r: r / Req_si, layer_boundaries)) # scale all to unitless radius
    li_layers = [""] * len(arr_i) # initialize li_layers with empty strings
    for i in range(len(arr_i)):
        if arr_r_i0_mid[i] <= layer_boundaries[0]: # below the innermost layer boundary (usually CMB)
            li_layers[i] = layers[0]
        for j in range(len(layer_boundaries)-1): # only consider the first three boundaries (the last boundary is planet surface)
            if arr_r_i0_mid[i] >= layer_boundaries[j] and arr_r_i0_mid[i] < layer_boundaries[j+1]:
                li_layers[i] = layers[j+1]

    # return all the initialized arrays/lists
    # note that arr_r_i0 is in planetary (unitless) units, while arr_r is in meters
    return arr_i, mu, arr_zeta, arr_r_i0, arr_r, arr_rho_si, arr_rho_pu, arr_p_si, arr_p_pu, arr_t, \
        arr_lambda, arr_delta, arr_Jin, arr_Jinp, arr_Jinpp, arr_u, arr_f, arr_quadw, li_layers


def step_planet_array_precise(arr_i, mu, arr_zeta, arr_r_i0, arr_r, arr_rho_si, arr_rho_pu, arr_p_si, arr_p_pu,
                      arr_t, arr_lambda, arr_delta, arr_Jin, arr_Jinp, arr_Jinpp, arr_u, arr_f, arr_quadw,
                      li_layers, omega, Req_si, M_pmass_si, mode, del_zeta=1.0e-3, rho_p_func=None):
    """ Basically the same as step_planet_array(), with the important change that this version avoids using SI
    whenever possible. All quantities are calculated in planetary units (PU) to avoid inaccuracy due to float point
    precision.
    """
    # assertion
    assert (mode == "initial" or mode == "iterate"), "Invalid mode. Use 'initial' or 'iterate'."

    # derive some useful parameters
    M_pmass_PU = M_planet_mass(arr_delta, arr_lambda, arr_zeta, mu, arr_quadw) # planetary mass in PU, following Eq. 20
    n_max = np.shape(arr_Jin)[1] # maximum gravity harmonics order n
    N_L = len(arr_i)
    # print("1032", arr_r_i0)
    assert arr_r_i0[0] == 1.0, "Equatorial radius at surface is not scaled to 1!"
    Req_PU = 1.0 # scaled equatorial radius of the planet, which should be precisely 1.0

    # unchanged parameters
    arr_i_new = arr_i # number of layers (and hence index of each layer) won't change
    mu_new = mu # angles won't change
    arr_quadw_new = arr_quadw # quadrature weights won't change
    arr_t_new = arr_t # the CMS model is temperature independent
    li_layers_new = li_layers # layer names won't change
    arr_r_i0_new = arr_r_i0 # it's assumed that the equatorial radii of every spheroid is fixed
    arr_lambda_new = arr_lambda # hence, lambda_i = r_i,0 / r_0,0 won't change

    # update the shape function zeta
    arr_zeta_new = np.ones_like(arr_zeta)
    arr_r_new = np.ones_like(arr_r)
    if mode == "initial": # first step; run a single Newton step to get new zeta_im array (equation 8)
        # calculate f'_{i, m}, new shape functions, and new r_{i, m}
        for i in range(N_L):
            for m in range(len(mu_new)):
                if m == 0:
                    arr_zeta_new[i][m] = 1.0 # radius at equator is fixed
                else:
                    fp_im = f_im_derivative(arr_zeta, mu_new, arr_lambda_new, arr_delta, arr_Jin, arr_Jinp, arr_Jinpp,
                                            i, m, M_pmass_PU, N_L, arr_r_i0_new, omega, n_max)
                    diff_zeta = arr_f[i][m] / fp_im
                    # print("1057", type(diff_zeta), type(del_zeta))
                    # print("1058", del_zeta)
                    diff_zeta_sign = 1 if diff_zeta >= 0 else -1
                    if np.abs(diff_zeta) > del_zeta: # the Newton step produces a shape change greater than tolerance
                        diff_zeta = del_zeta * diff_zeta_sign
                    arr_zeta_new[i][m] = arr_zeta[i][m] - diff_zeta # equation 8
                    # if i == 10:
                    #     print("538", i, m, arr_zeta_new[i][m], arr_f[i][m], fp_im, arr_f[i][m]/fp_im)
                    # shape is updated, so we can update radius array using r_i(mu) = zeta_i(mu) * r_i(0)
                arr_r_new[i][m] = arr_zeta_new[i][m] * arr_r_i0[i] # in planetary unit
    elif mode == "iterate":
        for i in range(N_L):
            for m in range(len(mu_new)):
                if m == 0:
                    arr_zeta_new[i][m] = 1.0 # radius at equator is fixed
                else:
                    # define the function we want to minimize to 0, i.e., the potential difference of a layer
                    def func_zeta(zeta_im):
                        """ This is V_i(zeta_im, mu) + Q_i(zeta_im, mu) = U_i(1, 0) and will be used to solve for 
                        the zeta_im that make V_i + Q_i - U_i = 0
                        """
                        vim = Vim_gravity_potential(arr_zeta, mu_new, arr_lambda_new, arr_delta, arr_Jin, arr_Jinp, 
                                                    arr_Jinpp, i, m, N_L, n_max, zeta_im)
                        qim = Qim_centrifugal_potential_zeta(zeta_im, arr_r_i0_new[i], mu[m], omega, Req_si, 
                                                             M_pmass_si, "PU")
                        ui0 = arr_u[i, 0]
                        return np.abs(vim + qim - ui0)
                    
                    zeta_root = minimize_scalar(func_zeta, 
                                                bounds=((1.0-del_zeta)*arr_zeta[i][m], 
                                                        (1.0+del_zeta)*arr_zeta[i][m]), # change by at most del_zeta for each step 
                                                method='bounded') # required method to use bounds
                    zeta_root = zeta_root.x
                    
                    # assert that this zeta_root will not lead to layers crossing each other, i.e. 
                    # (1) if i=0 (surface), zeta_root * r_{i=0, mu=0} > zeta_{1, m} * r_{1, 0}, but note that
                    # (2) if i is between 0 and NL-1 (intermediate layer), then
                    #     zeta_{i+1, 0} * r_{i+1, 0} < zeta_root * r_{i, 0} < zeta_{i-1, 0} * r_{i-1, 0}
                    # (3) if i=NL-1 (core), zeta_root * r_{NL-1, 0} < zeta_{NL-2, 0} * r_{NL-2, 0}
                    # if layer crossing happens, adjust zeta_root to avoid it
                    if i == 0:
                        inlayer_r = arr_zeta[i+1][m] * arr_r_i0_new[i+1]
                        if zeta_root * arr_r_i0_new[i] < inlayer_r: # crossing inner layer
                            zeta_root = inlayer_r / arr_r_i0_new[i] 
                    if i > 0 and i < N_L-1:
                        outlayer_r = arr_zeta_new[i-1][m] * arr_r_i0_new[i-1]
                        inlayer_r = arr_zeta[i+1][m] * arr_r_i0_new[i+1]
                        if zeta_root * arr_r_i0_new[i] > outlayer_r: # crossing outer layer
                            zeta_root = outlayer_r / arr_r_i0_new[i]
                        if zeta_root * arr_r_i0_new[i] < inlayer_r: # crossing inner layer
                            zeta_root = inlayer_r / arr_r_i0_new[i]
                    elif i == N_L-1:
                        outlayer_r = arr_zeta_new[i-1][m] * arr_r_i0_new[i-1]
                        if zeta_root * arr_r_i0_new[i] > outlayer_r: # crossing outer layer
                            zeta_root = outlayer_r / arr_r_i0_new[i]

                    arr_zeta_new[i][m] = zeta_root

                arr_r_new[i][m] = arr_zeta_new[i][m] * arr_r_i0_new[i] # in planetary unit

    # update J_i & J'_i, which depend on zeta
    # note that J''_i,0 depend only on M, a, and delta_i (i.e., density), and has no shape dependence, so its
    # update should be saved for later
    arr_Jin_new = np.repeat([np.zeros(n_max)], len(arr_i), axis=0) # arr_Jin has shape (NL, n_max)
    arr_Jinp_new = np.repeat([np.zeros(n_max)], len(arr_i), axis=0) # arr_Jinp has the same shape (NL, n_max)
    for i in range(N_L):
        for n in range(n_max):
            if n % 2 == 0: # only calculate the even n harmonics                
                arr_Jin_new[i, n] = Jin_interior_harmonics(n, M_pmass_PU, arr_delta[i], arr_lambda_new[i],
                                                           arr_zeta_new[i], mu_new, arr_quadw_new)
                arr_Jinp_new[i, n] = Jin_p_exterior_harmonics(n, M_pmass_PU, arr_delta[i], arr_lambda_new[i],
                                                              arr_zeta_new[i], mu_new, arr_quadw_new)
    
    # update U_i and f_i
    arr_u_new = np.ones_like(arr_u)
    arr_f_new = np.ones_like(arr_f)
    for i in range(N_L):
        for m in range(len(mu_new)):
            vim_new = Vim_gravity_potential(arr_zeta_new, mu_new, arr_lambda_new, arr_delta,
                                            arr_Jin_new, arr_Jinp_new, arr_Jinpp, 
                                            i, m, N_L, n_max)
            qim_new = Qim_centrifugal_potential(arr_r_new[i, m], mu_new[m], omega, Req_si, M_pmass_si, "PU")
            arr_u_new[i, m] = vim_new + qim_new 
            if m != 0:
                arr_f_new[i, m] = vim_new + qim_new - arr_u_new[i, 0]
            else:
                arr_f_new[i, m] = 0.0 

    # update pressure (equation 9), in both SI and planetary unit (PU)
    arr_p_pu_new = np.ones_like(arr_p_pu)
    arr_p_pu_new[0] = arr_p_pu[0] # P_0 (surface pressure) is keep fixed
    for i in range(1, N_L):
        arr_p_pu_new[i] = arr_p_pu_new[i-1] + arr_rho_pu[i-1] * (arr_u_new[i, 0] - arr_u_new[i-1, 0])
        arr_p_pu_new[i] = abs(arr_p_pu_new[i]) # added to avoid negative pressure when U_i < U_i-1
    # Pressure in SI is just for record; don't use it for calculation
    arr_p_si_new = arr_p_pu_new / (Req_si**4.0/(smr.G * M_pmass_si**2.0)) # scale PU to SI

    # update density (equation 10) and hence density difference delta, in both SI and PU
    arr_rho_pu_new = np.ones_like(arr_rho_pu)
    for i in range(N_L-1):
        p_avg = 0.5 * (arr_p_pu_new[i+1] + arr_p_pu_new[i])
        t_avg = 0.5 * (arr_t_new[i+1] + arr_t_new[i])
        arr_rho_pu_new[i] = cms_eos(p_avg, t_avg, li_layers_new[i], 
                                    rho_p_func, use_rho_p_func=True) # note that rho_p_func() here should take rho in PU as input
    arr_rho_pu_new[N_L-1] = arr_rho_pu[N_L-1] # core density remains unchanged

    # Density in SI is just for record; don't use it for calculation
    arr_rho_si_new = arr_rho_pu_new / (Req_si**3.0/M_pmass_si) # scale by a^3 / M to convert to SI
    
    arr_delta_new = np.diff(arr_rho_pu_new) # delta_i = rho_i - rho_{i-1}
    arr_delta_new = np.insert(arr_delta_new, 0, arr_rho_pu_new[0]) # delta_0 = rho_0

    # update J''_{i,0}
    arr_Jinpp_new = Ji0_pp(arr_delta_new, M_pmass_PU, 1.0) # here use dimensionless Req=1

    return arr_i_new, mu_new, arr_zeta_new, arr_r_i0_new, arr_r_new, arr_rho_si_new, arr_rho_pu_new, \
        arr_p_si_new, arr_p_pu_new, arr_t_new, arr_lambda_new, arr_delta_new, arr_Jin_new, arr_Jinp_new, \
        arr_Jinpp_new, arr_u_new, arr_f_new, arr_quadw_new, li_layers_new


def iterate_planet_array_precise(params, steps, output_name, savefile=True, del_zeta=1.0, rho_p_func=None):
    """ Given an initial set of parameters (either generated by initialize_planet_array() or from a
    previous run), run [steps] steps of iteration, and save all parameters as text files. 
    
    Note that del_zeta can either be a float (all step uses the same del_zeta), or it can be a list of
    floats with the same length as steps. Such that each step can has a different user-defined del_zeta.
    """
    assert len(params) == 22 # total number of parameters
    if type(del_zeta) == float:
        del_zeta = [del_zeta] * steps

    args_arr_results = []

    start_time = datetime.datetime.now()

    for i in tqdm(range(steps)):
        args_arr_new = \
        step_planet_array_precise(params[0], params[1], params[2], params[3], params[4], 
                                params[5], params[6], params[7], params[8], params[9], 
                                params[10], params[11], params[12], params[13],
                                params[14], params[15], params[16], params[17], 
                                params[18], params[19], params[20], params[21], 
                                mode="initial", del_zeta=del_zeta[i], rho_p_func=rho_p_func)
        args_arr_results.append(args_arr_new)
        args_arr_new += (params[19], params[20], params[21]) # append parameters not included in step_planet_array() outputs
        params = args_arr_new # update initial parameters

    if savefile:
        save_cms_to_file(args_arr_results, output_name)

    print('Ran ' + str(steps) + ' steps.')
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

    return args_arr_results # return results in case need to direct access (i.e., not read from text files)


# ====== TOF7 - implementation following Nettelmann+2021 ====================================================
def AkQ():
    """ Calculate (m_rot / 3) * sum{c_i0k} following Eq. A1. """
    pass # TODO


def AkV():
    """ Calculate sum{c_i0k S0} + sum{c_i2k S2} + ... + sum{c'_i0k S'0} + ... following the expression on the
    upper left of page 17, assuming that all Sn and Sn' terms are known. """
    pass # TODO