#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:26:00 2023

@author: linzifan

Implements the sigle-layer Maclaurin-based model from Hubbard 2012 (doi:10.1088/2041-8205/756/1/L15).
"""

# ============================================================================================================
# Import relevant packages
# ============================================================================================================
import numpy as np
from scipy.special import legendre
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from tqdm.notebook import tqdm


# ============================================================================================================
# Hardcoded weight and abscissa data for N=64 quadrature points
# To use more or less grid points, the user would need to change this part
# Should make N a variable in the future
# Source: https://pomax.github.io/bezierinfo/legendre-gauss.html#n64
# ============================================================================================================
quad_w64 = [0.048690957, 0.048690957, 0.048575467, 0.048575467, 0.048344762, 0.048344762, 0.047999389, 0.047999389, 
       0.047540166, 0.047540166, 0.046968183, 0.046968183, 0.046284797, 0.046284797, 0.045491628, 0.045491628, 
       0.044590558, 0.044590558, 0.043583725, 0.043583725, 0.042473515, 0.042473515, 0.041262563, 0.041262563, 
       0.039953741, 0.039953741, 0.038550153, 0.038550153, 0.037055129, 0.037055129, 0.035472213, 0.035472213, 
       0.033805162, 0.033805162, 0.032057928, 0.032057928, 0.030234657, 0.030234657, 0.028339673, 0.028339673, 
       0.02637747, 0.02637747, 0.024352703, 0.024352703, 0.022270174, 0.022270174, 0.020134823, 0.020134823, 
       0.017951716, 0.017951716, 0.01572603, 0.01572603, 0.013463048, 0.013463048, 0.011168139, 0.011168139, 
       0.00884676, 0.00884676, 0.006504458, 0.006504458, 0.004147033, 0.004147033, 0.001783281, 0.001783281]
quad_x64 = [-0.024350293, 0.024350293, -0.072993122, 0.072993122, -0.121462819, 0.121462819, -0.16964442, 0.16964442,
       -0.217423644, 0.217423644, -0.264687162, 0.264687162, -0.311322872, 0.311322872, -0.357220158, 0.357220158, 
       -0.402270158, 0.402270158, -0.446366017, 0.446366017, -0.489403146, 0.489403146, -0.531279464, 0.531279464, 
       -0.571895646, 0.571895646, -0.611155355, 0.611155355, -0.648965471, 0.648965471, -0.685236313, 0.685236313, 
       -0.71988185, 0.71988185, -0.752819907, 0.752819907, -0.783972359, 0.783972359, -0.813265315, 0.813265315, 
       -0.840629296, 0.840629296, -0.865999398, 0.865999398, -0.889315446, 0.889315446, -0.910522137, 0.910522137, 
       -0.929569172, 0.929569172, -0.946411375, 0.946411375, -0.9610088, 0.9610088, -0.973326828, 0.973326828, 
       -0.983336254, 0.983336254, -0.991013371, 0.991013371, -0.996340117, 0.996340117, -0.999305042, 0.999305042]


# ============================================================================================================
# Functions
# ============================================================================================================
def q_coeff(omega, req, mass, fenv):
    return fenv**3.0 * (omega**2.0 * req**3.0) / (G * mass)


def legendre_mu(n, mu):
    """
    Given degree n of Legendre polynomial and mu = cos(theta), calculate and return the Pn(mu) value.
    """
    legendre_poly = legendre(n) # this contains polynomial prefactors
    return legendre_poly(mu)


def wi_quadrature(i):
    return quad_w64[i]


def Jn_Hubbard12_3layer(n, fcore, fenv, mu_arr, zeta_arr):
    assert n % 2 == 0 # calculate even Jn components only
    assert np.shape(mu_arr) == np.shape(zeta_arr)
    
    a = (-3 * (1-fcore)) / (n+3)
    b = 0.0
    c = 0.0
    for i in range(len(mu_arr)):
        b += wi_quadrature(i) * legendre_mu(n, mu_arr[i]) * zeta_arr[i]**(n+3)
        c += wi_quadrature(i) * zeta_arr[i]**3.0
    
    return fenv**n * a * (b / c) # combining Eq. 15, 16, 18


def solve_zeta(J2k_arr, q, mu):
    """ Numerically solve for zeta(mu) at each quadrature point mu_i. """
    def func_zeta(zeta):
        sum1 = 0.0
        sum2 = 0.0
        for k in range(len(J2k_arr)):
            n = 2 * (k+1) # k ranges from 0 to 14, but should be 1 to 15 in practice, so that n = 2, 4, ..., 30
            sum1 += zeta**(-n) * J2k_arr[k] * legendre_mu(n, mu)
            sum2 += J2k_arr[k] * legendre_mu(n, 0.0)
            
        a = (1/zeta) * (1 - sum1)
        b = (q/3.0) * zeta**2.0 * (1 - legendre_mu(2, mu))
        
        result = a + b - (q/2) - 1 + sum2
        return result    
    
    # Here, try different ways to find root for zeta
    zeta_root = fsolve(func_zeta, [1.0], epsfcn=1.0e-16, maxfev=10000) # zeta should be close to Req as the deformation is small
    if func_zeta(zeta_root[0]) > 1.0e-12:
        print("Warning! func_zeta(zeta_root) is greater than the permitted error of 1.0e-12.")
    return zeta_root[0]
    
    
def iterate_zeta_Jn(fcore, fenv, q, filename, mu_arr=quad_x64, J2k_arr_initial=None, steps=10000, return_J2k=True):
    """ Given a list of angles mu, fcore, fenv, and the small parameter q, iterate between Eq. 14 and 15
    to solve for J_n for even n from 2 to 30, until all J_n values converge within ~2.0e-16. """
    # first, initialize guesses of J_2k (set as zeros) if no J2k values from a previous run are given
    if J2k_arr_initial is None:
        J2k_arr_initial = np.array(np.zeros(15)) # k ranges from 0 to 15, so n=2k ranges from 0 to 30
    else:
        assert np.shape(np.zeros(15)) == np.shape(J2k_arr_initial) # assert J2k is from J2 to J30
    
    fn = filename
    fn += '_fcore' + str(fcore)
    fn += '_fenv' + str(fenv)
    fn += '_' + str(steps) + 'steps'
    
    # write J_2k harmonics calculated at each step to file
    with open(fn + '.txt', 'a') as outfile:
        outfile.write('\t'.join(['# step', 'J2', 'J4', 'J6', 'J8', 'J10', 'J12', 'J14', 'J16', 'J18', 'J20', 
                                 'J22', 'J24', 'J26', 'J28', 'J30']))
        outfile.write('\n')
        J2k_arr_initial_str = list(map(lambda x: str(x), J2k_arr_initial))
        outfile.write('0\t')
        outfile.write('\t'.join(J2k_arr_initial_str))
        outfile.write('\n')

        most_recent_J2k = J2k_arr_initial

        for s in tqdm(range(steps)):
            zeta_mui_step = []
            for i in range(len(mu_arr)): # calculate zeta_i for each mu_i
                new_zeta = solve_zeta(np.array(most_recent_J2k), q, mu_arr[i])
                zeta_mui_step.append(new_zeta)

            J2k_step = []
            for k in range(len(most_recent_J2k)):
                new_J2k = Jn_Hubbard12_3layer(2*(k+1), fcore, fenv, mu_arr, np.array(zeta_mui_step))
                J2k_step.append(new_J2k)
            most_recent_J2k = J2k_step # update J2k list
            J2k_step = list(map(lambda x: str(x), J2k_step))
            outfile.write(str(s+1))
            outfile.write('\t')
            outfile.write('\t'.join(J2k_step))
            outfile.write('\n')
            
    if return_J2k:
        return most_recent_J2k

