#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:16:00 2024

@author: linzifan

This script is modified from interpolate_CMS_allout.ipynb. Its function is to read in previous allout files from
CMS simulations, and change the number of layers (say, from NL=128 to 512) by interpolation.
"""

import simpleMR as smr
import run_CMS as runcms
import gravity_harmonics as grav
import numpy as np
from scipy.ndimage import map_coordinates, zoom
import datetime
from multiprocessing import Pool
import re # regular expression
from os import listdir
from os.path import isfile, join
import sys
np.set_printoptions(threshold=sys.maxsize, floatmode='unique') # avoid output truncation

# define some useful planetary constants
period_uranus = 0.71832 * 24 * 3600 # day converted to s
omega_uranus = 2 * np.pi / period_uranus # rad/s
Uranus_M_tot = 8.6663789767e+25 # kg
Uranus_R_eq = 25559.0e3 # m, 1 bar level, https://nssdc.gsfc.nasa.gov/planetary/factsheet/uranusfact.html

# "natural sorting" using regular expression
# see https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# define the interpolation function for a single model
def interpolate_CMS_results(input_params, out_fn, old_NL, new_NL, omega, Req_si, M_pmass_si, layer_boundaries,
                            rcore, layers=["mantle", "water", "H/He"], n_max=33):
    """ Given input_params read from a previous allout file, change the number of layers from old_NL
    to new_NL (say, from 128 to 512 layers) using interpolation, and return the new CMS results list.
    """
    cms_results_interp = [] # new list for interpolation
    
    # i = 0, arr_i
    cms_results_interp.append(np.arange(0, new_NL))
    assert len(cms_results_interp[0]) == new_NL

    # i = 1, mu, do nothing
    cms_results_interp.append(input_params[1])

    # i = 2-10, zeta, r, rho, p, t, lambda, simply linearly interpolate
    for i in range(2, 11):
        if np.shape(input_params[i]) == (old_NL,):
            new_parami = zoom(input_params[i], new_NL/old_NL, order=1)
            assert np.shape(new_parami) == (new_NL,)
        elif np.shape(input_params[i]) == (old_NL, len(input_params[1])) \
        or np.shape(input_params[i]) == (old_NL, n_max):
            new_parami = zoom(input_params[i], (new_NL/old_NL, 1), order=1)
        cms_results_interp.append(new_parami)
    
    # i = 11, delta, calculate from rho
    arr_rho_pu_new = cms_results_interp[6]
    arr_delta_new = np.diff(arr_rho_pu_new) # delta_i = rho_i - rho_{i-1}
    arr_delta_new = np.insert(arr_delta_new, 0, arr_rho_pu_new[0]) # delta_0 = rho_0
    assert len(arr_delta_new) == new_NL
    cms_results_interp.append(arr_delta_new)

    # i = 12, 13, 14, Jin, Jinp, and Jinpp, needs to be calculated from interpolated shape
    # first, calculate M in planetary unit (Eqn. 20 in Militzer et al., 2019)
    M_pmass = grav.M_planet_mass(arr_delta_new, 
                                 cms_results_interp[10], # interpolated lambda
                                 cms_results_interp[2], # interpolated zeta
                                 input_params[1], # mu
                                 input_params[17]) # quadw
    # calculate Jin and Jinp
    arr_Jin_new = np.repeat([np.zeros(n_max)], new_NL, axis=0) # arr_Jin has shape (NL, n_max)
    arr_Jinp_new = np.repeat([np.zeros(n_max)], new_NL, axis=0) # arr_Jinp has the same shape (NL, n_max)
    print("Calculating new Jin and Jinp...")
    for i in range(new_NL):
        for n in range(n_max):
            if n % 2 == 0: # only calculate the even n harmonics                
                arr_Jin_new[i, n] = grav.Jin_interior_harmonics(n, M_pmass, arr_delta_new[i], 
                                                                cms_results_interp[10][i], # lambda
                                                                cms_results_interp[2][i], # zeta
                                                                input_params[1], # mu
                                                                input_params[17]) # quadw
                arr_Jinp_new[i, n] = grav.Jin_p_exterior_harmonics(n, M_pmass, arr_delta_new[i], 
                                                                   cms_results_interp[10][i], # lambda
                                                                   cms_results_interp[2][i], # zeta
                                                                   input_params[1], # mu
                                                                   input_params[17]) # quadw
    arr_Jinpp_new = grav.Ji0_pp(arr_delta_new, M_pmass, 1.0) # here use dimensionless Req=1

    cms_results_interp.append(arr_Jin_new)
    cms_results_interp.append(arr_Jinp_new)
    cms_results_interp.append(arr_Jinpp_new)

    # i = 15, 16, u and f
    # note that calculating u and f directly is extremely slow (because its time complexity goes as NxM)
    # therefore, use interpolation
    arr_u_new = zoom(input_params[15], (new_NL/old_NL, 1), order=1)
    arr_f_new = zoom(input_params[16], (new_NL/old_NL, 1), order=1)
    cms_results_interp.append(arr_u_new)
    cms_results_interp.append(arr_f_new)
    
    # i = 17, quadw, unchanged
    cms_results_interp.append(input_params[17])
    
    # i = 18, li_layers, need to check for layer densities
    layer_boundaries = list(map(lambda r: r / Req_si, layer_boundaries)) # scale all to unitless radius
    li_layers_new = [""] * new_NL # initialize li_layers with empty strings
    
    arr_r_i0_mid = np.ones_like(cms_results_interp[3]) # radius at the midpoints between layer boundaries
    for i in range(len(cms_results_interp[3]) - 1):
        arr_r_i0_mid[i] = 0.5 * (cms_results_interp[3][i] + cms_results_interp[3][i+1])
    arr_r_i0_mid[len(cms_results_interp[3])-1] = rcore # for the innermost layer, midpoint radius is simply rcore
    
    for i in range(new_NL):
        if arr_r_i0_mid[i] <= layer_boundaries[0]: # below the innermost layer boundary (usually CMB)
            li_layers_new[i] = layers[0]
        for j in range(len(layer_boundaries)-1): # only consider the first three boundaries (the last boundary is planet surface)
            if arr_r_i0_mid[i] >= layer_boundaries[j] and arr_r_i0_mid[i] < layer_boundaries[j+1]:
                li_layers_new[i] = layers[j+1]
    cms_results_interp.append(li_layers_new)
    
    assert len(cms_results_interp) == len(input_params)
    
    # print the interpolated results to a text file
    grav.save_cms_to_file([cms_results_interp], out_fn)


# parallel version of the above function
def interpolate_CMS_results_parallel(fns_models, fns_allout, out_fn_list, old_NL, new_NL, omega=omega_uranus,
                                     Req_si=Uranus_R_eq, M_pmass_si=Uranus_M_tot, rcore=10.0, 
                                     layers=["mantle", "water", "H/He"], n_max=33):
    assert len(fns_models) == len(fns_allout)
    num_models = len(fns_models)

    # Read input_params from fns_allout (allout files produced from CMS runs)
    input_params_list = []
    for i in range(len(fns_allout)):
        input_str = runcms.read_file_after_last_target_string(fns_allout[i], 'Iteration step')
        input_str = input_str.replace("array", "np.array")
        input_params = eval(input_str)
        input_params_list.append(input_params)

    # Read layer_boundaries from fns_models (forward models)
    layer_boundaries_list = []
    for i in range(len(fns_models)):
        rbs = [0.0, 0.0, 0.0] # placeholder
        with open(fns_models[i]) as infile:
            for num, line in enumerate(infile, 1):
                if "Radius at layer boundaries (Re):" in line:
                    rbs = line.split(":")[1]
                    rbs = rbs.split()[:3] # the first three boundary radii
                    rbs = list(map(lambda x: float(x) * smr.Re, rbs)) # convert into float in SI units
        layer_boundaries_list.append(rbs)

    # Set up other parameters
    assert len(out_fn_list) == len(fns_models)
    old_NL_list = [old_NL] * num_models
    new_NL_list = [new_NL] * num_models
    omega_list = [omega] * num_models
    Req_si_list = [Req_si] * num_models
    M_pmass_si_list = [M_pmass_si] * num_models
    rcore_list = [rcore] * num_models
    layers_list = [layers] * num_models
    n_max_list = [n_max] * num_models

    args = list(zip(input_params_list, out_fn_list, old_NL_list, new_NL_list, omega_list, Req_si_list, 
                    M_pmass_si_list, layer_boundaries_list, rcore_list, layers_list, n_max_list))
    
    # Begin parallel run
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')
    with Pool() as pool:
        pool.starmap(interpolate_CMS_results, args)
    print()
    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def run_interp_CMS_Uranus_IG_128_512():
    """ Call interpolate_CMS_results_parallel(), interpolate Uranus IG runs from NL=128 to 512. """
    forward_model_dir = "CMS_forward_models/Uranus_IG_forwardParallel/Tc5500K_v2/"
    fns_models = [f for f in listdir(forward_model_dir) if isfile(join(forward_model_dir, f))]
    fns_models = list(filter(lambda x: ".DS_Store" not in x, fns_models))
    fns_models.sort(key=natural_keys)
    fns_models = list(map(lambda x: forward_model_dir + x, fns_models))

    allout_dir = "CMS_results/Uranus_NM25_NL128_IGbisect_precise/Tc5500K_v2/allout_step152/"
    fns_allout_fnonly = [f for f in listdir(allout_dir) if isfile(join(allout_dir, f))]
    fns_allout_fnonly = list(filter(lambda x: ".DS_Store" not in x, fns_allout_fnonly))
    fns_allout_fnonly.sort(key=natural_keys)
    fns_allout = list(map(lambda x: allout_dir + x, fns_allout_fnonly))

    out_fn_list = list(map(lambda x: x.replace("NL128", "NL512"), fns_allout_fnonly))
    
    interpolate_CMS_results_parallel(fns_models, fns_allout, out_fn_list, 127, 512, omega=omega_uranus,
                                     Req_si=Uranus_R_eq, M_pmass_si=Uranus_M_tot, rcore=10.0, 
                                     layers=["mantle", "water", "H/He"], n_max=32)


def main():
    run_interp_CMS_Uranus_IG_128_512()


if __name__ == "__main__":
    main()