#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:42 2022

@author: linzifan

Retrieval function for the mass and radius code. Given mass, radius, surface temperature and pressure, solve
for the central pressure, central temperature, and layered composition that can reproduce the observed
parameters. 
"""

import numpy as np
import emcee
import simpleMR as smr
import sys, os
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

# Raise error when execution time of a funciton call exceeds time limit, used to avoid unreasonably slow
# interior simulation calls that slow the entire retrieval down
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# Disable print statements
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore print statements
def enablePrint():
    sys.stdout = sys.__stdout__


def retrieve_planet(mass, mass_err, radius, radius_err, ps, ps_err, ts, ts_err,
                    frac_iron_init, frac_silicate_init, frac_water_init, frac_hhe_init, pc_init, tc_init,
                    n_steps=1000):
    """
    Run interior retrieval on a planet given: mass, radius, surface pressure, surface temperature (and their errors).
    Return the most probable composition fractions (Fe:MgSiO3:H2O:H/He), central pressure, central temperature.
    Save the results to a npy file.
    """
    
    def log_likelihood(theta, ind, parameters, para_err):
        """
        Compute the log likelihood given theta, ind, parameters, para_err
        
        theta unpacks input parameters for run_single_planet_4layers(), which will be run in 
        bc_ignore == 'mass' mode. Parameters included in theta are:
        - layer_mass_fractions: 4 parameters in total (frac_iron, frac_silicate, frac_water, frac_hhe)
        - log10 pc
        - tc
        Other input parameters for run_single_planet_4layers() will take the default value or some
        fixed value
        
        (pmass, layer_mass_fractions, pc=330.0e9, ps=1.0e5, tc=5500.0, rc=10.0, delr=100.0,
        bc_ignore='', pradius=None, mmw=2.0920177217007874e-27, teff=50.0, teq=300.0,
        guillot_f=0.5, guillot_gamma=1.0, tmode_core='isothermal')
        
        ind are just indexes (0, 1, 2, 3)
        
        parameters are the things we want to solve for: planet mass, planet radius, surface pressure,
        and surface temperature
        
        para_err are the errors of the parameters
        """
        
        frac_iron, frac_silicate, frac_water, frac_hhe, pc, tc = theta
        
        # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
        sum = frac_iron + frac_silicate + frac_water + frac_hhe
        frac_iron = int(frac_iron / sum * 1000000)
        frac_silicate = int(frac_silicate / sum * 1000000)
        frac_water = int(frac_water / sum * 1000000)
        frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
        
        if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
            return -np.inf
        if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
            return -np.inf
        if tc < 300:
            return -np.inf
        
        print(frac_iron, frac_silicate, frac_water, frac_hhe)
        print(pc, tc)
        print(parameters)
        print(para_err)
        
        try:
            _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _ = \
                smr.run_single_planet_4layers(pmass=parameters[0],
                                            layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                            pc=10**pc, ps=1.0e3, # use some small (0.01 bar) ps for now 
                                            tc=300.0, # isothermal atmosphere for now
                                            rc=10.0, delr=100.0,
                                            bc_ignore='', pradius=parameters[1],
                                            tmode_core='isothermal')
        except:
            return -np.inf
        
        residual = 0.0
        residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
        residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
        residual += (ps - parameters[2])**2.0 / para_err[2] ** 2.0 # pressure error is not well defined, use something like 0.1 bar for now
        # skip temperature outer boundary condition for now. Using isothermal profile
        return -0.5 * residual
    

    init_guess = np.column_stack((frac_iron_init, frac_silicate_init, frac_water_init, frac_hhe_init, pc_init, tc_init))
    _, n_dim = np.shape(init_guess)
    
    n_walkers = len(frac_iron_init) # number of walkers
    
    ind = np.array([0,1,2])
    parameters = np.array([mass, radius, ps])
    para_err = np.array([mass_err, radius_err, ps_err])
    
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood, args=[ind, parameters, para_err])
    sampler.run_mcmc(init_guess, n_steps, progress=True)

    # save the results
    samples = sampler.chain
    save_name = "MR_retrieval_sample.npy"
    np.save(save_name, samples)


def retrieve_gasless_planet(mass, mass_err, radius, radius_err, ps, ps_err, ts, ts_err,
                            frac_iron_init, frac_silicate_init, frac_water_init, pc_init, tc_init,
                            n_steps=1000):
    """
    The same as retrieve_planet() but only support three layers: Fe, MgSiO3, H2O.
    Surface temperature is also fitted.
    """
    
    def log_likelihood_gasless(theta, ind, parameters, para_err):
                
        frac_iron, frac_silicate, frac_water, pc, tc = theta
        
        # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
        sum = frac_iron + frac_silicate + frac_water
        frac_iron = int(frac_iron / sum * 1000000)
        frac_silicate = int(frac_silicate / sum * 1000000)
        frac_water = 1000000 - (frac_iron + frac_silicate)
        
        if max([frac_iron, frac_silicate, frac_water]) > 1000000:
            return -np.inf
        if min([frac_iron, frac_silicate, frac_water]) < 0:
            return -np.inf
        if tc < 300:
            return -np.inf
        
        # print(frac_iron, frac_silicate, frac_water)
        # print(pc, tc)
        # print(parameters)
        # print(para_err)
        
        # try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, 0],
                                        pc=10**pc, ps=1.0e4, # use 0.1 bar to allow pressure to vary around 1 bar a bit 
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal')
        # except:
        #     return -np.inf
        
        log_ps = np.log10(ps) # convert surface pressure from Pa to log Pa
        
        residual = 0.0
        residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
        residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
        residual += (log_ps - parameters[2])**2.0 / para_err[2]**2.0 # pressure error is not well defined, use something like 0.1 bar for now
        residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
        return -0.5 * residual
    

    init_guess = np.column_stack((frac_iron_init, frac_silicate_init, frac_water_init, pc_init, tc_init))
    _, n_dim = np.shape(init_guess)
    
    n_walkers = len(frac_iron_init) # number of walkers
    
    ind = np.array([0,1,2])
    parameters = np.array([mass, radius, ps, ts])
    para_err = np.array([mass_err, radius_err, ps_err, ts_err])
    
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood_gasless, args=[ind, parameters, para_err])
    sampler.run_mcmc(init_guess, n_steps, progress=True)

    # save the results
    samples = sampler.chain
    save_name = "MR_retrieval_gasless_sample.npy"
    np.save(save_name, samples)
    
    
def log_likelihood_gasless(theta, ind, parameters, para_err):
                
    frac_iron, frac_silicate, frac_water, pc, tc = theta
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    sum = frac_iron + frac_silicate + frac_water
    frac_iron = int(frac_iron / sum * 1000000)
    frac_silicate = int(frac_silicate / sum * 1000000)
    frac_water = 1000000 - (frac_iron + frac_silicate)
    
    if max([frac_iron, frac_silicate, frac_water]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    # print(frac_iron, frac_silicate, frac_water)
    # print(pc, tc)
    # print(parameters)
    # print(para_err)
    
    # try:
    blockPrint()
    _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _ = \
        smr.run_single_planet_4layers(pmass=parameters[0],
                                    layer_mass_fractions=[frac_iron, frac_silicate, frac_water, 0],
                                    pc=10**pc, ps=1.0e4, # use 0.1 bar to allow pressure to vary around 1 bar a bit 
                                    tc=tc, # includes central temperature in retrieval
                                    rc=10.0, delr=100.0,
                                    bc_ignore='', pradius=parameters[1],
                                    tmode_core='isothermal')
    enablePrint()
    # except:
    #     return -np.inf
    
    log_ps = np.log10(ps) # convert surface pressure from Pa to log Pa
    
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    residual += 0.0 if np.abs(log_ps - parameters[2]) < para_err[2] \
        else (log_ps - parameters[2])**2.0 / para_err[2]**2.0 # if pressure is between 0.1 and 10 bar, accept
    # residual += 0.0 if np.abs(pmass_calculated - parameters[0]) < para_err[0] \
    #     else (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0 # no residual if mass within 1 sigma error bar
    # residual += 0.0 if np.abs(rs - parameters[1]) < para_err[1] \
    #     else (rs - parameters[1])**2.0 / para_err[1]**2.0 # no residual if radius within 1 sigma error bar
    # residual += (log_ps - parameters[2])**2.0 / para_err[2]**2.0 # pressure error is not well defined, use something like 0.1 bar for now
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual
    
    
def log_likelihood_4layer(theta, ind, parameters, para_err):
    """ log_likelihood_gasless() with additional H/He layer. """
                
    frac_iron, frac_silicate, frac_water, frac_hhe, pc, tc = theta
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    sum = frac_iron + frac_silicate + frac_water + frac_hhe
    frac_iron = int(frac_iron / sum * 1000000)
    frac_silicate = int(frac_silicate / sum * 1000000)
    frac_water = int(frac_water / sum * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    # print(frac_iron, frac_silicate, frac_water, frac_hhe)
    # print(pc, tc)
    # print(parameters)
    # print(para_err)
    
    blockPrint()
    try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                        pc=10**pc, ps=100.0,
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal',
                                        mmw=2.0920177217007874e-27,
                                        teff=50.0, # change Teff assumption for each planet
                                        teq=parameters[3], # change Teq assumption for each planet
                                        guillot_f=0.5, guillot_gamma=1.0)
    except:
        # print('Error encountered with these parameters:')
        # print('frac_iron, frac_silicate, frac_water, frac_hhe, pc, tc')
        # print(frac_iron, frac_silicate, frac_water, frac_hhe, pc, tc)
        return -np.inf
    enablePrint()
    
    # log_ps = np.log10(ps) # convert surface pressure from Pa to log Pa
    
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps > 0.0:
        log_ps = np.log10(ps)
        residual += 0.0 if log_ps < 6.0 \
            else (log_ps - parameters[2])**2.0 / para_err[2]**2.0 # if pressure is below 1.0e6 Pa (10 bar), it's acceptable
    else: # we got negative surface pressure, discard solution
        return -np.inf
    # residual += (log_ps - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_4layer_logHHe(theta, ind, parameters, para_err):
    """ Basically the same as log_likelihood_4layer(), but the mass fraction of H/He spreads log-linearly
    between 1.0 and 1.0e-6. """
                
    frac_iron, frac_silicate, frac_water, frac_hhe_log, pc, tc = theta
    
    if frac_hhe_log > 0 or frac_hhe_log < -6: # X_hhe should be between 1.0 and 1.0e-6
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_hhe = int(1000000 * 10.0 ** frac_hhe_log) # should be between 1 and 1000000
    
    sum = frac_iron + frac_silicate + frac_water
    frac_iron = int(frac_iron / sum * (1000000 - frac_hhe))
    frac_silicate = int(frac_silicate / sum * (1000000 - frac_hhe))
    frac_water = 1000000 - frac_iron - frac_silicate - frac_hhe
    # frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    # print(frac_iron, frac_silicate, frac_water, frac_hhe)
    # print(pc, tc)
    # print(parameters)
    # print(para_err)
    
    blockPrint()
    try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                        pc=10**pc, ps=100.0,
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal',
                                        mmw=2.0920177217007874e-27,
                                        teff=50.0, # change Teff assumption for each planet
                                        teq=parameters[3], # change Teq assumption for each planet
                                        guillot_f=0.5, guillot_gamma=1.0)
    except:
        return -np.inf
    enablePrint()
    
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps > 0.0:
        log_ps = np.log10(ps)
        residual += 0.0 if log_ps < 6.0 \
            else (log_ps - parameters[2])**2.0 / para_err[2]**2.0 # if pressure is below 1.0e6 Pa (10 bar), it's acceptable
    else: # we got negative surface pressure, discard solution
        return -np.inf
    # residual += (log_ps - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_solarFeSi(theta, ind, parameters, para_err):
    """ Core + water + H/He, where the core has solar like Fe/Si ratio (33.2% Fe + 66.8% Si, according to
    Adibekyan et al. 2021). There are only two composition-related free parameters, X_core and X_Z, the latter
    being core mass fraction + water mass fraction. X_HHe is simply 1 - X_core - X_Z. Because it depends
    only on X_core and X_Z (high degree of correlation), it does not need its own free parameter.
    
    Surface pressure below 10 bar (1.0e6 Pa) is accepted with 0 residual; between 10 bar and 1 GPa (1.0e9 Pa) is
    acceptable with nonzero residual; greater than 1 GPa is rejected with -inf residual.
    """
                
    x_core, x_z, pc, tc = theta
    
    # assert that x_core and x_z are between 0 and 1, and x_z > x_core
    if x_core > 1.0 or x_core < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    if x_z < x_core:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = int(x_core * 0.332 * 1000000)
    frac_silicate = int(x_core * 0.668 * 1000000)
    frac_water = int((x_z - x_core) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                        pc=10**pc, ps=100.0,
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal',
                                        mmw=2.0920177217007874e-27,
                                        teff=50.0, # change Teff assumption for each planet
                                        teq=parameters[3], # change Teq assumption for each planet
                                        guillot_f=0.5, guillot_gamma=1.0)
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_solarFeSi_cmf(theta, ind, parameters, para_err):
    """ Basically the same as log_likelihood_3layer_solarFeSi, but now x_core change into cmf (core mass fraction),
    so that mass fraction of iron + silicate is x_z * cmf, iron mass fraction is x_z * cmf * 0.332, and silicate
    mass fraction is x_z * cmf * 0.668. Water mass fraction is x_z * (1-cmf). This will hopefully reduce correlation 
    between parameters.
    """
    
    cmf, x_z, pc, tc = theta
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = int(x_z * cmf * 0.332 * 1000000)
    frac_silicate = int(x_z * cmf * 0.668 * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                        pc=10**pc, ps=100.0,
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal',
                                        mmw=2.0920177217007874e-27,
                                        teff=50.0, # change Teff assumption for each planet
                                        teq=parameters[3], # change Teq assumption for each planet
                                        guillot_f=0.5, guillot_gamma=1.0)
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_pureRock_cmf(theta, ind, parameters, para_err):
    """ Basically the same as log_likelihood_3layer_solarFeSi_cmf(), but instead of assuming solar Fe/Si ratio
    (0.332 iron core + 0.668 silicate mantle), this function assumes a purely rocky core (following other ice
    giant interior structure papers, e.g., Nettelmann et al. 2013).
    """
    
    cmf, x_z, pc, tc = theta
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = 0
    frac_silicate = int(x_z * cmf * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                        pc=10**pc, ps=100.0,
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal',
                                        mmw=2.0920177217007874e-27,
                                        teff=50.0, # change Teff assumption for each planet
                                        teq=parameters[3], # change Teq assumption for each planet
                                        guillot_f=0.5, guillot_gamma=1.0)
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_LmaMixedIce_cmf(theta, ind, parameters, para_err):
    """ Basically the same as log_likelihood_3layer_pureRock_cmf(), but instead of assuming a pure H2O ice layer,
    this function assumes a mixed-composition ice layer with H2O, CH4, and NH3 with solar C:N:O ratio.
    """
    
    cmf, x_z, pc, tc = theta
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = 0
    frac_silicate = int(x_z * cmf * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                        pc=10**pc, ps=100.0,
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal',
                                        mmw=2.0920177217007874e-27,
                                        teff=50.0, # change Teff assumption for each planet
                                        teq=parameters[3], # change Teq assumption for each planet
                                        guillot_f=0.5, guillot_gamma=1.0,
                                        maxsteps=2000000,
                                        eos_water="AVL_planetary_ice 7 4 1 0.0")
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_LmaMixedIceHHe_cmf(theta, ind, parameters, para_err):
    """ Basically the same as log_likelihood_3layer_LmaMixedIce_cmf(), but this version includes 5% H/He in the
    ice layer by number fraction.
    """

    cmf, x_z, pc, tc = theta
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = 0
    frac_silicate = int(x_z * cmf * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                        pc=10**pc, ps=100.0,
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal',
                                        mmw=2.0920177217007874e-27,
                                        teff=50.0, # change Teff assumption for each planet
                                        teq=parameters[3], # change Teq assumption for each planet
                                        guillot_f=0.5, guillot_gamma=1.0,
                                        maxsteps=2000000,
                                        eos_water="AVL_planetary_ice 7 4 1 0.6315789473684211")
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_LmaH2OHHe_cmf(theta, ind, parameters, para_err):
    """ This version allows some H/He in the H2O ice layer and some H2O in the H/He envelope by adding two addition
    parameters xh2o_inner and xh2o_outer.
    """

    cmf, x_z, pc, tc, xh2o_inner, xh2o_outer = theta
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that xh2o_inner and xh2o_outer are between 0 and 1
    if xh2o_inner > 1.0 or xh2o_inner < 0.0 or xh2o_outer > 1.0 or xh2o_outer < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = 0
    frac_silicate = int(x_z * cmf * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
            smr.run_single_planet_4layers(pmass=parameters[0],
                                        layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                        pc=10**pc, ps=100.0,
                                        tc=tc, # includes central temperature in retrieval
                                        rc=10.0, delr=100.0,
                                        bc_ignore='', pradius=parameters[1],
                                        tmode_core='isothermal',
                                        mmw=2.0920177217007874e-27,
                                        teff=50.0, # change Teff assumption for each planet
                                        teq=parameters[3], # change Teq assumption for each planet
                                        guillot_f=0.5, guillot_gamma=1.0,
                                        maxsteps=2000000,
                                        eos_water="AVL_H2O_HHe %f %f" % (xh2o_inner, 1.0-xh2o_inner), # numbers are x(H2O):x(HHe)
                                        eos_hhe="AVL_H2O_HHe %f %f" % (xh2o_outer, 1.0-xh2o_outer))
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_pureRock_cmf_timeout(theta, ind, parameters, para_err):
    """ Basically the same as log_likelihood_3layer_solarFeSi_cmf(), but instead of assuming solar Fe/Si ratio
    (0.332 iron core + 0.668 silicate mantle), this function assumes a purely rocky core (following other ice
    giant interior structure papers, e.g., Nettelmann et al. 2013).
    
    The ice layer is assumed to be pure H2O (using the default AQUA EOS for water).
    """
    
    cmf, x_z, pc, tc = theta
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = 0
    frac_silicate = int(x_z * cmf * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        with time_limit(120): # limit run_single_planet_4layers() call to < 120 seconds
            _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
                smr.run_single_planet_4layers(pmass=parameters[0],
                                            layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                            pc=10**pc, ps=100.0,
                                            tc=tc, # includes central temperature in retrieval
                                            rc=10.0, delr=100.0,
                                            bc_ignore='', pradius=parameters[1],
                                            tmode_core='isothermal',
                                            mmw=2.0920177217007874e-27,
                                            teff=50.0, # change Teff assumption for each planet
                                            teq=parameters[3], # change Teq assumption for each planet
                                            guillot_f=0.5, guillot_gamma=1.0)
    except TimeoutException as e: # if time limit is exceeded, discard run and print
        print("smr.run_single_planet_4layers Timed out!")
        return -np.inf
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_LmaMixedIce_cmf_timeout(theta, ind, parameters, para_err):
    """ Basically the same as log_likelihood_3layer_pureRock_cmf(), but instead of assuming a pure H2O ice layer,
    this function assumes a mixed-composition ice layer with H2O, CH4, and NH3 with solar C:N:O ratio.
    """
    
    cmf, x_z, pc, tc = theta
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = 0
    frac_silicate = int(x_z * cmf * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        with time_limit(120): # limit run_single_planet_4layers() call to < 120 seconds
            _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
                smr.run_single_planet_4layers(pmass=parameters[0],
                                            layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                            pc=10**pc, ps=100.0,
                                            tc=tc, # includes central temperature in retrieval
                                            rc=10.0, delr=100.0,
                                            bc_ignore='', pradius=parameters[1],
                                            tmode_core='isothermal',
                                            mmw=2.0920177217007874e-27,
                                            teff=50.0, # change Teff assumption for each planet
                                            teq=parameters[3], # change Teq assumption for each planet
                                            guillot_f=0.5, guillot_gamma=1.0,
                                            maxsteps=2000000,
                                            eos_water="AVL_planetary_ice 7 4 1 0.0")
    except TimeoutException as e: # if time limit is exceeded, discard run and print
        print("smr.run_single_planet_4layers Timed out!")
        return -np.inf
    except: # if any other error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_LmaMixedIceHHe_cmf_timeout(theta, ind, parameters, para_err, 
                                                     ratios=[7, 4, 1, 0.6315789473684211], z2=0.0):
    """ Basically the same as log_likelihood_3layer_LmaMixedIce_cmf(), but this version includes 5% H/He in the
    ice layer by number fraction.
    """

    cmf, x_z, pc, tc = theta
    assert len(ratios) == 4 # O, C, N, H/He number fractions; default values are for 5 wt% H/He
    ratios = list(map(lambda x:str(x), ratios))
    ratios = " ".join(ratios)
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = 0
    frac_silicate = int(x_z * cmf * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        with time_limit(120): # limit run_single_planet_4layers() call to < 120 seconds
            _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
                smr.run_single_planet_4layers(pmass=parameters[0],
                                            layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                            pc=10**pc, ps=100.0,
                                            tc=tc, # includes central temperature in retrieval
                                            rc=10.0, delr=100.0,
                                            bc_ignore='', pradius=parameters[1],
                                            tmode_core='isothermal',
                                            mmw=2.0920177217007874e-27,
                                            teff=50.0, # change Teff assumption for each planet
                                            teq=parameters[3], # change Teq assumption for each planet
                                            guillot_f=0.5, guillot_gamma=1.0,
                                            maxsteps=2000000,
                                            eos_water="AVL_planetary_ice " + ratios, 
                                            eos_hhe="AVL_H2O_HHe %f %f" % (z2, 1.0-z2))
    except TimeoutException as e: # if time limit is exceeded, discard run and print
        print("smr.run_single_planet_4layers Timed out!")
        return -np.inf
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_3layer_LmaH2OHHe_cmf_timeout(theta, ind, parameters, para_err):
    """ This version allows some H/He in the H2O ice layer and some H2O in the H/He envelope by adding two addition
    parameters xh2o_inner and xh2o_outer.
    """

    cmf, x_z, pc, tc, xh2o_inner, xh2o_outer = theta
    
    # assert that cmf and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that xh2o_inner and xh2o_outer are between 0 and 1
    if xh2o_inner > 1.0 or xh2o_inner < 0.0 or xh2o_outer > 1.0 or xh2o_outer < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = 0
    frac_silicate = int(x_z * cmf * 1000000)
    frac_water = int(x_z * (1-cmf) * 1000000)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_water)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_water, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_water, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    try:
        with time_limit(120): # limit run_single_planet_4layers() call to < 120 seconds
            _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
                smr.run_single_planet_4layers(pmass=parameters[0],
                                            layer_mass_fractions=[frac_iron, frac_silicate, frac_water, frac_hhe],
                                            pc=10**pc, ps=100.0,
                                            tc=tc, # includes central temperature in retrieval
                                            rc=10.0, delr=100.0,
                                            bc_ignore='', pradius=parameters[1],
                                            tmode_core='isothermal',
                                            mmw=2.0920177217007874e-27,
                                            teff=50.0, # change Teff assumption for each planet
                                            teq=parameters[3], # change Teq assumption for each planet
                                            guillot_f=0.5, guillot_gamma=1.0,
                                            maxsteps=2000000,
                                            eos_water="AVL_H2O_HHe %f %f" % (xh2o_inner, 1.0-xh2o_inner), # numbers are x(H2O):x(HHe)
                                            eos_hhe="AVL_H2O_HHe %f %f" % (xh2o_outer, 1.0-xh2o_outer))
    except TimeoutException as e: # if time limit is exceeded, discard run and print
        print("smr.run_single_planet_4layers Timed out!")
        return -np.inf
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_4layer_carbon_planet_free_timeout(theta, ind, parameters, para_err):
    """ Free retrieval likelihood function for carbon planets. There are 6 free parameters:
     - cmf (Fe core mass fraction in the iron-silicate part)
     - x_fe_mgsio3 (Fe + MgSiO3 part mass fraction in the iron-silicate-carbon part)
     - x_z (heavy element mass fraction, mass fraction of the inner three layers of iron, mgsio3, and carbon)
     - pc
     - tc
     - xz_atm (heavy element mass fraction in the atmosphere, represented by changing the MMW in the atmosphere)
    """

    cmf, x_fe_mgsio3, x_z, pc, tc, xz_atm = theta
    
    # assert that cmf, x_fe_mgsio3, and x_z are between 0 and 1
    if cmf > 1.0 or cmf < 0.0 or x_fe_mgsio3 > 1.0 or x_fe_mgsio3 < 0.0 or x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that cmf <= x_fe_mgsio3 <= x_z:
    if cmf > x_fe_mgsio3 or cmf > x_z or x_fe_mgsio3 > x_z:
        return -np.inf
    # assert that xz_atm is between 0 and 1
    if xz_atm > 1.0 or xz_atm < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = int(cmf * x_fe_mgsio3 * x_z * 1.0e6)
    frac_silicate = int((1.0-cmf) * x_fe_mgsio3 * x_z * 1.0e6)
    frac_carbon = int((1.0-x_fe_mgsio3) * x_z * 1.0e6)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_carbon)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_carbon, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_carbon, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    massH = 1.66053906660e-27 # mass of H atom in kg
    try:
        with time_limit(120): # limit run_single_planet_4layers() call to < 120 seconds
            _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
                smr.run_single_planet_4layers(pmass=parameters[0],
                                            layer_mass_fractions=[frac_iron, frac_silicate, frac_carbon, frac_hhe],
                                            pc=10**pc, ps=100.0,
                                            tc=tc, # includes central temperature in retrieval
                                            rc=10.0, delr=100.0,
                                            bc_ignore='', pradius=parameters[1],
                                            mmw=(2.1732*(1.0-xz_atm) + 18.01528*xz_atm)*massH, # scaled MMW based on atmospheric metallicity
                                            teff=50.0, # change Teff assumption for each planet
                                            teq=parameters[3], # change Teq assumption for each planet
                                            guillot_f=0.5, guillot_gamma=1.0,
                                            maxsteps=2000000,
                                            tmode_core="isothermal",
                                            tmode_water ='isothermal',
                                            eos_core="tab_Zeng21 Zeng2021_core",
                                            eos_mantle="tab_Zeng21 Zeng2021_mantle",
                                            eos_water="tabulated carbon",
                                            eos_hhe="AVL_H2O_HHe %f %f" % (xz_atm, 1.0-xz_atm))
    except TimeoutException as e: # if time limit is exceeded, discard run and print
        print("smr.run_single_planet_4layers Timed out!")
        return -np.inf
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual


def log_likelihood_4layer_carbon_planet_solar_timeout(theta, ind, parameters, para_err):
    """ Also for carbon planets, but in this case, the Fe/MgSiO3/carbon mass fraction of the heavy element core is 
    fixed. There are only 4 free parameters:
     - x_z (heavy element mass fraction, mass fraction of the inner three layers of iron, mgsio3, and carbon)
     - pc
     - tc
     - xz_atm (heavy element mass fraction in the atmosphere, represented by changing the MMW in the atmosphere)

    It is assumed that cmf = 0.332 (solar f_fe), carbon mass fraction is 20% of the heavy element core.
    """

    x_z, pc, tc, xz_atm = theta
    
    # assert that x_z is between 0 and 1
    if x_z > 1.0 or x_z < 0.0:
        return -np.inf
    # assert that xz_atm is between 0 and 1
    if xz_atm > 1.0 or xz_atm < 0.0:
        return -np.inf
    # assert that pc is positive (so that the central pressure is at least 10**0=1 Pa)
    if pc < 0.0:
        return -np.inf
    
    # convert frac_iron, frac_silicate, frac_water, frac_hhe from random floats to int summing to 1000000
    frac_iron = int(0.332 * 0.8 * x_z * 1.0e6)
    frac_silicate = int((1.0-0.332) * 0.8 * x_z * 1.0e6)
    frac_carbon = int(0.2 * x_z * 1.0e6)
    frac_hhe = 1000000 - (frac_iron + frac_silicate + frac_carbon)
    
    # assert that the mass fractions are non-negative and smaller than 1,000,000
    if max([frac_iron, frac_silicate, frac_carbon, frac_hhe]) > 1000000:
        return -np.inf
    if min([frac_iron, frac_silicate, frac_carbon, frac_hhe]) < 0:
        return -np.inf
    if tc < 300:
        return -np.inf
    
    blockPrint()
    massH = 1.66053906660e-27 # mass of H atom in kg
    try:
        with time_limit(120): # limit run_single_planet_4layers() call to < 120 seconds
            _, pmass_calculated, rs, ps, ts, _, _, _, _, _, _, _, _, _, _, _, _ = \
                smr.run_single_planet_4layers(pmass=parameters[0],
                                            layer_mass_fractions=[frac_iron, frac_silicate, frac_carbon, frac_hhe],
                                            pc=10**pc, ps=100.0,
                                            tc=tc, # includes central temperature in retrieval
                                            rc=10.0, delr=100.0,
                                            bc_ignore='', pradius=parameters[1],
                                            mmw=(2.1732*(1.0-xz_atm) + 18.01528*xz_atm)*massH, # scaled MMW based on atmospheric metallicity
                                            teff=50.0, # change Teff assumption for each planet
                                            teq=parameters[3], # change Teq assumption for each planet
                                            guillot_f=0.5, guillot_gamma=1.0,
                                            maxsteps=2000000,
                                            tmode_core="isothermal",
                                            tmode_water ='isothermal',
                                            eos_core="tab_Zeng21 Zeng2021_core",
                                            eos_mantle="tab_Zeng21 Zeng2021_mantle",
                                            eos_water="tabulated carbon",
                                            eos_hhe="AVL_H2O_HHe %f %f" % (xz_atm, 1.0-xz_atm))
    except TimeoutException as e: # if time limit is exceeded, discard run and print
        print("smr.run_single_planet_4layers Timed out!")
        return -np.inf
    except: # if any error arises from the forward model, discard run
        return -np.inf
    enablePrint()
    
    # calculate residual based on forward model computed planetary parameters
    residual = 0.0
    residual += (pmass_calculated - parameters[0])**2.0 / para_err[0]**2.0
    residual += (rs - parameters[1])**2.0 / para_err[1]**2.0
    if ps < 0.0 or ps > 1.0e9: # discard solution if we get negative surface pressure or Ps > 1 GPa
        return -np.inf
    elif ps < 1.0e6: # if surface pressure is smaller than 10 bar, it has no impact on the residual
        residual += 0.0
    else: # surface pressure is between 10 bar and 1 GPa
        residual += (np.log10(ps) - parameters[2])**2.0 / para_err[2]**2.0
    residual += (ts - parameters[3])**2.0 / para_err[3]**2.0
    return -0.5 * residual