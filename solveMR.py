#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 21:22:07 2021

@author: linzifan

A simple solid exoplanet mass-radius relation code based on Seager et al.
2007 "MASS-RADIUS RELATIONSHIPS FOR SOLID EXOPLANETS" (S07)

Basic assumptions:
    - mass conservation (eqn 1 of S07)
    - hydrostatic equilibrium (eqn 2 of S07)
    - low-pressure, constant temperature EOS (Vinet or BME EOS)
    
All units are in SI.

This code focus on solving for M, R, P_center, composition, etc. of a
planet, given some known observables (e.g. we know its M and R, trying to
solve for P_center and composition that matches the observation).
"""

#=========================================================================
#====== import necessary packages
#=========================================================================

import numpy as np
from scipy.optimize import fsolve
from EOS import *
from simpleMR import *
import plotMR as pltmr
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
hfont = {'fontname':'times'}
import datetime
from tqdm import tqdm
import sys, os

#=========================================================================
#====== helper functions
#=========================================================================

# Disable print statements
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore print statements
def enablePrint():
    sys.stdout = sys.__stdout__

#=========================================================================
#====== numerical functions
#=========================================================================

def solve_pc_bisect(p1, p2, mt, layers, layer_mass_fractions, 
                    err_fraction=1.0e-4,
                    ps=1.0e5, rhoc_guess=1.0e4, 
                    rc=10.0, delr=100.0,
                    plot_profiles=False,
                    plot_prem=False,
                    save_figures=False,
                    pname='single_multilayer_planet'):
    """
    Assuming we know the total mass and composition of a planet, solve
    for a pc (central pressure) that makes total mass of the planet equal
    mt (the a priori total mass of the planet), to match the M(R)=M outer
    boundary condition. (In reality M(R) does not exactly equal M due to
    finite step size - the program always overshoots a little).

    Parameters
    ----------
    p1 : float
        Lower bound of the solution range. p1 must < p2.
    p2 : float
        Upper bound of the solution range.
    mt : float
        Known total mass of the planet in kg.
    err_fraction : float, optional
        Tolerable error in total mass fraction. By default it is 1.0e-4.
    Other parameters are the same as run_single_planet_multilayer()

    Returns
    -------
    Solution pm that minimize surface pressure, if the function can be solved.
    Runs recursively if error is > err.
    Spits out an exception if a root cannot befound within (p1, p2).

    """
    assert p1 < p2
    err_mass = err_fraction * mt  # convert err fraction into err mass in kg
    
    # check if p1 and p2 bound a root
    m1 = run_single_planet_multilayer(mass=mt, layers=layers,
                                      layer_mass_fractions=layer_mass_fractions,
                                      pc=p1, ps=ps, rhoc_guess=rhoc_guess,
                                      rc=rc, delr=delr, plot_profiles=plot_profiles,
                                      plot_prem=plot_prem,
                                      save_figures=save_figures,
                                      pname=pname)[1]
    m2 = run_single_planet_multilayer(mass=mt, layers=layers,
                                      layer_mass_fractions=layer_mass_fractions,
                                      pc=p2, ps=ps, rhoc_guess=rhoc_guess,
                                      rc=rc, delr=delr, plot_profiles=plot_profiles,
                                      plot_prem=plot_prem,
                                      save_figures=save_figures,
                                      pname=pname)[1]
    delm1 = m1 - mt
    delm2 = m2 - mt
    # print(p1, p2, m1, m2, delm1, delm2)
    if np.sign(delm1) == np.sign(delm2):
        raise Exception("The scalars p1 and p2 do not bound a root")
        
    # get midpoint
    pm = (p1 + p2) / 2.0
    
    delmm = run_single_planet_multilayer(mass=mt, layers=layers,
                                      layer_mass_fractions=layer_mass_fractions,
                                      pc=pm, ps=ps, rhoc_guess=rhoc_guess,
                                      rc=rc, delr=delr, plot_profiles=plot_profiles,
                                      plot_prem=plot_prem,
                                      save_figures=save_figures,
                                      pname=pname)[1] - mt
    print("{:.6e}".format(p1), "{:.6e}".format(p2), "{:.6e}".format(err_mass),
          "{:.6e}".format(delmm))
    if np.abs(delmm) < err_mass:
        return pm
    elif np.sign(delm1) == np.sign(delmm):
        return solve_pc_bisect(pm, p2, mt, layers, layer_mass_fractions,
                               err_fraction=err_fraction,
                               ps=ps, rhoc_guess=rhoc_guess,
                               rc=rc, delr=delr, plot_profiles=plot_profiles,
                               plot_prem=plot_prem,
                               save_figures=save_figures,
                               pname=pname)
    elif np.sign(delm2) == np.sign(delmm):
        return solve_pc_bisect(p1, pm, mt, layers, layer_mass_fractions,
                               err_fraction=err_fraction,
                               ps=ps, rhoc_guess=rhoc_guess,
                               rc=rc, delr=delr, plot_profiles=plot_profiles,
                               plot_prem=plot_prem,
                               save_figures=save_figures,
                               pname=pname)


def solve_pc_minp(p1, p2, mt, layers, layer_mass_fractions, 
                    err_fraction=1.0e-4,
                    ps=1.0e5, rhoc_guess=1.0e4, 
                    rc=10.0, delr=100.0,
                    plot_profiles=False,
                    plot_prem=False,
                    save_figures=False,
                    pname='single_multilayer_planet'):
    """
    Similar to solve_pc_bisect(), but this function iteratively solves for
    the minimum pressure that satisfy a looser err_fraction, in order to
    minimize surface pressure while keeping total mass relatively well
    defined.

    Parameters
    ----------
    See documentation for solve_pc_bisect()

    Returns
    -------
    None.

    """
    pass


def solve_pc_fixedM_fixedFraction(p1, p2, mass_fractions, mass_list, output_filename, 
                                  m_err_fraction=1.0e-4,  # error fraction of total mass
                                  cmf_err_fraction=1.0e-4,  # error fraction of core mass fraction
                                  bc_ignore='mass_mgsio3', tmode_core='isothermal',
                                  eos_core="tab_Zeng21 Zeng2021_core",
                                  eos_mantle="tab_Zeng21 Zeng2021_mantle",
                                  psurf=1.0e4, mc=None, rc=10.0):
    """
    This version aims to solve for a central pressure that satisfy the
    total mass and the mass fraction given.
    
    Start with iron + silicate 2-layer planet and extend to multilayer
    planets if possible.
    
    Parameters
    ----------
    p1, p2 : float
        Pressure in Pa. p1 is the lower bound of the solution range. p1 must < p2. p2 is the upper bound of the 
        solution range.
    mass_list : list of float
    output_filename : string

    Returns
    -------
    Writes the P(0), M(R), R, P(R), core mass fraction, core radius fraction
    to [output_filename]
    """
    #===== beginning of recursive function ============================
    def solve_pc_fixedM_bisect(p1, p2, mt, layer_mass_fractions, 
                    m_err_fraction=1.0e-4,  # error fraction of total mass
                    cmf_err_fraction=1.0e-4,  # error fraction of core mass fraction
                    ps=psurf, rc=10.0, delr=100.0, tc=300.0,
                    bc_ignore='mass_mgsio3', tmode_core='isothermal',
                    eos_core="tab_Zeng21 Zeng2021_core",
                    eos_mantle="tab_Zeng21 Zeng2021_mantle", mc=None):
        """ Similar to solve_pc_bisect, but solves for the central mass
        of a iron + silicate 2-layer planet. """
        assert p1 < p2
        assert layer_mass_fractions[2] == layer_mass_fractions[3] == 0  # consider only 2-layer planet for now
        print("204", layer_mass_fractions)
        cmf = layer_mass_fractions[0] / sum(layer_mass_fractions)  # mass fraction of iron core
        err_mass = m_err_fraction * mt  # convert err fraction into err mass in kg
        
        # check if p1 and p2 bound a root
        _, m1, r1, _, _, _, mlist1, rlist1, plist1, tlist1, rholist1, _, mb1, rb1, pb1, tb1, _ = \
            run_single_planet_4layers(mt, layer_mass_fractions, pc=p1, ps=psurf, tc=tc, 
                                      rc=rc, delr=delr, bc_ignore=bc_ignore,
                                      tmode_core=tmode_core, eos_core=eos_core,
                                      eos_mantle=eos_mantle, mc=mc)
        _, m2, r2, _, _, _, mlist2, rlist2, plist2, tlist2, rholist2, _, mb2, rb2, pb2, tb2, _ = \
            run_single_planet_4layers(mt, layer_mass_fractions, pc=p2, ps=psurf, tc=tc, 
                                      rc=rc, delr=delr, bc_ignore=bc_ignore,
                                      tmode_core=tmode_core, eos_core=eos_core,
                                      eos_mantle=eos_mantle, mc=mc)
        # cmf1 = mb1[0] / m1
        # cmf2 = mb2[0] / m2

        delm1 = m1 - mt
        delm2 = m2 - mt
        # print(p1, p2, m1, m2, delm1, delm2)
        if np.sign(delm1) == np.sign(delm2):
            raise Exception("The scalars p1 and p2 do not bound a root!")
            
        # get midpoint
        pm = (p1 + p2) / 2.0
        
        _, mm, rm, _, _, _, mlistm, rlistm, plistm, tlistm, rholistm, _, mbm, rbm, pbm, tbm, _ = \
            run_single_planet_4layers(mt, layer_mass_fractions, pc=pm, ps=psurf, tc=tc, 
                                      rc=rc, delr=delr, bc_ignore=bc_ignore,
                                      tmode_core=tmode_core, eos_core=eos_core,
                                      eos_mantle=eos_mantle, mc=mc)
        delmm = mm - mt
        cmfm = mbm[0] / mm
        delcmf = np.abs(cmfm - cmf)
        print("Bisect function stepping...")
        print("P1 P2 allowed_error(kg) error(kg) expected_cmf calculated_cmf delta_cmf")
        print("{:.6e}".format(p1), 
              "{:.6e}".format(p2), 
              "{:.6e}".format(err_mass),
              "{:.6e}".format(delmm),
              "{:.6f}".format(cmf),
              "{:.6f}".format(cmfm),
              "{:.6f}".format(delcmf))
        
        if np.abs(delmm) < err_mass and delcmf < cmf_err_fraction:  # [and] is the strict condition; can loosen to [or] when cmf is not very small (<~ 1%)
            return pm, mm, rm, mbm, rbm, pbm
        elif np.sign(delm1) == np.sign(delmm):
            return solve_pc_fixedM_bisect(pm, p2, mt, layer_mass_fractions,
                                   m_err_fraction=m_err_fraction,
                                   cmf_err_fraction=cmf_err_fraction,
                                   ps=psurf, rc=rc, delr=delr, tc=tc,
                                   bc_ignore=bc_ignore, tmode_core=tmode_core,
                                   eos_core=eos_core, eos_mantle=eos_mantle, mc=mc)        
        elif np.sign(delm2) == np.sign(delmm):
            return solve_pc_fixedM_bisect(p1, pm, mt, layer_mass_fractions,
                                   m_err_fraction=m_err_fraction,
                                   cmf_err_fraction=cmf_err_fraction,
                                   ps=psurf, rc=rc, delr=delr, tc=tc,
                                   bc_ignore=bc_ignore, tmode_core=tmode_core,
                                   eos_core=eos_core, eos_mantle=eos_mantle, mc=mc)
    #===== end of recursive function ============================
    
    for i in range(len(mass_list)):
        mt = mass_list[i]
        tc = 300.0
        # bc_ignore='mass_mgsio3'
        # ps = 1.0e4
        print("271", mass_fractions)

        pc, M, R, mb, rb, pb = \
            solve_pc_fixedM_bisect(p1, p2, mt, mass_fractions,
                                   m_err_fraction=m_err_fraction, 
                                   cmf_err_fraction=cmf_err_fraction, 
                                   tc=tc, ps=psurf, rc=rc, bc_ignore=bc_ignore,
                                   tmode_core=tmode_core, eos_core=eos_core,
                                   eos_mantle=eos_mantle, mc=mc)
        cmf = mb[0] / M
        crf = rb[0] / R
        psurf_output = pb[-1]
        
        with open(output_filename, 'a') as outfile:
            # if i == 0:
            #     outfile.write('P_core (Pa)\tM (Me)\tR (Re)\tP_surf (Pa)\tCMF\tCRF')
            #     outfile.write('\n')
            outfile.write('{:.6e}'.format(pc))
            outfile.write('\t')
            outfile.write('{:.6f}'.format(M / Me))
            outfile.write('\t')
            outfile.write('{:.6f}'.format(R / Re))
            outfile.write('\t')
            outfile.write('{:.6e}'.format(psurf_output))
            outfile.write('\t')
            outfile.write('{:.6f}'.format(cmf))
            outfile.write('\t')
            outfile.write('{:.6f}'.format(crf))
            outfile.write('\n')
        
        print('PC solved for mass = %f M_Earth' % (mt / Me))
    
    # pcs = np.linspace(3884.210526e9, 3894.736842e9, 20)
    # ms = []
    # cmfs = []
    # pss = []
    
    # for pc in pcs:
    #     mt = 9.95 * Me
    #     mass_fractions = [300000, 700000, 0, 0]
    #     # pc = 1565.0e9
    #     tc = 300.0
    #     bc_ignore='mass_mgsio3'
    #     ps = 1.0e4
    #     print('Input conditions are:')
    #     print('Total mass = %f M_Earth' % (mt / Me))
    #     print('Mass fractions:', mass_fractions)
    #     print('P(0) and T(0): %f GPa, %f K' % (pc/1.0e9, tc))
    #     print('Outer boundary conditions: P_surf = %f bar, ignore %s' % (ps/1.0e5, bc_ignore))
    #     print()
    #     begin_time = datetime.datetime.now()
    #     _, m, r, _, _, _, mlist, rlist, plist, tlist, rholist, mb, rb, pb, tb = \
    #         run_single_planet_4layers(mt, mass_fractions, pc=pc, tc=tc, 
    #                                   bc_ignore=bc_ignore, ps=ps)
    #     print('Calculated M = %f M_Earth' % (m / Me))
    #     cmf = mb[0] / m
    #     print('Calculated CMF = %f' % (cmf))
    #     print()
    #     print('=====================')
    #     print()
        
    #     ms.append(m)
    #     cmfs.append(cmf)
    #     pss.append(pb[-1]/1.0e5)
    
    # print(list(pcs), ms, cmfs, pss)


def solve_tc_bisect(tc1, tc2, mtot, layer_mass_fractions, pc,
                    err_fraction=0.01, # ts and teq should be within 1%
                    teq=300.0, # equilibrium temperature to match
                    ps=1.0e5, rhoc_guess=1.0e4, 
                    rc=10.0, delr=100.0, bc_ignore='mass', tmode_core='adiabatic_Boujibar_2020'):
    """ Similar to solve_pc_bisect(), but instead solves for the central temperture tc that makes the surface
    temperature ts converge with equilibrium tempearature teq. Central pressure, pc, total mass of the planet,
    and mass fractions of layers are fixed. The only variable is tc. """
    assert tc1 < tc2
    err_temp = err_fraction * teq  # convert err fraction into err temp in K
    
    # check if tc1 and tc2 bound a root
    _, _, _, _, _, _, _, _, _, tlist1, _, _, _, _, _, _, _ =\
        run_single_planet_4layers(mtot, layer_mass_fractions, 
                                  pc=pc, ps=ps, tc=tc1,
                                  rc=rc, delr=delr,
                                  bc_ignore=bc_ignore,
                                  tmode_core=tmode_core,
                                  maxsteps=1000000,
                                  fully_adiabatic_hhe=True)
    tsurf1 = tlist1[-1]
    _, _, _, _, _, _, _, _, _, tlist2, _, _, _, _, _, _, _ =\
        run_single_planet_4layers(mtot, layer_mass_fractions, 
                                  pc=pc, ps=ps, tc=tc2,
                                  rc=rc, delr=delr,
                                  bc_ignore=bc_ignore,
                                  tmode_core=tmode_core,
                                  maxsteps=1000000,
                                  fully_adiabatic_hhe=True)
    tsurf2 = tlist2[-1]

    delt1 = tsurf1 - teq
    delt2 = tsurf2 - teq

    if np.sign(delt1) == np.sign(delt2):
        raise Exception("The scalars tc1 and tc2 do not bound a root! tc1 and tc2:", tc1, tc2)
    
    # get midpoint
    tcm = (tc1 + tc2) / 2.0
    
    # calculate tsurf using midpoint tcore
    _, _, _, _, _, _, _, _, _, tlistm, _, _, _, _, _, _, _ =\
        run_single_planet_4layers(mtot, layer_mass_fractions, 
                                  pc=pc, ps=ps, tc=tcm,
                                  rc=rc, delr=delr,
                                  bc_ignore=bc_ignore,
                                  tmode_core=tmode_core,
                                  maxsteps=1000000,
                                  fully_adiabatic_hhe=True)
    tsurfm = tlistm[-1]
    deltm = tsurfm - teq
    print("Bisect function stepping...")
    print("Tc1", "Tc2", "err_temp", "delt_midpoint")
    print("{:.6e}".format(tc1), "{:.6e}".format(tc2), "{:.6e}".format(err_temp),
          "{:.6e}".format(deltm))
    if np.abs(deltm) < err_temp:
        return tcm
    elif np.sign(delt1) == np.sign(deltm):
        return solve_tc_bisect(tcm, tc2, mtot, layer_mass_fractions, pc,
                               err_fraction=err_fraction, teq=teq,
                               ps=ps, rhoc_guess=rhoc_guess,
                               rc=rc, delr=delr, bc_ignore=bc_ignore, tmode_core=tmode_core)    
    elif np.sign(delt2) == np.sign(deltm):
        return solve_tc_bisect(tc1, tcm, mtot, layer_mass_fractions, pc,
                               err_fraction=err_fraction, teq=teq,
                               ps=ps, rhoc_guess=rhoc_guess,
                               rc=rc, delr=delr, bc_ignore=bc_ignore, tmode_core=tmode_core)  

    
def write_MPrho(mlist, rlist, plist, rholist):
    pass


# ================================================================================================
# ====== Some functions for the Runge-Kutta method
# ================================================================================================
def rk4_step(f, m, y, h, rho):
    """
    Perform a single RK4 step for a system of ODEs. 
    See e.g., https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    k1 = f(m, y, rho)
    k2 = f(m + h / 2, y + h * k1 / 2, rho)
    k3 = f(m + h / 2, y + h * k2 / 2, rho)
    k4 = f(m + h, y + h * k3, rho)
    y_next = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_next


def solve_rk4(f, eosf, m0, y0, m_end, rho0, step_ratio=1.0e-4, h_min=1.0e10):
    """
    Solve a system of ODEs using the RK4 method with dynamic M, rho, and r.
    
    - f is the ODE function(s)
    - eosf is the equation of state function (depend on mass and pressure)
    
    h_min is the minimum mass step, which is roughly the mass of a ~100 m radius iron sphere.
    """
    # define m_vals with variable step size
    m_vals = [m0]
    y_vals = [y0]
    rho_vals = [rho0] # rho at the surface
    
    while m_vals[-1] > m_end:
        m_current = m_vals[-1]
        r_current = y_vals[-1][0]
        rho_current = rho_vals[-1]
        
        # Calculate the new mass using a variable step size
        # depending on the current radius, decrease step_ratio, to avoid divergence near r=0
        step_ratio_scaled = step_ratio
        if r_current < 1000.0*1.0e3:
            step_ratio_scaled *= 0.1
        if r_current < 100.0*1.0e3:
            step_ratio_scaled *= 0.1
        if r_current < 50.0*1.0e3:
            step_ratio_scaled *= 0.1
        if r_current < 10.0*1.0e3:
            step_ratio_scaled *= 0.1
        if r_current < 5.0*1.0e3:
            step_ratio_scaled *= 0.1
        
        h = m_current * step_ratio_scaled  # Decrease step size based on current mass
        if h < h_min:
            h = h_min  # Ensure that h does not go below the minimum step size
        
        # Perform RK4 step
        y_next = rk4_step(f, m_current, y_vals[-1], -h, rho_current)  # Step backwards in mass
        
        rho_updated = eosf(m_current-h, y_next[1], m0)  # Update rho using the updated mass and pressure
        
        # if r < 100 m, stop iteration
        if y_next[0] <= 100:
            if y_next[0] > 0:
                # add new value to m, y, and rho vals, then break
                m_vals.append(m_current - h)  # Update the mass
                y_vals.append(y_next)
                rho_vals.append(rho_updated)
                break
            else: # we run into negative r
                break # break directly without storing the values
        
        # add new value to m, y, and rho vals
        m_vals.append(m_current - h)  # Update the mass
        y_vals.append(y_next)
        rho_vals.append(rho_updated)
    
    # Convert to numpy arrays for easier handling
    m_vals = np.array(m_vals)
    y_vals = np.array(y_vals)
    rho_vals = np.array(rho_vals)
    return m_vals, y_vals, rho_vals


def coupled_odes(m, y, rho):
    """
    Define the coupled ODEs: dr/dm and dp/dm.
    """
    r, p = y
    
    dr_dm = 1 / (4 * np.pi * r**2 * rho)
    dp_dm = -G * m / (4 * np.pi * r**4)
    
    return np.array([dr_dm, dp_dm])


def mass_radius_at_core(eosf, r0, m0, m_end, rho0):
    """
    Calculate the radius at the planet's core for a given initial radius r0.
    """
    # Update the initial radius in the initial state
    y0 = [r0, 1.0e5]  # Initial state: [radius, pressure]
    m_vals, y_vals, _ = solve_rk4(coupled_odes, eosf, m0, y0, m_end, rho0)
    return m_vals[-1], y_vals[-1, 0]  # mass and radius at the core


def bisection_solve_radius(eosf, m0, m_end, r_end, rho0, r_min, r_max, r_tolerance=1.0e-3*Re):
    """
    Use the bisection method to solve for the initial radius that results in mass/radius -> zero at the core.
    """
    assert r_max > r_min
    
    i = 0
    while r_max - r_min > r_tolerance:
        r_mid = (r_min + r_max) / 2
        m_core_mid, r_core_mid = mass_radius_at_core(eosf, r_mid, m0, m_end, rho0)
        
        if m_core_mid <= m_end and r_core_mid <= r_end: # both M and R inner boundary conditions met
            print(f"Converged. Mcore = {m_core_mid:.8e}, Rcore = {r_core_mid:.8e}")
            return r_mid  # Found the solution
        
        # Determine the direction to adjust
        m_core_min, _ = mass_radius_at_core(eosf, r_min, m0, m_end, rho0)
        m_core_max, _ = mass_radius_at_core(eosf, r_max, m0, m_end, rho0)
        # First judge the mass: see if m_core_min, m_core_mid, and m_core_max are below m_end
        # there are 4 scenarios
        # (1) min, mid, max are all above m_end, which means r_max is too small
        if m_core_min > m_end and m_core_mid > m_end and m_core_max > m_end:
            if i == 0:
                raise ValueError("r_min and r_max do not bound a root! r_max is too small!")
            else: # the largest Rp is closest to the solution
                return r_max
        # (2) min, mid, max are all below m_end, which means r_min is too large
        elif m_core_min <= m_end and m_core_mid <= m_end and m_core_max <= m_end:
            if i == 0:
                raise ValueError("r_min and r_max do not bound a root! r_min is too large!")
            else: # the smallest Rp is closest to the solution
                return r_min
        # (3) min is above m_end, and mid and max are both below. The root is between min and mid
        elif m_core_min > m_end and m_core_mid <= m_end and m_core_max <= m_end:
            r_max = r_mid # update r_max to find root between r_min and r_mid
        # (4) min and mid are above m_end, and max is below. The root is between mid and max
        elif m_core_min > m_end and m_core_mid > m_end and m_core_max <= m_end:
            r_min = r_mid # update r_min to find root between r_mid and r_max
        
        # R_core condition is ignored here
        
        print(f"Iteration {i} finished. Rp is between {r_min/Re:.8f} Re and {r_max/Re:.8f} Re.")
        i += 1
    
    m_converged, r_converged = mass_radius_at_core(eosf, (r_min + r_max) / 2, m0, m_end, rho0)
    if m_converged > m_end:
        print("CAUTION! Mass inner boundary condition not met!")
    if r_converged > r_end:
        print("CAUTION! Radius inner boundary condition not met!")
    print(f"Bisection run ended. Mcore = {m_converged:.8e}, Rcore = {r_converged:.8e}")
    return (r_min + r_max) / 2  # Best estimate of the root


def solve_rk4_envelope(f, eosf, m0, y0, m_end, rho0, t0, lmf, step_ratio=1.0e-4, h_min=1.0e10, t_min=300.0):
    """
    Solve a system of ODEs using the RK4 method with dynamic M, rho, and r.
    
    - f is the ODE function(s)
    - eosf is the equation of state function (depend on mass and pressure)
    
    h_min is the minimum mass step, which is roughly the mass of a ~100 m radius iron sphere.
    
    This version can calculate the P-T profile (calling some temperature function) and determine if the
    H/He envelope is radiative or convective.
    
    Note that the eosf in this case should return new density and new temperature. Wrap the RCB determination
    into the eosf.
    """
    # define m_vals with variable step size
    m_vals = [m0]
    y_vals = [y0]
    rho_vals = [rho0] # rho at the surface
    t_vals = [t0] # temperature at the surface
    
    while m_vals[-1] > m_end:
        m_current = m_vals[-1]
        r_current = y_vals[-1][0]
        rho_current = rho_vals[-1]
        t_current = t_vals[-1]
        
        # Calculate the new mass using a variable step size
        # depending on the current radius, decrease step_ratio, to avoid divergence near r=0
        # step_ratio should also be decreased in the H/He atmosphere to keep delr small (~100 m)
        if m_current <= lmf[0] * m0: # in the core, gradually decrease the ratio as we move towards the core
            step_ratio_scaled = step_ratio
            if r_current < 1000.0*1.0e3:
                step_ratio_scaled *= 0.1
            if r_current < 100.0*1.0e3:
                step_ratio_scaled *= 0.1
            if r_current < 50.0*1.0e3:
                step_ratio_scaled *= 0.1
            if r_current < 10.0*1.0e3:
                step_ratio_scaled *= 0.1
            if r_current < 5.0*1.0e3:
                step_ratio_scaled *= 0.1
        elif m_current > lmf[0] * m0 and m_current <= (lmf[0]+lmf[1]) * m0:
            step_ratio_scaled = step_ratio
        elif m_current > (lmf[0]+lmf[1]) * m0 and m_current <= (lmf[0]+lmf[1]+lmf[2]) * m0:
            mass_100m_layer = 4 * np.pi * r_current**2.0 * 100.0 * rho_current
            step_ratio_scaled = mass_100m_layer / m0 # scaling added in case the third layer is water
        else: # we are in the H/He envelope, scale h based on density; calculate the mass of a 100m thick layer
            mass_100m_layer = 4 * np.pi * r_current**2.0 * 100.0 * rho_current
            step_ratio_scaled = mass_100m_layer / m0
        
        # h should be small in the H/He envelope to avoid delr overshoot
        h = m_current * step_ratio_scaled  # Decrease step size based on current mass
        if h < h_min:
            h = h_min  # Ensure that h does not go below the minimum step size
        
        # Perform RK4 step
        y_next = rk4_step(f, m_current, y_vals[-1], -h, rho_current)  # Step backwards in mass
        
        # update rho and t using the EOS function
        rho_updated, t_updated = eosf(m_current-h, y_next, y_vals[-1], t_current, m0, lmf, t_min)
        
        # if r < 100 m, stop iteration
        if y_next[0] <= 100.0:
            if y_next[0] > 0:
                # add new value to m, y, and rho vals, then break
                m_vals.append(m_current - h)  # Update the mass
                y_vals.append(y_next)
                rho_vals.append(rho_updated)
                t_vals.append(t_updated)
                break
            else: # we run into negative r
                break # break directly without storing the values
        
        # add new value to m, y, and rho vals
        m_vals.append(m_current - h)  # Update the mass
        y_vals.append(y_next)
        rho_vals.append(rho_updated)
        t_vals.append(t_updated)
    
    # Convert to numpy arrays for easier handling
    m_vals = np.array(m_vals)
    y_vals = np.array(y_vals)
    rho_vals = np.array(rho_vals)
    t_vals = np.array(t_vals)
    return m_vals, y_vals, rho_vals, t_vals


def outer_bc_pressure(Mp, Rp, t, mmw=3.6087305699338575e-27, gamma=1.0):
    """ Find the pressure outer boundary condition using Eq. (8)-(12) from Rogers & Seager (2010). """
    assert Mp != None, 'Need to supply a valid planet mass (in kg)!'
    assert Rp != None, 'Need to supply a valid planet radius (in meters)!'

    # defined constants
    alpha = 0.68
    beta = 0.45
    C = np.e**(-7.32)
    # compute p(R) at R where tau becomes 1
    a = t ** (0.5-beta) / (C * gamma)
    b1 = G * Mp * (alpha+1) * kb
    b2 = 2 * np.pi * Rp**3.0 * mmw
    pR = (a * np.sqrt(b1/b2)) ** (1/(alpha+1))
    return pR


def mass_radius_at_core_envelope(eosf, r0, m0, m_end, rho0, t0, lmf, p0):
    """
    Calculate the radius at the planet's core for a given initial radius r0.
    """
    # Update the initial radius in the initial state
    y0 = [r0, p0]  # Initial state: [radius, pressure]
    m_vals, y_vals, _, _ = solve_rk4_envelope(coupled_odes, eosf, m0, y0, m_end, rho0, t0, lmf)
    return m_vals[-1], y_vals[-1, 0]  # mass and radius at the core


def bisection_solve_radius_envelope(eosf, m0, m_end, r_end, rho0, r_min, r_max, t0, lmf, p0,
                                    r_tolerance=1.0e-3*Re):
    """
    Use the bisection method to solve for the initial radius that results in mass/radius -> zero at the core.
    """
    assert r_max > r_min
    
    i = 0
    while r_max - r_min > r_tolerance:
        r_mid = (r_min + r_max) / 2

        m_core_mid, r_core_mid = mass_radius_at_core_envelope(eosf, r_mid, m0, m_end, rho0, t0, lmf, p0)
        
        if m_core_mid <= m_end and r_core_mid <= r_end: # both M and R inner boundary conditions met
            print(f"Converged. Mcore = {m_core_mid:.8e}, Rcore = {r_core_mid:.8e}")
            return r_mid  # Found the solution
        
        # Determine the direction to adjust
        m_core_min, _ = mass_radius_at_core_envelope(eosf, r_min, m0, m_end, rho0, t0, lmf, p0)
        m_core_max, _ = mass_radius_at_core_envelope(eosf, r_max, m0, m_end, rho0, t0, lmf, p0)
        # First judge the mass: see if m_core_min, m_core_mid, and m_core_max are below m_end
        # there are 4 scenarios
        # (1) min, mid, max are all above m_end, which means r_max is too small
        if m_core_min > m_end and m_core_mid > m_end and m_core_max > m_end:
            if i == 0:
                raise ValueError("r_min and r_max do not bound a root! r_max is too small!")
            else: # the largest Rp is closest to the solution
                return r_max
        # (2) min, mid, max are all below m_end, which means r_min is too large
        elif m_core_min <= m_end and m_core_mid <= m_end and m_core_max <= m_end:
            if i == 0:
                raise ValueError("r_min and r_max do not bound a root! r_min is too large!")
            else: # the smallest Rp is closest to the solution
                return r_min
        # (3) min is above m_end, and mid and max are both below. The root is between min and mid
        elif m_core_min > m_end and m_core_mid <= m_end and m_core_max <= m_end:
            r_max = r_mid # update r_max to find root between r_min and r_mid
        # (4) min and mid are above m_end, and max is below. The root is between mid and max
        elif m_core_min > m_end and m_core_mid > m_end and m_core_max <= m_end:
            r_min = r_mid # update r_min to find root between r_mid and r_max
        
        # R_core condition is ignored here
        
        print(f"Iteration {i} finished. Rp is between {r_min/Re:.8f} Re and {r_max/Re:.8f} Re.")
        i += 1
    
    m_converged, r_converged = mass_radius_at_core_envelope(eosf, (r_min + r_max) / 2, m0, m_end, rho0, t0, 
                                                            lmf, p0)
    if m_converged > m_end:
        print("CAUTION! Mass inner boundary condition not met!")
    if r_converged > r_end:
        print("CAUTION! Radius inner boundary condition not met!")
    print(f"Bisection run ended. Mcore = {m_converged:.8e}, Rcore = {r_converged:.8e}")
    return (r_min + r_max) / 2  # Best estimate of the root
# ================================================================================================


def find_pc():
    """
    Runs the solve_pc_bisect() function and plots the solution.

    Returns
    -------
    None.

    """
    begin_time = datetime.datetime.now()
    
    # K2-18 b, 8.63 Me, 2.61 Re, assuming three end members from Madhusudhan et al. 2020
   
    # define collection of variables
    
    # # (1) 94.7% Fe, 0.3% H2O, 5% H/He
    # p1 = 4000.0e9; p2 = 6000.0e9  # edicated guess basd on planet mass
    # mt = 8.63 * Me * 0.95  # mass without H/He envelope
    # err_fraction = 1.0e-5
    # layers = ['tabulated fe', 'tabulated h2o']
    # layer_mass_fractions = [996842, 3158]
    # plot_profiles=True
    # save_figures=True
    # pname='K2-18b_Madhu20-1'
    
    # # (2) 45% Earth-like core (13.5% Fe, 31.5% silicates), 54.97% H2O, 0.03% H/He
    # p1 = 2000.0e9; p2 = 4000.0e9  # edicated guess basd on planet mass
    # mt = 8.63 * Me * 0.9997
    # err_fraction = 1.0e-4
    # layers = ['tabulated fe', 'tabulated mgsio3', 'tabulated h2o']
    # layer_mass_fractions = [135041, 315095, 549864]
    # plot_profiles=True
    # save_figures=True
    # pname='K2-18b_Madhu20-2'
    
    # (3) 10% Earth-like core (3% Fe, 7% silicates), 89.994% H2O, 0.006% H/He
    p1 = 600.0e9; p2 = 2000.0e9  # edicated guess basd on planet mass
    mt = 8.63 * Me  # H/He mass fraction is too low so assume to be 0
    err_fraction = 1.0e-4
    layers = ['tabulated fe', 'tabulated mgsio3', 'tabulated h2o']
    layer_mass_fractions = [30000, 70000, 900000]
    plot_profiles=True
    save_figures=False
    pname='K2-18b_Madhu20-3'
    
    # GJ 436 b (Mp = 23.17 Me, Rp = 4.22 Re, T0 = 663 K, Teff = 70 K,) as a 
    # proxy for HAT-P-26 b, 18.59 Me, 6.197 Re; note that the latter is much
    # more puffy due to high Teff ~ 1000 K
    
    # three possible compositions from Rogers & Seager 2010
    # (1) 10% H/He, 10% water, 80% core + mantle (0.5 core + 0.5 mantle)
    # (2) 10% H/He, 40% water , 50% core (pure iron core)
    # (3) 5% H/He, 80% water, 15% core + mantle (0.5 core + 0.5 mantle)
    
    # # (1) set mt to be 0.9 * 23.17 Me, water:mantle:iron = 111111 : 444444 : 444445
    # p1 = 6000.0e9; p2 = 12000.0e9  # edicated guess basd on planet mass
    # mt = 0.9 * 23.17 * Me  # ignore H/He mass in this case because it is < 1.0e-4 tolerance
    # layers = ['tabulated fe', 'tabulated mgsio3', 'tabulated h2o']
    # layer_mass_fractions = [444445, 444444, 111111]
    # plot_profiles=True
    # save_figures=True
    # pname='GJ436b_RS10-1'
    
    # # (2) set mt to be 0.9 * 23.17 Me, water:iron = 444445 : 555555
    # p1 = 12000.0e9; p2 = 20000.0e9  # edicated guess basd on planet mass
    # mt = 0.9 * 23.17 * Me  # ignore H/He mass in this case because it is < 1.0e-4 tolerance
    # layers = ['tabulated fe', 'tabulated h2o']
    # layer_mass_fractions = [555555, 444445]
    # plot_profiles=True
    # save_figures=True
    # pname='GJ436b_RS10-2'
    
    # # (3) set mt to be 0.95 * 23.17 Me, water:mantle:iron = 888888 : 55556 : 55556
    # p1 = 1000.0e9; p2 = 20000.0e9  # edicated guess basd on planet mass
    # mt = 0.95 * 23.17 * Me  # ignore H/He mass in this case because it is < 1.0e-4 tolerance
    # layers = ['tabulated fe', 'tabulated mgsio3', 'tabulated h2o']
    # layer_mass_fractions = [55556, 55556, 888888]
    # plot_profiles=True
    # save_figures=True
    # pname='GJ436b_RS10-3'
    
    pc_solution = solve_pc_bisect(p1=p1, p2=p2, mt=mt, layers=layers,
                                  layer_mass_fractions=layer_mass_fractions,
                                  err_fraction=err_fraction,
                                  plot_profiles=False,
                                  save_figures=False,
                                  pname=pname)
    
    # pc_solution=2300.0e9
    mass, m, r, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary =\
        run_single_planet_multilayer(mt, layers, 
                                     layer_mass_fractions, pc=pc_solution, 
                                     plot_profiles=plot_profiles,
                                     save_figures=save_figures,
                                     pname=pname)
    print("Input mass: ",mass)
    print("Calculated mass: ", m)
    print("Calculated radius: ", r)
    print("M at layer boundaries: ", mboundary)
    print("R at layer boundaries: ", rboundary)
    print("P at layer boundaries: ", pboundary)
    print("Calculated central pressure: ", pc_solution/1.0e9, " GPa")
    print("Execution time: ", datetime.datetime.now() - begin_time)
    
    # write M(r), P(r), rho(r) into text files
    write_MPrho(mlist, rlist, plist, rholist)
    

def MR_curve_2layer():
    """ Write the metadata of a planet on a given M-R curve to file. """
    
    # fixed composition, varying central pressure (and therefore M, R)
    
    # For TOI-1075 b's purpose, use 0.01 Me < M < 15 Me
    # The corresponding pressure range is 1.0e10 to 6.5e12 Pa
    # CMF: try 0.2, 0.3, 0.5, 0.7, 0.8
    # CMF: 0.3, 0.5, 0.7 are used in the final figure
    
    # mlist = np.logspace(-2, np.log10(15), 30)
    # mlist = np.logspace(np.log10(0.5), np.log10(20), 100)
    mlist = np.linspace(0.5, 20, 50)
    # mlist = np.array([20.0])
    # mlist = mlist[1:-1]
    mlist *= Me
    # mlist_n = []
    # for i in range(len(mlist)-1):
    #     mlist_n.append((mlist[i] + mlist[i+1])/2.0)
    # # mlist = [15.0 * Me]
    # mlist_n = np.array(mlist_n)
    # # print(mlist_n / Me)
    
    # 30% iron, 70% silicate
    # solve_pc_fixedM_fixedFraction(9.9e12, 11.0e12, [300000, 700000, 0, 0],
    #                               mlist, 'MR_curve_2layer_Zeng2021EOS_cmf03.txt')
    # 50% iron, 50% silicate
    # solve_pc_fixedM_fixedFraction(10.7e12, 15.0e12, [500000, 500000, 0, 0],
    #                                 mlist, 'MR_curve_2layer_Zeng2021EOS_cmf05.txt')
    # # 20% iron, 80% silicate
    # solve_pc_fixedM_fixedFraction(8.5e9, 5.5e12, [200000, 800000, 0, 0],
    #                               mlist_n, 'MR_curve_2layer_cmf02.txt')
    # 70% iron, 30% silicate 
    solve_pc_fixedM_fixedFraction(10.0e12, 16.0e12, [700000, 300000, 0, 0],
                                  mlist, 'MR_curve_2layer_Zeng2021EOS_cmf07.txt')
    # # 80% iron, 30% silicate 
    # solve_pc_fixedM_fixedFraction(1.2e10, 11.0e12, [800000, 200000, 0, 0],
    #                               mlist, 'MR_curve_2layer_cmf08.txt')
    
    # solve_pc_fixedM_fixedFraction(3.8e12, 4.0e12, [300000, 700000, 0, 0],
                                  # [9.95*Me], 'MR_cmf03.txt')
    # solve_pc_fixedM_fixedFraction(1.0e7, 1.0e12, [300000, 700000, 0, 0],
                                  # [1.0e-2*Me], 'MR_cmf03.txt')
    # solve_pc_fixedM_fixedFraction(6.3e12, 6.4e12, [300000, 700000, 0, 0],
    #                               [15.0*Me], 'MR_cmf03.txt')
    
    # fixed mass and radius, varying composition
    # cmf_list = [0.382748]
    cmf_list = np.linspace(0.005, 0.01, 10)
    # cmf_list = np.array([0.24444444, 0.25555556, 0.26666667, 0.27777778, 0.28888889, 0.3])
    # cmf_list = cmf_list[1:-1]
    print(cmf_list)
    
    for cmf in cmf_list:
        fe_fraction = int(cmf * 1000000)
        mgsio3_fraction = 1000000 - fe_fraction
        solve_pc_fixedM_fixedFraction(4.95e12, 5.00e12, [fe_fraction, mgsio3_fraction, 0, 0],
                                      [9.95*Me], 'TOI1075b_cmf_to_crf_Zeng2016EOS.txt')
    

def MR_curve_homogeneous():
    # pcs = [1.449668e+12]
    # pcs = np.logspace(np.log10(7.0e+08), 11, 30)
    # pcs = np.append(pcs, np.logspace(np.log10(1.05e+11), 13, 50))
    pcs = np.logspace(np.log10(8.780786e+10), np.log10(4.615836e+12), 150) # so that M is between 0.5 and 20 Me for pure MgSiO3
    # pcs = np.logspace(np.log10(3.199267e+11), np.log10(1.832981e+13), 150) # so that M is between 0.5 and 20 Me for pure Fe
    # pcs = np.logspace(np.log10(1.383014e+12), np.log10(1.383209e+12), 10)
    # print(pcs)
    # pcs = pcs[1:-1]
    
    for i in tqdm(range(len(pcs))):
        # initialize parameters
        pc = pcs[i]
        tc = 300.0
        rc = 10.0
        ps = 1.0e4
        delr = 100.0
        
        # #===== code block for Fe ==============================
        # # simulate homogeneous planet
        # blockPrint()
        # ms, rs, _, _, _ =\
        #     solve_mprhot_iter(pc, tc, "tab_Zeng21 Zeng2021_core", "isothermal",
        #                       rc=rc, delr=delr, ps=ps, 
        #                       max_mass=None,
        #                       maxsteps=1000000)
        # enablePrint()
        # M = ms[-1]
        # R = rs[-1]
        
        # # write P(0), M(R), R to file
        # with open('core_homogeneous_Zeng2021EOS.txt', 'a') as outfile:
        #     if i == 0:
        #         outfile.write('# P_core (Pa)\tM (Me)\tR (Re)')
        #         outfile.write('\n')
        #     outfile.write('{:.6e}'.format(pc))
        #     outfile.write('\t')
        #     outfile.write('{:.6f}'.format(M / Me))
        #     outfile.write('\t')
        #     outfile.write('{:.6f}'.format(R / Re))
        #     outfile.write('\n')
        # #======================================================
        
        #===== code block for MgSiO3 ==============================
        # simulate homogeneous planet
        blockPrint()
        ms, rs, ps, _, _ =\
            solve_mprhot_iter(pc, tc, "tab_Zeng21 Zeng2021_mantle", "isothermal",
                              rc=rc, delr=delr, ps=ps, 
                              max_mass=None,
                              maxsteps=1000000)
        enablePrint()
        M = ms[-1]
        R = rs[-1]
        # print(ps[-1])
        
        # write P(0), M(R), R to file
        with open('mantle_homogeneous_Zeng2021_mantle.txt', 'a') as outfile:
            if i == 0:
                outfile.write('# P_core (Pa)\tM (Me)\tR (Re)')
                outfile.write('\n')
            outfile.write('{:.6e}'.format(pc))
            outfile.write('\t')
            outfile.write('{:.6f}'.format(M / Me))
            outfile.write('\t')
            outfile.write('{:.6f}'.format(R / Re))
            outfile.write('\n')
        #======================================================
        
        # #===== code block for H2O ==============================
        # # simulate homogeneous planet
        # ms, rs, _, _, _ =\
        #     solve_mprhot_iter(pc, tc, "tab_Zeng21 Zeng2021_water", "isothermal",
        #                       rc=rc, delr=delr, ps=ps, 
        #                       max_mass=None,
        #                       maxsteps=500000)
        # M = ms[-1]
        # R = rs[-1]
        
        # # write P(0), M(R), R to file
        # with open('H2O_homogeneous_Zeng2021EOS.txt', 'a') as outfile:
        #     if i == 0:
        #         outfile.write('P_core (Pa)\tM (Me)\tR (Re)')
        #         outfile.write('\n')
        #     outfile.write('{:.6e}'.format(pc))
        #     outfile.write('\t')
        #     outfile.write('{:.6f}'.format(M / Me))
        #     outfile.write('\t')
        #     outfile.write('{:.6f}'.format(R / Re))
        #     outfile.write('\n')
        # #======================================================
    
    
def main():
    # MR_curve_2layer()
    MR_curve_homogeneous()


if __name__ == '__main__':
    main()
    
    
    
    