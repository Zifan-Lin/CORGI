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
            # raise Exception("The scalars p1 and p2 do not bound a root!")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan # instead of rasing an exception, simply return NaN to avoid script from termination; check output files for NaN and rerun those
        
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
    
    for i in tqdm(range(len(mass_list))):
        mt = mass_list[i]
        tc = 300.0
        # bc_ignore='mass_mgsio3'
        # ps = 1.0e4
        # print("271", mass_fractions)

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
            outfile.write('{:.6e}'.format(pc)) # Pc in Pa
            outfile.write('\t')
            outfile.write('{:.6f}'.format(M / Me)) # Mp in Me
            outfile.write('\t')
            outfile.write('{:.6f}'.format(R / Re)) # Rp in Re
            outfile.write('\t')
            outfile.write('{:.6e}'.format(psurf_output)) # Psurf in Pa
            outfile.write('\t')
            outfile.write('{:.6f}'.format(cmf)) # CMF
            outfile.write('\t')
            outfile.write('{:.6f}'.format(crf)) # CRF
            outfile.write('\n')
        
        # print('PC solved for mass = %f M_Earth' % (mt / Me))


# ================================================================================================
def MR_curve_homogeneous(option, pcs):
    """ Generate M-R curve for homogeneous planet made of pure iron (option="iron"), pure silicate (option="silicate"),
        or pure water (option="water"). Note that the water planets are assumed to be isothermal at 300 K. On Li Zeng's
        website he supplied multiple pure water MR curves assuming 300-1000 K temperature.
    """
    if option not in ["iron", "silicate", "water"]:
        raise ValueError("Invalid option! Only [iron] and [silicate] and [water] allowed.")
    
    blockPrint()
    for i in tqdm(range(len(pcs))):
        # initialize parameters
        pc = pcs[i]
        tc = 300.0
        rc = 10.0
        ps = 1.0e4
        delr = 100.0
        
        if option == "iron":
            ms, rs, _, _, _ =\
                solve_mprhot_iter(pc, tc, "tab_Zeng16 Zeng2016_core", "isothermal",
                                  rc=rc, delr=delr, ps=ps, 
                                  max_mass=None,
                                  maxsteps=1000000)

            # write P(0), M(R), R to file
            with open('Results/MR_curve_rocky_Zeng2016EOS_cmf100.txt', 'a') as outfile:
                if i == 0:
                    outfile.write('# P_core (Pa)\tM (Me)\tR (Re)')
                    outfile.write('\n')
                outfile.write('{:.6e}'.format(pc))
                outfile.write('\t')
                outfile.write('{:.6f}'.format(ms[-1] / Me))
                outfile.write('\t')
                outfile.write('{:.6f}'.format(rs[-1] / Re))
                outfile.write('\n')
        
        elif option == "silicate":
            ms, rs, _, _, _ =\
                solve_mprhot_iter(pc, tc, "tab_Zeng16 Zeng2016_mantle", "isothermal",
                                  rc=rc, delr=delr, ps=ps, 
                                  max_mass=None,
                                  maxsteps=1000000)
            
            # write P(0), M(R), R to file
            with open('Results/MR_curve_rocky_Zeng2016EOS_cmf000.txt', 'a') as outfile:
                if i == 0:
                    outfile.write('P_core (Pa)\tM (Me)\tR (Re)')
                    outfile.write('\n')
                outfile.write('{:.6e}'.format(pc)) # Pc in Pa
                outfile.write('\t')
                outfile.write('{:.6f}'.format(ms[-1] / Me)) # Mp in Me
                outfile.write('\t')
                outfile.write('{:.6f}'.format(rs[-1] / Re)) # Rp in Re
                outfile.write('\n')
        
        elif option == "water":
            ms, rs, _, _, _ =\
                solve_mprhot_iter(pc, tc, "tab_AQUA", "isothermal",
                                  rc=rc, delr=delr, ps=ps, 
                                  max_mass=None,
                                  maxsteps=1000000)
            
            # write P(0), M(R), R to file
            with open('Results/MR_curve_water_AQUAEOS.txt', 'a') as outfile:
                if i == 0:
                    outfile.write('P_core (Pa)\tM (Me)\tR (Re)')
                    outfile.write('\n')
                outfile.write('{:.6e}'.format(pc))
                outfile.write('\t')
                outfile.write('{:.6f}'.format(ms[-1] / Me))
                outfile.write('\t')
                outfile.write('{:.6f}'.format(rs[-1] / Re))
                outfile.write('\n')
    enablePrint()
        
        
def MR_curve_2layer(mlist):
    """ Generate M-R curves for 2-layer planets with an iron core and a silciate mantle, using the Zeng+2016 EOS. """
    print("M-R curve computation starts...")
    
    blockPrint()
    # 50% rock + 50% water
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [500000, 500000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_50rock_50water.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_mantle",
                                  eos_mantle="tab_AQUA")
    
    # 10% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [100000, 900000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf010.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")
    
    # 20% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [200000, 800000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf020.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")
    
    # 30% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [300000, 700000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf030.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")
    
    # 40% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [400000, 600000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf040.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")
    
    # 50% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [500000, 500000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf050.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")
    
    # 60% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [600000, 400000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf060.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")
    
    # 70% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [700000, 300000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf070.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")
    
    # 80% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [800000, 200000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf080.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")
    
    # 90% CMF
    solve_pc_fixedM_fixedFraction(1.0e8, 1.0e14, [900000, 100000, 0, 0],
                                  mlist, 'Results/MR_curve_rocky_Zeng2016EOS_cmf090.txt', 
                                  eos_core="tab_Zeng16 Zeng2016_core",
                                  eos_mantle="tab_Zeng16 Zeng2016_mantle")

    enablePrint()
    print()
    print("All cases completed!")
    
    
def main():
    MR_curve_2layer()


if __name__ == '__main__':
    main()
    
    
    
    