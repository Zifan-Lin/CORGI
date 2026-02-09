#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:27:47 2020

@author: linzifan

A simple solid exoplanet mass-radius relation code based on Seager et al.
2007 "MASS-RADIUS RELATIONSHIPS FOR SOLID EXOPLANETS" (S07)

Basic assumptions:
    - mass conservation (eqn 1 of S07)
    - hydrostatic equilibrium (eqn 2 of S07)
    - low-pressure, constant temperature EOS (Vinet or BME EOS)
    
All units are in SI.
"""

#=========================================================================
#====== import necessary packages
#=========================================================================

# from selectors import EpollSelector
import numpy as np
import pandas as pd 
from scipy.optimize import fsolve
import EOS as eos
import temperature_profile as tpr
import plotMR as pltmr
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
hfont = {'fontname':'times'}
import datetime

#=========================================================================
#====== useful constants, in SI
#=========================================================================

G = 6.67430e-11  # gravitational constant, https://physics.nist.gov/cgi-bin/cuu/Value?bg
Re = 6371000.0  # Earth radius
Me = 5.9722e24  # Earth mass, https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
kb = 1.380649e-23 # Boltzmann constant, J/K
massH = 1.66053906660e-27 # mass of H atom in kg

# coefficients for calculating temperature
alpha_fe = 35.4E-6
alpha_mgsio3 = 24.4E-6
alpha_h2o = 210.0E-6
cp_fe = 460.0
cp_mgsio3 = 845.0  # using MgSiO4 from http://www.minsocam.org/ammin/AM67/AM67_470.pdf as a proxy
cp_h2o = 4184.0

#=========================================================================
#====== numerical functions
#=========================================================================

def plot_EOS():
    """
    Plot some of the equation of state functions, with log10 P being the x
    axis and rho being the y axis.

    Returns
    -------
    None.

    """
    N = 5000  # number of data points
    
    # generate data for P(rho)
    rho_array = np.logspace(2, 8, N)
    # rho_array = np.array((1.0e2))
    # rho_array = np.append(rho_array, np.logspace(3, 4, 70))
    rho_array = np.sort(rho_array)
    
    fe_p_array_BME3 = eos.eos_BME3(162.5e9, rho_array, 7.86e3, 5.5)
    fe_p_array_Vinet = eos.eos_Vinet(156.2e9, rho_array, 8.30e3, 6.08)
    # Experiment determined Fe-Si alloy EOS fitted with Vinet
    # cover high pressures up to 1300 GPa
    # from Wicks et al. Sci. Adv. (2018)
    fe_p_array_Wicks18_7wtSi = eos.eos_Vinet(136.2e9, rho_array, 8.30e3, 5.97)
    fe_p_array_Wicks18_15wtSi = eos.eos_Vinet(227.9e9, rho_array, 8.30e3, 4.74)
    
    mgsio3_p_array_BME3 = eos.eos_BME3(125.0e9, rho_array, 3.22e3, 5.0)
    mgsio3_p_array_BME4 = eos.eos_BME4(247.0e9, rho_array, 4.10e3, 3.97, -0.016e-9)
    # print(eos_BME4(247.0e9, 1.0e5, 4.10e3, 3.97, -0.016e9))
    
    h2o_p_array_BME3 = eos.eos_BME3(23.7e9, rho_array, 1.46e3, 4.15)
    
    # genrate data for inversed EOS, rho(P)
    P_list = list(np.logspace(7, 23, N))
    # P_list = list(np.logspace(7, 9, N)) + list(np.logspace(9, 11, N))[1:] + \
    #     list(np.logspace(11, 23, N))[1:]
    rho_guess_list = list(np.logspace(2, 11, N))
    
    # fe_rho_array_BME3 = []
    fe_rho_array_Vinet = []
    fe_rho_array_Vinet_episilon = []
    fe_rho_array_Vinet_alpha = []
    fe_rho_array_Wicks18_7wtSi = []
    fe_rho_array_Wicks18_15wtSi = []
    mgsio3_rho_array_BME3 = []
    # mgsio3_rho_array_BME4 = []
    h2o_rho_array_BME3 = []
    
    fe_rho_array_TFD = []
    mgsio3_rho_array_TFD = []
    h2o_rho_array_TFD = []
    
    # complete EOS with three segments
    fe_rho_array_full = []
    mgsio3_rho_array_full = []
    h2o_rho_array_full = []
    
    # tabulated EOS with three segments saved in 'EOS/'
    fe_rho_array_table = []
    mgsio3_rho_array_table = []
    h2o_rho_array_table = []
    
    fe_transition_rho = np.arange(7.916819e+03, 8.650964e+03, 1.0)
    fe_transition_p = eos.eos_Vinet(162.5e9, fe_transition_rho, 7.86e3, 5.5)
    fe_transition_p /= 1.0e9  # convert Pa to GPa
    # print(fe_transition_rho[:10])
    # print(fe_transition_p[:10])
    
    for i in range(len(P_list)):
        fe_rho_array_Vinet.append(eos.eos_experimental_inverse(P_list[i], "fe", rho_guess_list[i])) 
        # fe_rho_array_Vinet_episilon.append(eos_Vinet_inverse(156.2e9, P_list[i], 8.30e3, 6.08, rho_guess_list[i]))
        # fe_rho_array_Vinet_alpha.append(eos_Vinet_inverse(162.5e9, P_list[i], 7.86e3, 5.5, rho_guess_list[i]))
        # fe_rho_array_Wicks18_7wtSi.append(eos_Vinet_inverse(136.2e9, P_list[i], 8.30e3, 5.97, rho_guess_list[i])) 
        # fe_rho_array_Wicks18_15wtSi.append(eos_Vinet_inverse(227.9e9, P_list[i], 8.30e3, 4.74, rho_guess_list[i])) 
        mgsio3_rho_array_BME3.append(eos.eos_experimental_inverse(P_list[i], "mgsio3", rho_guess_list[i]))
        h2o_rho_array_BME3.append(eos.eos_experimental_inverse(P_list[i], "h2o", rho_guess_list[i]))
        # mgsio3_rho_array_BME4.append(eos_BME4_inverse(247.0e9, P_list[i], 4.10e3, 3.97, -0.016e-9, rho_guess_list[i]))
        
        fe_rho_array_TFD.append(eos.eos_TFD_SZ67_inverse(56.0, 26.0, P_list[i]))
        mgsio3_rho_array_TFD.append(eos.eos_TFD_SZ67_inverse(20.0, 10.0, P_list[i]))
        h2o_rho_array_TFD.append(eos.eos_TFD_SZ67_inverse(18.0, 10.0, P_list[i]))
        
        fe_rho_array_full.append(eos.eos_homogeneous_inverse(P_list[i], "fe", rho_guess_list[i]))
        mgsio3_rho_array_full.append(eos.eos_homogeneous_inverse(P_list[i], "mgsio3", rho_guess_list[i]))
        h2o_rho_array_full.append(eos.eos_homogeneous_inverse(P_list[i], "h2o", rho_guess_list[i]))

        fe_rho_array_table.append(eos.eos_tabulated(P_list[i], "fe"))
        mgsio3_rho_array_table.append(eos.eos_tabulated(P_list[i], "mgsio3"))
        h2o_rho_array_table.append(eos.eos_tabulated(P_list[i], "h2o"))
        
    # print("Fe BME3")
    # for i in range(len(rho_array)):
    #     print(rho_array[i], "\t", fe_p_array_BME3[i])
    # print()
    # print("MgSiO3 BME3")
    # for i in range(len(rho_array)):
    #     print(rho_array[i], "\t", mgsio3_p_array_BME3[i])
    # print()
    # print("H2O BME3")
    # for i in range(len(rho_array)):
    #     print(rho_array[i], "\t", h2o_p_array_BME3[i])
    
    P_list_GPa = list(map(lambda x: x / 1.0e9, P_list))  # convert Pa into GPa
    
    # plot rho vs P for Fe
    # plt.plot(P_list, fe_rho_array_BME3, 'b--', label='inversed BME3')
    # plt.plot(P_list, fe_rho_array_Vinet, 'r--', label='inversed Vinet')
    # plt.plot(fe_p_array_BME3, rho_array, 'bx', label='BME3')
    # plt.plot(fe_p_array_Vinet, rho_array, 'rx', label='Vinet')
    fig, ax=plt.subplots(1, figsize=(8,5))
    plt.plot(P_list_GPa, fe_rho_array_Vinet, 'b:', label='Vinet')
    # plt.plot(P_list_GPa, fe_rho_array_Vinet_episilon, 'b-', label='Vinet (hcp)')
    # plt.plot(P_list_GPa, fe_rho_array_Vinet_alpha, 'b--', label='Vinet (bcc)')
    plt.plot(P_list_GPa, fe_rho_array_TFD, 'r--', label='TFD')
    # plt.plot(P_list_GPa, fe_rho_array_Wicks18_7wtSi, 'g-', label='Fe + 7wt% Si')
    # plt.plot(P_list_GPa, fe_rho_array_Wicks18_15wtSi, 'g--', label='Fe + 15wt% Si')
    # plt.plot(fe_transition_p, fe_transition_rho, color='cyan', linestyle='-', label='transition')
    # plt.plot(P_list_GPa, fe_rho_array_full, 'k', label='EOS')
    plt.plot(P_list_GPa, fe_rho_array_table, 'k', label='EOS')
    # plt.plot(fe_p_array_Wicks18_7wtSi, rho_array, 'g-', label='Wicks+18, 7 wt% Si')
    # plt.plot((fe_p_array_BME3 + fe_p_array_Vinet)/2.0, rho_array, 'gx', label='average')
    plt.xlim([1.0e-2, 1.0e10])
    plt.ylim([1.0e2, 1.0e8])
    # plt.xlim([1.0e0, 1.0e2])
    # plt.ylim([6000, 22000])
    plt.title("Fe EOS", fontsize=16, **hfont)
    plt.xlabel('P (GPa)', fontsize=16, **hfont)
    plt.ylabel(r'$\rho$ (kg m$^{-3}$)', fontsize=16, **hfont)
    plt.xticks(fontsize=16, **hfont)
    plt.yticks(fontsize=16, **hfont)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize=16)
    plt.savefig("Fe_EOS_Wicks2018_Fe-Si.png", dpi=200, bbox_inches='tight')
    plt.show()
    
    # plot rho vs P for MgSiO3
    # plt.plot(P_list, mgsio3_rho_array_BME3, 'b--', label='inversed BME3')
    # plt.plot(mgsio3_p_array_BME3, rho_array, 'bx', label='BME3')
    # plt.plot(mgsio3_p_array_BME4, rho_array, 'rx', label='BME4')
    # plt.plot(mgsio3_p_array_BME4, rho_array, 'rx', label='BME4')
    fig, ax=plt.subplots(1, figsize=(8,5))
    plt.plot(P_list_GPa, mgsio3_rho_array_BME3, 'b:', label='BME3')
    plt.plot(P_list_GPa, mgsio3_rho_array_TFD, 'r--', label='TFD')
    # plt.plot(P_list_GPa, mgsio3_rho_array_full, 'k', label='EOS')
    plt.plot(P_list_GPa, mgsio3_rho_array_table, 'k', label='EOS')
    plt.xlim([1.0e-2, 1.0e10])
    plt.ylim([1.0e2, 1.0e8])
    plt.title(r"MgSiO$_3$ EOS", fontsize=16, **hfont)
    plt.xlabel('P (GPa)', fontsize=16, **hfont)
    plt.ylabel(r'$\rho$ (kg m$^{-3}$)', fontsize=16, **hfont)
    plt.xticks(fontsize=16, **hfont)
    plt.yticks(fontsize=16, **hfont)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper left', fontsize=16)
    plt.savefig("MgSiO3_EOS_3segments.png", dpi=200, bbox_inches='tight')
    plt.show()
    
    # plot rho vs P for H2O
    fig, ax=plt.subplots(1, figsize=(8,5))
    plt.plot(P_list_GPa, h2o_rho_array_BME3, 'b:', label='BME3')
    plt.plot(P_list_GPa, h2o_rho_array_TFD, 'r--', label='TFD')
    # plt.plot(P_list_GPa, h2o_rho_array_full, 'k', label='EOS')
    plt.plot(P_list_GPa, h2o_rho_array_table, 'k', label='EOS')
    plt.xlim([1.0e-2, 1.0e10])
    plt.ylim([1.0e2, 1.0e8])
    plt.title(r"H$_2$O EOS", fontsize=16, **hfont)
    plt.xlabel('P (GPa)', fontsize=16, **hfont)
    plt.ylabel(r'$\rho$ (kg m$^{-3}$)', fontsize=16, **hfont)
    plt.xticks(fontsize=16, **hfont)
    plt.yticks(fontsize=16, **hfont)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper left', fontsize=16)
    plt.savefig("H2O_EOS_3segments.png", dpi=200, bbox_inches='tight')
    plt.show()
    
    # save the EOS into text files
    with open("iron", "a") as iron_out:
        iron_out.write("{:.6e}".format(1.0))
        iron_out.write("\t")
        iron_out.write("{:.6e}".format(fe_rho_array_full[0]))
        iron_out.write("\n")
        for i in range(len(P_list)):
            iron_out.write("{:.6e}".format(P_list[i]))
            iron_out.write("\t")
            iron_out.write("{:.6e}".format(fe_rho_array_full[i]))
            iron_out.write("\n")
    with open("water", "a") as water_out:
        water_out.write("{:.6e}".format(1.0))
        water_out.write("\t")
        water_out.write("{:.6e}".format(h2o_rho_array_full[0]))
        water_out.write("\n")
        for i in range(len(P_list)):
            water_out.write("{:.6e}".format(P_list[i]))
            water_out.write("\t")
            water_out.write("{:.6e}".format(h2o_rho_array_full[i]))
            water_out.write("\n")
    with open("silicate", "a") as silicate_out:
        silicate_out.write("{:.6e}".format(1.0))
        silicate_out.write("\t")
        silicate_out.write("{:.6e}".format(mgsio3_rho_array_full[0]))
        silicate_out.write("\n")
        for i in range(len(P_list)):
            silicate_out.write("{:.6e}".format(P_list[i]))
            silicate_out.write("\t")
            silicate_out.write("{:.6e}".format(mgsio3_rho_array_full[i]))
            silicate_out.write("\n")
    
    
def solve_mprho_step(mr, pr, rhor, r, delr, eos_mode, eos_args):
    """
    Given mass, pressure, density at radius r, calculate the mass, pressure,
    and density at r + delr.
    
    Parameters
    ----------
    mr : float
        total mass inside radius r, in kg.
    pr : float
        pressure at radius r, in Pa.
    rhor : float
        density at radius r, in kg m-3.
    r : float
        radius, in m.
    delr : float
        radius step, in m.
    eos_mode : string
        decides which EOS function to use. Valid inputs include "Vinet",
        "BME3", "BME4", "homogeneous [component]", where [component] can
        be "h2o", "mgsio3", or "fe".
    eos_args : list of floats
        arguments for EOS functions. If using Vinet or BME3, the list will
        be [K0, rho0, K0p], if using BME4, the list will be 
        [K0, rho0, K0p, K0pp]

    Returns
    -------
    [mn, pn, rhon, rn], a list of floats, representing mass, pressure,
    density, and radius at the new radius (r + delr).
    """
    if "tabulated" not in eos_mode:
        assert len(eos_args) >= 3
        K0 = eos_args[0]
        rho0 = eos_args[1]
        K0p = eos_args[2]
    
    mn = mr + 4 * np.pi * r ** 2.0 * rhor * delr  # new mass from mass conservation
    pn = pr - ((G * mr * rhor) / r ** 2.0) * delr
    
    if eos_mode == "Vinet":
        rhon = eos.eos_Vinet_inverse(K0, pr, rho0, K0p, rhor)
    elif eos_mode == "BME3":
        rhon = eos.eos_BME3_inverse(K0, pr, rho0, K0p, rhor)
    elif eos_mode == "BME4":
        K0pp = eos_args[3]
        rhon = eos.eos_BME4_inverse(K0, pr, rho0, K0p, K0pp, rhor)
    elif "homogeneous" in eos_mode:
        rhon = eos.eos_homogeneous_inverse(pr, eos_mode.split()[1], rhor)
    elif "tabulated" in eos_mode:
        rhon = eos.eos_tabulated(pr, eos_mode.split()[1])
    else:
        raise ValueError("eos_mode is not valid! Check documentation for valid inputs!")
    
    return [mn, pn, rhon, r+delr]


def solve_mprhot_step(mr, pr, rhor, tr, r, delr=10.0, eos_mode="homogeneous h2o",
                      t_mode="isothermal", alpha=None, cp=None, ad_grad=None,
                      taur=None, teff=50.0, teq=300.0, f=0.5, gamma=1.0,
                      gamma0=1.48, gamma1=1.4, alpha0=20.0, 
                      rho0=4260.0, K0=324.0, K0p=3.3, mass_fractions=[0.33,0.33,0.34]):
    """
    Basically the same as solve_mprho_step(), but includes temperature 
    dependence.
    
    Five parameters have been added for H/He atmospheric envelope:
    - taur: optical depth at radius r, which is used to compute the Guillot (2010) temperature profile
    - teff: effective (internal) temperature of the planet
    - teq: equilibrium temperature of the planet
    - f and gamma are for calculating the Guillot (2010) temperature profile
    
    [gamma0, gamma1, alpha0, rho0, K0, K0p] are optional parameters required to compute Boujibar et al. (2020)
    adiabatic temperature profile. Their default values are set to post-perovskite values. Note that K0
    is in GPa.
    
    This function does not unpack EOS parameters - the EOS called by eos_mode
    must be some kind of tabulated EOS.
    
    Valid inputs for [eos_mode] include:
        "homogeneous [component]"
        "tabulated [component]"
        "tab_Seager07 [component]" (the same as "tabulated" in solve_mprho_step())
        "tab_Zeng16 Zeng2016_[component]" (calls the same eos_tabulated() function as "tab_Seager07")
        "tab_Zeng21 Zeng2021_[component]" (calls the same eos_tabulated() function as "tab_Seager07")
        "tab_AQUA"
        "tab_CD21_HHe"
        "AVL" (additive volume law EOS for mixture planets)
        "polytrope [K] [n]" (P = K rho^(1 + 1/n))
    
    Valid inputs for [t_mode] include:
        "isothermal" - t will remain constant
        "adiabatic_alpha_cp" - will call temperature_profile.adiabatic_t_r(), 
            need to supply the coefficient of volumetricthermal expansion 
            [alpha] and the specific heat capacity [cp]
        "adiabatic_Boujibar_2020" - will call temperature_profile adiabatic_t_r_boujibar20_eq9()
            need to supply rho0, K0, K0', gamma0, gamma1, and alpha0.
        "adiabatic_gradAd" - will call temperature_profile.adiabatic_t_r_gradAd(),
            need to supply the adiabatic gradient ad_grad = (d lnT/ d lnP)|S
            if ad_grad is not included in EOS.
        "two_stream_Guillot2010" - will call temperature_profile.two_stream_temp_g10(),
            which is the Guillot (2010) two-stream temperature profile. Needs to supply
            teff, teq, tau, f (default is 0.5), and gamma (default is 1.0). Note that
            tau comes from freedman2008_mean_opacity(p, t).
        "AVL" -  will call temperature_profile.adiabatic_profile_AVL() to calculate
            the adiabatic temperature profile of a mixture interior.

    Returns
    -------
    mn, pn, rhon, tn, rn, all are floats, representing mass, pressure,
    density, temperature, and radius at the new radius (r + delr)..

    """
    # print('353', mr, pr, rhor, tr, r)
    # print(type(mr), type(pr), type(rhor), type(tr), type(r))
    
    # compute M(r + dr) and P(r + dr)
    mn = mr + 4 * np.pi * r ** 2.0 * rhor * delr  # new mass from mass conservation
    pn = pr - ((G * mr * rhor) / r ** 2.0) * delr
    
    # compute rho(r + dr)
    if "homogeneous" in eos_mode:
        rhon = eos.eos_homogeneous_inverse(pr, eos_mode.split()[1], rhor)
    elif "TFD" in eos_mode: 
        assert t_mode == "isothermal", "TFD EOS requires isothermal (i.e. zero temperature) T profile!"
        if "HHe" in eos_mode: # "TFD HHe", special case for zero temperature H/He mixture
            rhon = eos.eos_TFD_SZ67_inverse(1.2598425196850394, 1.08661417, pr) # assuming He mass fraction Y=0.275 (solar); He number fraction is 0.08661417
        elif "pureH" in eos_mode:
            rhon = eos.eos_TFD_SZ67_inverse(1.00784, 1.0, pr) # pure hydrogen
        else:
            raise ValueError("Material not supported for TFD EOS!")
    elif "tab_Seager07" in eos_mode or "tab_Zeng16" in eos_mode or "tab_Zeng21" in eos_mode or \
        "OganovOno2004" in eos_mode or "HM89_rock" in eos_mode or "tabulated" in eos_mode:
        rhon = eos.eos_tabulated(pr, eos_mode.split()[1])
    elif eos_mode == "tab_AQUA":
        rhon, _ = eos.eos_AQUA(pr, tr)
    elif eos_mode == "tab_French2009_H2O":
        rhon = eos.eos_french2009_h2o(pr, tr)
    elif "AVL_planetary_ice" in eos_mode:
        # eos_mode should be in the format e.g. "AVL_planetary_ice 2 1 4"
        avl_mixed_ice_split = eos_mode.split()
        x_h2o = float(avl_mixed_ice_split[1])
        x_ch4 = float(avl_mixed_ice_split[2])
        x_nh3 = float(avl_mixed_ice_split[3])
        if len(avl_mixed_ice_split) > 4:
            x_hhe = float(avl_mixed_ice_split[4])
        else:
            x_hhe = 0.0
        rhon, _ = eos.avl_planetary_ice(pr, tr, x_h2o, x_ch4, x_nh3, x_hhe)
    elif "AVL_H2O_HHe" in eos_mode:
        # eos_mode should be in the format e.g. "AVL_H2O_HHe 90 10"
        avl_h2o_hhe_split = eos_mode.split()
        x_h2o = float(avl_h2o_hhe_split[1])
        x_hhe = float(avl_h2o_hhe_split[2])
        rhon, _ = eos.avl_h2o_hhe(pr, tr, x_h2o, x_hhe)
    elif eos_mode == "tab_CD21_HHe":
        rhon, _, _ = eos.eos_cd21_hhe(pr, tr)
    elif eos_mode == "AVL":
        rhon = eos.avl_eos(pr, tr, mass_fractions)
    elif eos_mode == "iron_with_phase_change": # iron EOS that transitions from liquid to solid
        if eos.is_liquid_Kraus2022(pr, tr):
            rhon = eos.eos_tabulated(pr, "Grant2021_liquidFe")
            # print(pr, tr, "liquid Fe")
        else:
            rhon = eos.eos_tabulated(pr, "Smith2018_solidFe")
            # print(pr, tr, "solid Fe")
    elif "polytrope" in eos_mode:
        poly_K = float(eos_mode.split()[1])
        poly_n = int(eos_mode.split()[2])
        rhon = (pr / poly_K) ** (poly_n/(poly_n+1))
    else:
        raise ValueError("eos_mode is not valid! Check documentation for valid inputs!")
    
    # compute tau(r + dr), if taur is supplied
    if taur != None:
        kappa = tpr.freedman2008_mean_opacity(pr, tr) # tabulated Rosseland mean opacity from P and T
        deltau = - kappa * rhor * delr # Equation (5), Rogers & Seager (2010)
        # print('Delta tau = %f' % deltau)
        # print('rho(r) = %f' % rhor)
        taun = taur + deltau
        # if optical depth < 0, switch to isothermal atmosphere at Teq
        # this is a temporary workaround - should be able to get physical tau once RCB condition is fixed
        if taun < 0:
            tr = teq
            tn = teq
            taun = 0.0
    else:
        taun = None
    
    # compute T(r + dr)
    if t_mode == "isothermal":
        tn = tr
    elif t_mode == "adiabatic_alpha_cp":
        tn = tpr.adiabatic_t_r(tr, mr, pr, rhor, r, delr, alpha, cp, tmin=min(teq, 300.0))
    elif t_mode == "adiabatic_Boujibar_2020":
        tn = tpr.adiabatic_t_r_boujibar20_eq9(tr, rhor, mr, r, delr, gamma0, gamma1, alpha0, rho0, K0, K0p)
    elif t_mode == "adiabatic_gradAd":
        if eos_mode == "tab_AQUA" or eos_mode == "tab_French2009_H2O": # use AQUA adiabatic gradient for the French+2009 EOS, because the latter does not supply ad_grad
            _, ad_grad = eos.eos_AQUA(pr, tr)
        elif "AVL_planetary_ice" in eos_mode:
            _, ad_grad = eos.avl_planetary_ice(pr, tr, 1, 1, 1, 0) # 1, 1, 1, 0 is just placeholder, ad_grad is assumed to be pure water ad_grad, so has no dependence on mass fractions
        elif "AVL_H2O_HHe" in eos_mode:
            avl_h2o_hhe_split = eos_mode.split()
            x_h2o = float(avl_h2o_hhe_split[1])
            x_hhe = float(avl_h2o_hhe_split[2])
            _, ad_grad = eos.avl_h2o_hhe(pr, tr, x_h2o, x_hhe)
        elif eos_mode == "tab_CD21_HHe":
            _, ad_grad, _ = eos.eos_cd21_hhe(pr, tr)
        else:
            assert ad_grad != 'None', 'Need to supply adiabatic gradient!'
        tn = tpr.adiabatic_t_r_gradAd(tr, mr, pr, rhor, r, ad_grad, delr, tmin=min(teq, 300.0))
    elif t_mode == "two_stream_Guillot2010":
        if tr <= teq: # temperature profile becomes isothermal once equilibrium temperature is reached
            tn = teq
        else:
            assert taun != None, "Optical depth tau must be a number! Currently tau == None."
            tn = tpr.two_stream_temp_g10(teff, teq, taun, f, gamma)
    elif t_mode == "AVL":
        tn = tpr.adiabatic_profile_AVL(pr, tr, mr, rhor, r, mass_fractions, delr, tmin=min(teq, 300.0))
    elif t_mode == "tabulated": # tabulted T(P) profile from previous models for validation
        tn = tpr.tabulated_tp(r)
    else:
        raise ValueError("t_mode is not valid! Check documentation for valid inputs!")

    # temperature profile becomes isothermal once equilibrium temperature is reached
    if tn <= teq:
        tn = teq
    
    if taun == None:
        return mn, pn, rhon, tn, r+delr
    else:
        return mn, pn, rhon, tn, r+delr, taun


def solve_mprho_iter(pc, rhoc_guess, rc, delr, eos_mode, eos_args, ps=1.0e5, 
                     mc=None, rhoc=None, maxsteps=10000, max_mass=None):
    """
    Iteratively solve for mass, pressure, and density in the interior of
    a planet, using two sets of boundary conditions:
        
    Inner boundary conditions
    ----------
    pc : float
        pressure at some small radius, usually delr, in Pa.
    rhoc_guess : float
        a guess value for the core density used for numerically solve for
        the actual density using EOS. Some large value at the order of 1.0e8
        is optimal based  on experience.
    delr : float
        radius step, in m.
    eos_mode : string
        decides which EOS function to use. Valid inputs include "Vinet",
        "BME3", "BME4", "homogeneous [component]", "tabulated [component]"
        where [component] can be "h2o", "mgsio3", or "fe".
    eos_args : list of floats
        arguments for EOS functions. If using Vinet or BME3, the list will
        be [K0, rho0, K0p], if using BME4, the list will be 
        [K0, rho0, K0p, K0pp]
    mc : float or None
        mass of the given inner core to start with. Calculated from rc and
        rhoc if None.
    rhoc : float or None
        density of the given inner core to start with. Calculated from
        EOS if None.
    Note that the density inner BC is calculated from p0.
    
    Outer boundary conditions
    ----------
    ps : float
        Pressure at surface, in Pa. Set to 1.0e5 Pa (1bar) by default.
    maxsteps : int or None
        Maximum steps of iteration. Set to 10000 by default. When it is 
        None, no boundary is placed on the maximum step
    max_mass : float or None
        Maximum mass of the planet/layer in kg. When it is 
        None, no boundary is placed on the maximum mass
        
    Returns
    -------
    mt : float
        Total mass of the planet, in kg.
    rt : float
        Total radius of the planet, in m.
    m_profile : list of floats
        Mass profile of the planet as a function of radius, sorted from
        interior to surface.
    r_profile: list of floats
        Radius array.
    p_profile : list of floats
        Pressure profile as a function of radius.
    rho_profile : list of floats:
        Density profile as a function of radius.
    """
    if "tabulated" not in eos_mode:
        assert len(eos_args) >= 3
        K0 = eos_args[0]
        rho0 = eos_args[1]
        K0p = eos_args[2]
    # if the boundary condition is already satisfied, return central condition
    if pc < ps or (mc != None and max_mass != None and mc > max_mass):
        return mc, rc, [mc]*2, [rc]*2, [pc]*2, [rhoc]*2 # *2 to avoid [-2] index out of range problem in run_solver_downstep
    
    # calculate rhoc from EOS
    if rhoc == None:
        if eos_mode == "Vinet":
            rhoc = eos.eos_Vinet_inverse(K0, pc, rho0, K0p, rhoc_guess)
        elif eos_mode == "BME3":
            rhoc = eos.eos_BME3_inverse(K0, pc, rho0, K0p, rhoc_guess)
        elif eos_mode == "BME4":
            K0pp = eos_args[3]
            rhoc = eos.eos_BME4_inverse(K0, pc, rho0, K0p, K0pp, rhoc_guess)
        elif "homogeneous" in eos_mode:
            rhoc = eos.eos_homogeneous_inverse(pc, eos_mode.split()[1], rhoc_guess)
        elif "tabulated" in eos_mode:
            rhoc = eos.eos_tabulated(pc, eos_mode.split()[1])
        else:
            raise ValueError("eos_mode is not valid! Only 'Vinet', 'BME3' and 'BME4' are legal!")
    
    # calcuate mc from rhoc and volume of the small core
    if mc == None:
        mc = (4.0/3.0) * np.pi * rc ** 3.0 * rhoc
    
    # initialize the output lists with inner boundary conditions
    m_profile = [mc]
    r_profile = [rc]
    p_profile = [pc]
    rho_profile = [rhoc]
    
    mr = mc
    pr = pc
    rhor = rhoc
    r = rc
    step = 0
    # the main loop of iteratively calculate m, p, and rho
    if maxsteps == None:
        while pr > ps and (max_mass == None or mr < max_mass):
            new_step_params = solve_mprho_step(mr, pr, rhor, r, delr, eos_mode, 
                                           eos_args)
            mr = new_step_params[0]
            pr = new_step_params[1]
            rhor = new_step_params[2]
            r = new_step_params[3]
            
            m_profile.append(mr)
            r_profile.append(r)
            p_profile.append(pr)
            rho_profile.append(rhor)
    else:
        while pr > ps and step < maxsteps and (max_mass == None or mr < max_mass):
        # while pr > ps and step < maxsteps and ((max_mass != None and mr < max_mass) or max_mass == None):
            new_step_params = solve_mprho_step(mr, pr, rhor, r, delr, eos_mode, 
                                               eos_args)
            mr = new_step_params[0]
            pr = new_step_params[1]
            rhor = new_step_params[2]
            r = new_step_params[3]
            
            m_profile.append(mr)
            r_profile.append(r)
            p_profile.append(pr)
            rho_profile.append(rhor)
            step += 1
            
            if step >= maxsteps:
                print("Maximum step reached at " + str(step) + " steps")
    
    mt = mr
    rt = r
    return mt, rt, m_profile, r_profile, p_profile, rho_profile


def solve_mprhot_iter(pc, tc, eos_mode, t_mode, rc=10.0, delr=10.0, ps=1.0e5,
                      mc=None, rhoc=None, rhoc_guess=1.0e4, maxsteps=50000, 
                      max_mass=None, mass_fraction_functions=[None, None, None],
                      pradius=None):
    """
    Iteratively solve for mass, pressure, density, and temperature in the 
    interior of a planet, using two sets of boundary conditions:

    Inner boundary conditions
    ----------
    pc : float
        Pressure at the center of the planet.
    tc : float
        Temperature at the center.
    eos_mode : string
        EOS mode. See solve_mprhot_step() for valid inputs.
    t_mode : string
        Temperature mode. See solve_mprhot_step() for valid inputs.
    rc : float, optional
        Radius of the initial center mass. Needed to avoid r=0 singularity.
        The default is 10.0.
    delr : float, optional
        Step size of radius. The default is 10.0.
    mc : float, optional
        Mass of the initial center mass. Not needed when [rc] is supplied
        because it can be calculated from rc, pc, tc, and EOS.
        The default is None.
    rhoc : float, optional
        Density of the initial center mass. Not needed when [rc] is supplied.
        The default is None.
    rhoc_guess : float, optional
        A guess value for the core density used for numerically solve for
        the actual density using EOS. Set to 1.0e4 kg/m3 by default.
    
    Outer boundary conditions
    ----------
    ps : float, optional
        Pressure at the surface. The default is 1.0e5 (1 bar).
    maxsteps : int or None, optional
        Maximum steps of iteration. The default is 50000. When it is 
        None, no boundary is placed on the maximum step
    max_mass : float or None, optional
        Maximum mass of the planet/layer in kg. When it is None, no boundary 
        is placed on the maximum mass. The default is None.
    pradius : float or None, optional
        Total radius of the planet. Used as a termination control when
        running mixture planets with AVL.
        
    mass_fraction_functions : list of three scipy.interp1d functions
        Each of the functions will return a number between 0 and 1 (including
        the endpoints) while given a radius as input. The first function
        represents mass fraction of rock, the second water, and the third H/He.
    
    Returns
    -------
    m_profile : list of floats
        Mass profile of the planet as a function of radius, sorted from
        interior to surface.
    r_profile: list of floats
        Radius array.
    p_profile : list of floats
        Pressure profile as a function of radius.
    t_profile : list of floats
        Temperature profile as a function of radius.
    rho_profile : list of floats:
        Density profile as a function of radius.

    """
    # print('642', mc, pc, tc, rhoc)
    
    # calculate rhoc from EOS if it is not supplied
    if rhoc == None:
        if "homogeneous" in eos_mode:
            rhoc = eos.eos_homogeneous_inverse(pc, eos_mode.split()[1], rhoc_guess)
        elif "TFD" in eos_mode: 
            assert t_mode == "isothermal", "TFD EOS requires isothermal (i.e. zero temperature) T profile!"
            if "HHe" in eos_mode: # "TFD HHe", special case for zero temperature H/He mixture
                rhoc = eos.eos_TFD_SZ67_inverse(1.2598425196850394, 1.08661417, pc) # assuming He mass fraction Y=0.275 (solar); He number fraction is 0.08661417
            elif "pureH" in eos_mode:
                rhoc = eos.eos_TFD_SZ67_inverse(1.00784, 1.0, pc) # pure hydrogen
            else:
                raise ValueError("Material not supported for TFD EOS!")
        elif "tab_Seager07" in eos_mode or "tab_Zeng16" in eos_mode or "tab_Zeng21" in eos_mode or \
            "OganovOno2004" in eos_mode or "HM89_rock" in eos_mode or "tabulated" in eos_mode:
            rhoc = eos.eos_tabulated(pc, eos_mode.split()[1])
        elif eos_mode == "tab_AQUA":
            rhoc, _ = eos.eos_AQUA(pc, tc)  # drop ad_grad because it is calculated in solve_mprhot_step()
        elif eos_mode == "tab_French2009_H2O":
            rhoc = eos.eos_french2009_h2o(pc, tc)
        elif "AVL_planetary_ice" in eos_mode:
            avl_mixed_ice_split = eos_mode.split()
            x_h2o = float(avl_mixed_ice_split[1])
            x_ch4 = float(avl_mixed_ice_split[2])
            x_nh3 = float(avl_mixed_ice_split[3])
            if len(avl_mixed_ice_split) > 4:
                x_hhe = float(avl_mixed_ice_split[4])
            else:
                x_hhe = 0.0
            rhoc, _ = eos.avl_planetary_ice(pc, tc, x_h2o, x_ch4, x_nh3, x_hhe)
        elif "AVL_H2O_HHe" in eos_mode:
            avl_h2o_hhe_split = eos_mode.split()
            x_h2o = float(avl_h2o_hhe_split[1])
            x_hhe = float(avl_h2o_hhe_split[2])
            rhoc, _ = eos.avl_h2o_hhe(pc, tc, x_h2o, x_hhe)
        elif eos_mode == "tab_CD21_HHe":
            rhoc, _, _ = eos.eos_cd21_hhe(pc, tc)
        elif eos_mode == "AVL":
            # initialize mass fractions
            mfc_rock = mass_fraction_functions[0](rc) # mass fraction of rock at the center
            mfc_water = mass_fraction_functions[1](rc) # mass fraction of water at the center
            mfc_hhe = mass_fraction_functions[2](rc) # mass fraction of H/He at the center
            rhoc = eos.avl_eos(pc, tc, [mfc_rock, mfc_water, mfc_hhe])
        elif eos_mode == "iron_with_phase_change": # iron EOS that transitions from liquid to solid
            if eos.is_liquid_Kraus2022(pc, tc):
                rhoc = eos.eos_tabulated(pc, "Grant2021_liquidFe")
            else:
                rhoc = eos.eos_tabulated(pc, "Smith2018_solidFe")
        elif "polytrope" in eos_mode: # the format should be "polytrope K n", e.g., "polytrope 0.3 1"
            poly_K = float(eos_mode.split()[1])
            poly_n = int(eos_mode.split()[2])
            rhoc = (pc / poly_K) ** (poly_n/(poly_n+1))
        else:
            raise ValueError("eos_mode is not valid! Check documentation for valid inputs!")
    
    # calcuate mc from rhoc and volume of the small core, if mc is not supplied
    if mc == None:
        mc = (4.0/3.0) * np.pi * rc ** 3.0 * rhoc
    
    # print('659', mc, rhoc)    
    
    # if the boundary condition is already satisfied, return central condition
    if pc < ps or (mc != None and max_mass != None and mc > max_mass):
        return [mc], [rc], [pc], [tc], [rhoc]
    
    # get thermal parameters if t_mode == "adiabatic_alpha_cp" or "adiabatic_Boujibar_2020"
    # note that if t_mode == "adiabatic_gradAd", ad_grad will be computed
    # in solve_mprhot_step() so don't need to be computed here.
    alpha = None; cp = None
    rho0 = None; K0 = None; K0p = None; gamma0 = None; gamma1 = None; alpha0 = None
    if t_mode == "adiabatic_alpha_cp":
        if eos_mode == "tab_AQUA":
            raise ValueError("Use tmode = adiabatic_gradAd when using AQUA EOS!")
        else:
            # determine what's the material based on EOS mode
            if 'tab_Zeng' in eos_mode:
                material = 'fe' if 'core' in eos_mode else 'mgsio3'
            elif "OganovOno2004" in eos_mode or "HM89_rock" in eos_mode: # the HM89 material is not strictly MgSiO3, though
                material = "mgsio3"
            elif eos_mode == "iron_with_phase_change":
                material = 'fe'
            else:
                material = eos_mode.split()[1]
            # choose the appropriate alpha and cp
            if material == "h2o":
                alpha = alpha_h2o; cp = cp_h2o
            elif material == "fe":
                alpha = alpha_fe; cp = cp_fe
            elif material == "mgsio3":
                alpha = alpha_mgsio3; cp = cp_mgsio3
            else:
                raise ValueError("Invalid component!")   
    elif t_mode == "adiabatic_Boujibar_2020":
        if eos_mode == "tab_AQUA":
            raise ValueError("Use tmode = adiabatic_gradAd when using AQUA EOS!")
        else:
            # determine what's the material based on EOS mode
            if 'tab_Zeng' in eos_mode:
                material = 'fe' if 'core' in eos_mode else 'mgsio3'
            elif "OganovOno2004" in eos_mode or "HM89_rock" in eos_mode:
                material = "mgsio3"
            elif eos_mode == "iron_with_phase_change":
                material = 'fe'
            else:
                material = eos_mode.split()[1]
            # choose the appropriate rho0, K0, K0', gamma0, gamma1, and alpha0 parameters
            # use values from Table 1 in Boujibar et al. (2020)
            # use ppv values for the entire MgSiO3 layer
            # (not entirely physical, but the parameters are not too different between phases)
            if material == "fe":
                if eos.is_liquid_Kraus2022(pc, tc):
                    rho0 = 7700.0; K0 = 125.0; K0p = 5.5; gamma0 = 1.6; gamma1 = 0.92; alpha0 = 40.0  # liquid Fe parameters, Wicks+2018
                else:
                    rho0 = 8160.0; K0 = 165.0; K0p = 4.9; gamma0 = 1.6; gamma1 = 0.92; alpha0 = 40.0  # solid Fe parameters, Smith+2018
            elif material == "mgsio3":
                rho0 = 4260.0; K0 = 324.0; K0p = 3.3; gamma0 = 1.48; gamma1 = 1.4; alpha0 = 20.0 # ppv, Sakai+2016
            else:
                raise ValueError("Invalid component!")
    
    # initialize the output lists with inner boundary conditions
    m_profile = [mc]
    r_profile = [rc]
    p_profile = [pc]
    t_profile = [tc]
    rho_profile = [rhoc]
    # print('633', type(mc), type(rc), type(pc), type(tc), type(rhoc))
    
    mr = mc
    r = rc
    pr = pc
    tr = tc
    rhor = rhoc
    step = 0
    
    # Check if either of the pressure, mass, or max steps BC is met
    # If eos_mode == "AVL", stop iterating when r reaches Rp
    def check_terminate(pr, ps, maxsteps, step, mr, max_mass, eos_mode, r):
        if maxsteps == None and max_mass == None:
            terminate = pr < ps
        elif maxsteps == None and max_mass != None:
            terminate = pr < ps or mr > max_mass
        elif maxsteps != None and max_mass == None:
            terminate = pr < ps or step > maxsteps
        else:
            terminate = pr < ps or mr > max_mass or step > maxsteps
            
        if eos_mode == "AVL":
            terminate = r >= pradius
            
        return terminate
    
    terminate = check_terminate(pr, ps, maxsteps, step, mr, max_mass, eos_mode, r)
    
    # the main loop of iteratively calculate m, p, t, and rho
    while not terminate:    
        # check for AVL mass fractions        
        if mass_fraction_functions[0] != None:
            mfr_rock = mass_fraction_functions[0](r) # mass fraction of rock at this layer
            mfr_water = mass_fraction_functions[1](r) # mass fraction of water at this layer
            mfr_hhe = mass_fraction_functions[2](r) # mass fraction of H/He at this layer
        else:
            mfr_rock = 0.33; mfr_water = 0.33; mfr_hhe = 0.34 # fall back to default values

        # check for iron phase and change the thermal parameters accordingly
        if eos_mode == "iron_with_phase_change":
            if eos.is_liquid_Kraus2022(pr, tr):
                rho0 = 7700.0; K0 = 125.0; K0p = 5.5; gamma0 = 1.6; gamma1 = 0.92; alpha0 = 40.0  # liquid Fe parameters, Wicks+2018
            else:
                rho0 = 8160.0; K0 = 165.0; K0p = 4.9; gamma0 = 1.6; gamma1 = 0.92; alpha0 = 40.0  # solid Fe parameters, Smith+2018

        # main function to calculate planet parameters
        mr, pr, rhor, tr, r = solve_mprhot_step(mr, pr, rhor, tr, r, delr,
                                                eos_mode, t_mode, alpha=alpha, cp=cp,
                                                rho0=rho0, K0=K0, K0p=K0p, 
                                                gamma0=gamma0, gamma1=gamma1, alpha0=alpha0,
                                                mass_fractions=[mfr_rock, mfr_water, mfr_hhe])
        m_profile.append(mr)
        r_profile.append(r)
        p_profile.append(pr)
        t_profile.append(tr)
        rho_profile.append(rhor)
        step += 1
        terminate = check_terminate(pr, ps, maxsteps, step, mr, max_mass, eos_mode, r)
    
    # remove negative pressures from the end of the list
    try:
        neg_ind = next(x for x, val in enumerate(p_profile) if val < 0.0)
    except StopIteration:
        neg_ind = len(p_profile)
        
    m_profile = m_profile[:neg_ind]
    r_profile = r_profile[:neg_ind]
    p_profile = p_profile[:neg_ind]
    t_profile = t_profile[:neg_ind]
    rho_profile = rho_profile[:neg_ind]
    
    print("Iteration stopped at step %d, P = %f bar, M = %f M_Earth, R = %f R_Earth" \
          % (step, p_profile[-1]/1.0e5, m_profile[-1]/Me, r_profile[-1]/Re))
    
    return m_profile, r_profile, p_profile, t_profile, rho_profile


def solve_mprhot_iter_hhe_gas(pc, tc, eos_mode, Mp, Rp, rc=10.0, delr=10.0, ps=1.0e3,
                              mc=None, rhoc=None, mmw=3.6087305699338575e-27,
                              maxsteps=50000, teff=50.0, teq=300.0, 
                              f=0.5, gamma=1.0, is_above_rcb=False):
    """ Largely the same as solve_mprhot_iter, but has two addition functions:
    (1) Track the change of specific entropy ds/dr, by comparing s(r) and s(r+delr). If ds/dr changes
      sign from > 0 to < 0, that means we are transitioning from the convective regime to the radiative
      regime. When in radiative regime, use adiabatic temperature profile. When in convective regime, use
      temperature_profile.two_stream_temp_g10()
    (2) Check for the outer pressure boundary condition p_R where opacitiy tau_R becomes 1, using equations
      7-15 in Rogers & Seager (2010).
      
    - max_mass is removed.
    - ps default value reduced to 1.0e3 Pa (0.01 bar) given that atmospheres are not necessarily transparent
    at 1 bar.
    - Mp and Rp are added and are not optional. They are the total planetary mass and radius. Mp and Rp
    are required to calculate g, which is required to calculate the pressure outer boundary condition.
    They should be in SI units (kg and m).
    - rhoc_guess is removed.
    - mmw is added as a variable. It represents the mean molecular mass (in kg) of the H/He envelope.
    Default value is 3.6087305699338575e-27 (1.66053906660e-27 kg * 1.2598425196850394 amu), which is the MMW of 
    H/He mixture assuming Y = 0.275 (helium cosmogonic abundance).
    - f and gamma (optional) are added for computing Gillion (2010) analytic temperature profile.
    - is_above_rcb (optional) is added to flag whether we are above the radiative-convective boundary, i.e.,
    has ds/dr crossed zero. When ds/dr changes sign from > 0 to < 0, change to True. While true, skip
    checking for ds/dr and use radiative temperature profile for atmosphere above.
    """
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
    
    
    # Calculate rhoc from EOS if it is not supplied. Currently only "tab_CD21_HHe" and "AVL_H2O_HHe" are supported.
    if rhoc == None:
        if eos_mode == "tab_CD21_HHe":
            rhoc, _, _ = eos.eos_cd21_hhe(pc, tc)
        elif "AVL_H2O_HHe" in eos_mode:
            # eos_mode should be in the format e.g. "AVL_H2O_HHe 90 10"
            avl_h2o_hhe_split = eos_mode.split()
            x_h2o = float(avl_h2o_hhe_split[1])
            x_hhe = float(avl_h2o_hhe_split[2])
            rhoc, _ = eos.avl_h2o_hhe(pc, tc, x_h2o, x_hhe)
        else:
            raise ValueError("eos_mode is not valid! Check documentation for valid inputs!")
    
    # Calcuate mc from rhoc and volume of the small core, if mc is not supplied
    # This should be rare for the H/He layer - why would anyone what to simulate a gas ball?
    # Print a warning when mc is not supplied for H/He layer
    if mc == None:
        print("WARNING! Central mass should be supplied for H/He gas envelope. Pure gas planet is not physical.")
        mc = (4.0/3.0) * np.pi * rc ** 3.0 * rhoc
        
    # initialize tauc (optical depth at center) if is_above_rcb
    if is_above_rcb:
        tauc = tpr.solve_tau_two_stream_temp_g10(tc, teff, teq, f, gamma)
    else:
        tauc = None
        
    # Check for boundary conditions. Terminate and return central conditions when either condition is satisfied:
    # (1) pc < ps
    # (2) pc < 3.0e7 Pa (300 bar), and pc < pR computed with outer_bc_pressure(); 300 bar is the max pressure
    #     of the kappa database from Freedman et al. (2008)
    # (3) step > maxstep
    def check_terminate(pr, ps, maxsteps, step, t, mmw=3.6087305699338575e-27, gamma=1.0):
        p_outer_bc = outer_bc_pressure(Mp, Rp, t, mmw, gamma)
        if ps < p_outer_bc:
            p_outer_bc = ps # override p_outer_bc if the user-supplied ps is smaller
        # if type(pr) != type(ps):
        #     print(type(pr), type(ps), pr, ps)
        cond1 = pr < ps
        cond2 = False
        if pr < 1.0e7:
            cond2 = pr < p_outer_bc
        cond3 = False
        if maxsteps != None:
            cond3 = step > maxsteps
        terminate = cond1 or cond2 or cond3 # terminate if outer BCs are reached
        return terminate

    terminate = check_terminate(pc, ps, maxsteps, 0, tc, mmw, gamma)
    if terminate:
        return [mc], [rc], [pc], [tc], [rhoc], [None], [None] # return nothing for tau and ds/dr if outer BC is already met at the bottom of H/He layer
    
    # initialize the output lists with inner boundary conditions
    m_profile = [mc]
    r_profile = [rc]
    p_profile = [pc]
    t_profile = [tc]
    rho_profile = [rhoc]
    tau_profile = [tauc]
    dsdr_profile = [0] # a list keeping track of the sign of ds/dr, 0 is negative (convective) and 1 positive (radiative)
    
    mr = mc
    r = rc
    pr = pc
    tr = tc
    rhor = rhoc
    taur = tauc # initialize optical depth as None; once is_above_rcb == True, numerical value will be calculated
    step = 0
    
    # the main loop of iteratively calculate m, p, t, and rho
    while not terminate:
        tau_profile.append(taur)
        if is_above_rcb:
            try:
                mr, pr, rhor, tr, r, taur = solve_mprhot_step(mr, pr, rhor, tr, r, delr, eos_mode, 
                                                              'two_stream_Guillot2010',
                                                              taur=taur, teff=teff, teq=teq, f=f, gamma=gamma)
            # print(step, pr, tr, taur)
            except ValueError:
                print("ValueError encountered at step %d, check the following numbers:" % step)
                print('mr, pr, rhor, tr, r, delr, eos_mode, teff, teq, f, gamma')
                print(mr, pr, rhor, tr, r, delr, eos_mode, teff, teq, f, gamma)
                return
        else:
            if taur == None:
                mr, pr, rhor, tr, r = solve_mprhot_step(mr, pr, rhor, tr, r, delr, eos_mode, 'adiabatic_gradAd',
                                                        taur=taur, teff=teff, teq=teq, f=f, gamma=gamma)
            else:
                mr, pr, rhor, tr, r, _ = solve_mprhot_step(mr, pr, rhor, tr, r, delr, eos_mode, 'adiabatic_gradAd',
                                                           taur=taur, teff=teff, teq=teq, f=f, gamma=gamma)
        m_profile.append(mr)
        r_profile.append(r)
        p_profile.append(pr)
        t_profile.append(tr)
        rho_profile.append(rhor)
        step += 1
        terminate = check_terminate(pr, ps, maxsteps, step, tr, mmw, gamma)
        
        # calculate ds/dr, note that this step can significantlly slow the code down, need to find a workaround
        if eos_mode == "tab_CD21_HHe" or "AVL_H2O_HHe" in eos_mode:
            _, _, s0 = eos.eos_cd21_hhe(p_profile[-2], t_profile[-2]) # s(r)
            _, _, s1 = eos.eos_cd21_hhe(pr, tr) # s(r+delr)
            dels = s1 - s0
            ds_dr = dels / delr
            dsdr_profile.append(0 if ds_dr < 0 else 1)
        else:
            raise ValueError("eos_mode is not valid! Check documentation for valid inputs!")
    
    print("Iteration stopped at step %d, P = %f bar, M = %f M_Earth, R = %f R_Earth" \
          % (step, pr/1.0e5, mr/Me, r/Re))
    
    return m_profile, r_profile, p_profile, t_profile, rho_profile, tau_profile, dsdr_profile


def hhe_gas_find_rcb_solve(pc, tc, eos_mode, Mp, Rp, rc=10.0, delr=10.0, ps=1.0e3,
                           mc=None, rhoc=None, mmw=3.6087305699338575e-27,
                           maxsteps=50000, teff=50.0, teq=300.0, 
                           f=0.5, gamma=1.0, hhe_bc_mode='mass'):
    """ Calls solve_mprhot_iter_hhe_gas() to solve for the H/He envelope of a planet. However it
    has two steps:
    (1) calculating the H/He layer assuming the entire layer is convective, and find the lowest pressure such
        that ds/dr < 0 (which we assume to be the "true radiative-convective boundary (RCB)").
    (2) once we know the RCB conditions, initialize another run using the planet below RCB as core, and compute
        the radiative part of the H/He envelope.
        
    [hhe_bc_mode] is a parameter defining the outer BC of the atmosphere. 'opacity' will ignore the mass limit
    and terminates where tau reaches 1. 'mass' will force the function to return layers below the upper mass
    limit. 'ps' will ignore both opacity and mass BCs and calculate until the minimum pressure is reached.
    """
    # first, calculate the entire H/He envelope assuming it is fully convective
    print("Calculating H/He envelope (convective layer) ...")
    begin_time = datetime.datetime.now()
    ms_con, rs_con, ps_con, ts_con, rhos_con, taus_con, dsdrs_con =\
        solve_mprhot_iter_hhe_gas(pc, tc, eos_mode, Mp, Rp, rc, delr, ps, mc, rhoc, mmw, maxsteps, teff, teq, 
                                  f, gamma, is_above_rcb=False)
    print("H/He envelope (convective layer) simulation completed.")
    print("Execution time: ", datetime.datetime.now() - begin_time)
    # print()
    # find the lowest pressure such that ds/dr < 0, i.e. the "true RCB"
    dsdrs_con.reverse()
    skip_rad = False
    try:
        index = dsdrs_con.index(0)
    except ValueError:
        index = 0 # when ds/dr never crosses 0 (fully convective atmosphere), return 0, or the layer at the TOA
        skip_rad = True
    index = len(dsdrs_con) - index - 1 # index of the last 0 in dsdrs
    pc = ps_con[index]
    tc = ts_con[index]
    rc = rs_con[index]
    mc = ms_con[index]
    rhoc = rhos_con[index]
    print("The RCB is at P = %f bar" % (pc/1.0e5))
    # print(index, pc, tc, rc, mc, rhoc)
    # print()
    
    if skip_rad:
        ms_rad, rs_rad, ps_rad, ts_rad, rhos_rad, taus_rad = [mc], [rc], [pc], [tc], [rhoc], [None]
    else:
        # then, calculate the radiative layer
        print("Calculating H/He envelope (radiative layer) ...")
        begin_time = datetime.datetime.now()
        ms_rad, rs_rad, ps_rad, ts_rad, rhos_rad, taus_rad, _ = solve_mprhot_iter_hhe_gas(pc, tc, eos_mode, Mp, Rp, rc, 
                                                                                        delr, ps, mc, rhoc, mmw, 
                                                                                        maxsteps, teff, teq, f, gamma, 
                                                                                        is_above_rcb=True)
        print("H/He envelope (radiative layer) simulation completed.")
        print("Execution time: ", datetime.datetime.now() - begin_time)

    # patch the convective layer output and radiative layer output together
    ms_out = ms_con[:index] + ms_rad # profile[:index] is the profile for the convective part
    rs_out = rs_con[:index] + rs_rad
    ps_out = ps_con[:index] + ps_rad
    ts_out = ts_con[:index] + ts_rad
    rhos_out = rhos_con[:index] + rhos_rad
    
    # finally, sum the kappa/g from TOA towards the lower atmosphere in pressure coordinates, and find
    # where tau reaches 1, and only return the atmosphere below tau=1
    taus_from_above = 0.0 # tau at the TOA is set to be 0
    index_tau_eq_1 = 0 # index where optical depth crosses 1
    ms_out.reverse()
    rs_out.reverse()
    ps_out.reverse()
    ts_out.reverse()
    rhos_out.reverse()
    for i in range(len(ps_out)-1):
        delta_p = ps_out[i+1] - ps_out[i]
        m_mid = (ms_out[i+1] + ms_out[i])/2.0
        r_mid = (rs_out[i+1] + rs_out[i])/2.0
        p_mid = (ps_out[i+1] + ps_out[i])/2.0
        t_mid = (ts_out[i+1] + ts_out[i])/2.0
        g_mid = G * m_mid / r_mid**2.0 # gravitational acceleration at this differential layer
        kappa_mid = tpr.freedman2008_mean_opacity(p_mid, t_mid)
        taus_from_above += kappa_mid/g_mid * delta_p # tau at the lower boundary of this differential layer
        # print(p_mid, kappa_mid, taus_from_above)
        if taus_from_above >= 1:
            index_tau_eq_1 = i
            break
    if len(ps_out) > 0:
        print('Optical depth = 1 at %f bar' % (ps_out[index_tau_eq_1]/1.0e5))
    else: # there is no atmosphere, i.e., boundary condition already met at layers below
        print('No atmosphere. Outer boundary conditions met at layers below.')
    
    if hhe_bc_mode != 'ps':
        ms_out = ms_out[index_tau_eq_1:]
        rs_out = rs_out[index_tau_eq_1:]
        ps_out = ps_out[index_tau_eq_1:]
        ts_out = ts_out[index_tau_eq_1:]
        rhos_out = rhos_out[index_tau_eq_1:]
    ms_out.reverse()
    rs_out.reverse()
    ps_out.reverse()
    ts_out.reverse()
    rhos_out.reverse()
    taus_out = [None] * len(ms_out) # dummy list; remains here in case we want to calculate tau into the deep atmosphere and return a tau profile
    
    # if hhe_bc_mode == 'mass', check if the calculated mass exceeds given mass; if exceeds, truncate the list
    # and only return those with M(r) < Mp
    if hhe_bc_mode == 'mass':
        if len(ms_out) > 0 and ms_out[-1] > Mp: # mass outer BC exceeded
            m_bc_ind = next(x for x, val in enumerate(ms_out) if val > Mp)
            print("Calculated mass greater than input mass, upper atmosphere truncated.")
            return ms_out[:m_bc_ind], rs_out[:m_bc_ind], ps_out[:m_bc_ind], ts_out[:m_bc_ind], rhos_out[:m_bc_ind], taus_out[:m_bc_ind]
        else: # mass outer BC not exceeded, return normally
            return ms_out, rs_out, ps_out, ts_out, rhos_out, taus_out
    else:    
        return ms_out, rs_out, ps_out, ts_out, rhos_out, taus_out


def run_solver_downstep(mc, pc, rhoc, rhoc_guess, rc, delr,
                        eos_mode, eos_args, max_solver_steps,
                        max_decrease_steps, decrease_factor,
                        max_mass=None, ps=1.0e5):
    """
    Run the mprho solver with decreasing r step.

    Parameters
    ----------
    mc : float
        Core mass in kg.
    pc : float
        Core pressure in Pa.
    rhoc : float
        Core density in kg m-3.
    rhoc_guess : float
        Density estimate for numerically solving EOS.
    rc : float
        Core radius in m.
    delr : float
        Radius step.
    eos_mode : string
        Decides which EOS function to use. Valid inputs include "Vinet",
        "BME3", "BME4".
    eos_args : list of floats
        Arguments for EOS functions. If using Vinet or BME3, the list will
        be [K0, rho0, K0p], if using BME4, the list will be 
        [K0, rho0, K0p, K0pp]
    max_solver_steps : int
        Maximum number of steps of the solver.
    max_decrease_steps : int
        Maximum number of steps this function attempts to decrease delr.
    decrease_factor : float
        Divide delr by this number each step.
    max_mass : float
        Surface boundary condition. Maximum mass of integration in kg.
        If m > max_mass, stop calculation. If max_mass = None, do nothing.
    ps : float
        Surface boundary condition. Surface pressure in Pa. If p < ps, stop
        calculation.

    Returns
    -------
    mt, rt, m_profile, r_profile, p_profile, rho_profile.
    The same as solve_mprho_iter()
    """
    if pc < ps or (mc != None and max_mass != None and mc > max_mass):
        return mc, rc, [mc], [rc], [pc], [rhoc] if rhoc != None else [mc/((4.0/3.0)*np.pi*rc**3.0)]

    decrease_steps = 0
    
    # run the first iteration
    mt, rt, m_profile, r_profile, p_profile, rho_profile = \
            solve_mprho_iter(pc, rhoc_guess, rc, delr, eos_mode, eos_args,
                             ps=ps, mc=mc, rhoc=rhoc, maxsteps=max_solver_steps,
                             max_mass=max_mass)
    decrease_steps += 1
    
    while decrease_steps < max_decrease_steps:
        # terminate while either boundary condition is met
        # if p_profile[-1] < ps or (max_mass != None and mt > max_mass):
        #     return mt, rt, m_profile, r_profile, p_profile, rho_profile
        
        if len(p_profile) >= 2:  # [-2] is necessary becase the code sometimes overshoot with [-1]
            mt, rt, m_profile_s, r_profile_s, p_profile_s, rho_profile_s = \
                solve_mprho_iter(p_profile[-2], rho_profile[-2], r_profile[-2], 
                                  delr / decrease_factor, eos_mode, eos_args,
                                  ps=ps, mc=m_profile[-2], rhoc=rho_profile[-2], 
                                  maxsteps=max_solver_steps, max_mass = max_mass)
        else:
            mt, rt, m_profile_s, r_profile_s, p_profile_s, rho_profile_s = \
                solve_mprho_iter(p_profile[-1], rho_profile[-1], r_profile[-1], 
                                  delr / decrease_factor, eos_mode, eos_args,
                                  ps=ps, mc=m_profile[-1], rhoc=rho_profile[-1], 
                                  maxsteps=max_solver_steps, max_mass=max_mass)
            
        m_profile = m_profile + m_profile_s
        r_profile = r_profile + r_profile_s
        p_profile = p_profile + p_profile_s
        rho_profile = rho_profile + rho_profile_s
        decrease_steps += 1
        
        if p_profile_s[-1] < ps or (max_mass != None and mt > max_mass):
            return mt, rt, m_profile, r_profile, p_profile, rho_profile
        
    return mt, rt, m_profile, r_profile, p_profile, rho_profile
        
    
def run_plot_homogeneous():
    """
    Run and plot the M-R relation of homogeneous planets given a list of 
    central pressures to start with.

    Returns
    -------
    None.

    """    
    mlist_silicate = []
    rlist_silicate = []
    
    mlist_waterice = []
    rlist_waterice = []
    
    mlist_iron = []
    rlist_iron = []
    
    # pc_list = list(np.logspace(10,12,20))
    pc_list = list(np.logspace(10,21,100))
    # pc_list = list([1.0e12, 1.0e16])

    # runs for homogeneous silicate planet
    for pc in pc_list:
        mt, rt, m_profile, r_profile, p_profile, rho_profile = \
            run_solver_downstep(mc=None, pc=pc, rhoc_guess=1.0e4, rhoc=None,
                                rc=10.0, delr=1000.0, 
                                eos_mode='tabulated mgsio3', eos_args=[],
                                max_solver_steps=10 ** 6, max_decrease_steps=5, 
                                decrease_factor=100.0)
            # run_solver_downstep(mc=None, pc=pc, rhoc_guess=1.0e4, rhoc=None,
            #                     rc=10.0, delr=1000.0, 
            #                     eos_mode='homogeneous mgsio3', eos_args=[125.0e9, 3.22e3, 5.0],
            #                     max_solver_steps=10 ** 6, max_decrease_steps=5, 
            #                     decrease_factor=100.0)
        mlist_silicate.append(mt)
        rlist_silicate.append(rt)
        print("Silicate")
        print("M (kg)\t", mt)
        print("R\t", rt)
        print("P (Pa)\t", pc)
        print(p_profile[-2])
        print()
    
    # runs for homogeneous water ice planet
    for pc in pc_list:
        mt, rt, m_profile, r_profile, p_profile, rho_profile = \
            run_solver_downstep(mc=None, pc=pc, rhoc_guess=1.0e4, rhoc=None,
                                rc=10.0, delr=1000.0, 
                                eos_mode='tabulated h2o', eos_args=[],
                                max_solver_steps=10 ** 6, max_decrease_steps=5, 
                                decrease_factor=100.0)
            # run_solver_downstep(mc=None, pc=pc, rhoc_guess=1.0e4, rhoc=None,
            #                     rc=10.0, delr=1000.0, 
            #                     eos_mode='homogeneous h2o', eos_args=[23.7e9, 1.46e3, 4.15],
            #                     max_solver_steps=10 ** 6, max_decrease_steps=5, 
            #                     decrease_factor=100.0)
        mlist_waterice.append(mt)
        rlist_waterice.append(rt)
        print("Water ice")
        print("M (m)\t", mt)
        print("R\t", rt)
        print("P (Pa)\t", pc)
        print(p_profile[-2])
        print()
    
    for pc in pc_list:
        mt, rt, m_profile, r_profile, p_profile, rho_profile = \
            run_solver_downstep(mc=None, pc=pc, rhoc_guess=1.0e4, rhoc=None,
                                rc=10.0, delr=1000.0, 
                                eos_mode='tabulated fe', eos_args=[],
                                max_solver_steps=10 ** 6, max_decrease_steps=5, 
                                decrease_factor=100.0)
            # run_solver_downstep(mc=None, pc=pc, rhoc_guess=1.0e4, rhoc=None,
            #                     rc=10.0, delr=1000.0, 
            #                     eos_mode='homogeneous fe', eos_args=[162.5e9, 7.86e3, 5.5],
            #                     max_solver_steps=10 ** 6, max_decrease_steps=5, 
            #                     decrease_factor=100.0)
        mlist_iron.append(mt)
        rlist_iron.append(rt)
        print("Iron")
        print("M (m)\t", mt)
        print("R\t", rt)
        print("P (Pa)\t", pc)
        # print(p_profile[-2])
        print()
        
    # convert into Earth units
    mlist_silicate = list(map(lambda x: x / Me, mlist_silicate))
    mlist_waterice = list(map(lambda x: x / Me, mlist_waterice))
    mlist_iron = list(map(lambda x: x / Me, mlist_iron))
    rlist_silicate = list(map(lambda x: x / Re, rlist_silicate))
    rlist_waterice = list(map(lambda x: x / Re, rlist_waterice))
    rlist_iron = list(map(lambda x: x / Re, rlist_iron))
    
    # save M and R of each planet as a text file
    with open('mass_vs_radius_homogeneous.txt', 'a') as outfile:
        outfile.write("Three columns are:\n")
        outfile.write("Central pressure (Pa)\tMass(Me)\tRadius(Re)\n\n")
        outfile.write("silicate M-R\n")
        for i in range(len(mlist_silicate)):
            outfile.write("{:.6e}".format(pc_list[i]) + "\t" + str(mlist_silicate[i]) + "\t" + str(rlist_silicate[i]) + '\n')
        outfile.write('\n')
        outfile.write("water ice M-R\n")
        for i in range(len(mlist_waterice)):
            outfile.write("{:.6e}".format(pc_list[i]) + "\t" + str(mlist_waterice[i]) + "\t" + str(rlist_waterice[i]) + '\n')
        outfile.write('\n')
        outfile.write("iron M-R\n")
        for i in range(len(mlist_iron)):
            outfile.write("{:.6e}".format(pc_list[i]) + "\t" + str(mlist_iron[i]) + "\t" + str(rlist_iron[i]) + '\n')
    
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(mlist_silicate, rlist_silicate, 'rx', label="silicate")
    plt.plot(mlist_waterice, rlist_waterice, 'bx', label="water ice")
    plt.plot(mlist_iron, rlist_iron, 'gx', label="iron")
    plt.xlabel('M (Me)', fontsize=16, **hfont)
    plt.ylabel('R (Re)', fontsize=16, **hfont)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig("MR_homogeneous.pdf", bbox_inches="tight")
    plt.savefig("MR_homogeneous.png", bbox_inches="tight", dpi=200)
    plt.show()
        

def run_single_planet_multilayer(mass, layers, layer_mass_fractions, 
                                 pc=330.0e9, ps=1.0e5, rhoc_guess=1.0e4, 
                                 rc=10.0, delr=100.0,
                                 plot_profiles=False,
                                 plot_prem=False,
                                 save_figures=False,
                                 pname='single_multilayer_planet'):
    """
    Run a single planet with multiple layers, assuming we know its mass.

    Parameters
    ----------
    mass : float
        Total mass of the planet in kg.
    layers : list of strings
        A list of layer compositions, ranking from the heaviest (central) to
        the lightest (exterior). For example, a valid list is:
            ["tabulated fe", "tabulated mgsio3", "tabulated h2o"]
        note that layer names should be valid tabulated eos_mode
    layer_mass_fractions : list of int
        A list of integer that sums to 1,000,000 that represent the mass
        fraction of each layer. Using int instead of float here to avoid
        float issues (such as not summing exactly to 1). Parts per million
        precision should be more than enough and can be increased if necessary.
            Note that len(layer_mass_fractions) must equal len(layers).
            In addition, mass fraction of a layer should be nonzero. In the 
        future we may want to update the code to allow 0 fraction so that the
        code can solve for e.g. pure water planet case. But right now for
        simplicity let's not allow mass fraction to be 0.
    pc : float, optional
        Central pressure in Pa. The default is 330.0e9 (estimated Earth central pressure).
    ps : float, optional
        Surface pressure outer boundary condition in Pa. The default is 1.0e5 (1 bar).
    rhoc_guess : float, optional
        Guess of central density. The default is 1.0e4.
    rc : float, optional
        Central radius. The default is 10.0.
    delr : float, optional
        Starting step size of radius. The default is 1000.0.
    plot_profiles : bool, optional
        Plot M(r), P(r), rho(r) or not. The default is False.
    plot_prem : bool, optional
        Overplot PERM (Preliminary Reference Earth Model) or not. Only
        relevant to Earth-like planets.
    pname : string, optional
        Name of the planet. Will be used as saved figure file names.

    Returns
    -------
    - total mass of the planet (float, same as input mass)
    - calculated total mass of the planet (float, to see if there's deviation)
    - total radius of the planet (float)
    - list of M(r)
    - list of radius at each step
    - list of P(r)
    - list of rho(r)
    - radius at the boundaries
    - masses at the boundaries
    - pressures at the boundaries
    
    density at the boundary are not returned due to discontinuity
    
    This program also plots M(r), P(r), rho(r)

    """
    assert len(layers) > 0, "There should at least be 1 layer!"
    # force mass fraction to be 100% if there's only 1 layer
    if len(layers) == 1:
        layer_mass_fractions = [1000000]
    assert len(layers) == len(layer_mass_fractions)
    assert sum(layer_mass_fractions) == 1000000
    
    mlist = []
    rlist = []
    plist = []
    rholist = []
    mboundary = []
    rboundary = []
    pboundary = []
    
    # calculate the central layer (core)
    m0, r0, m_profile0, r_profile0, p_profile0, rho_profile0 = \
        run_solver_downstep(mc=None, pc=pc, rhoc_guess=1.0e4, rhoc=None,
                            rc=10.0, delr=delr, 
                            eos_mode=layers[0], eos_args=[],
                            max_solver_steps=10 ** 6, max_decrease_steps=5, 
                            decrease_factor=1.0, ps=ps,
                            max_mass=mass*layer_mass_fractions[0]/1000000.0)
    mlist = m_profile0
    rlist = r_profile0
    plist = p_profile0
    rholist = rho_profile0
    mboundary.append(m0)
    rboundary.append(r0)
    pboundary.append(p_profile0[-1])
    # return results if there's only 1 layer
    if len(layers) == 1:
        if plot_profiles:
            pltmr.plot_single_planet_multilayer(mlist, rlist, plist, rholist, 
                                          plot_prem=plot_prem,
                                          save_figures=save_figures,
                                          pname=pname)
        return mass, m0, r0, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary
    
    # calculate the outer layers
    for i in range(1, len(layers)):
        fractions_sum = sum(layer_mass_fractions[:i+1])  # total fraction from 0th to ith layer
        mi, ri, m_profilei, r_profilei, p_profilei, rho_profilei = \
            run_solver_downstep(mc=mlist[-1], pc=plist[-1], rhoc_guess=rholist[-1], 
                                rhoc=None, rc=rlist[-1], delr=delr,
                                eos_mode=layers[i], eos_args=[],
                                max_solver_steps=10 ** 6, max_decrease_steps=5, 
                                decrease_factor=1.0, ps=ps,
                                max_mass=mass*fractions_sum/1000000.0)
        mlist += m_profilei
        rlist += r_profilei
        plist += p_profilei
        rholist += rho_profilei
        mboundary.append(mi)
        rboundary.append(ri)
        pboundary.append(p_profilei[-1])
    
    if plot_profiles:
            pltmr.plot_single_planet_multilayer(mlist, rlist, plist, rholist, 
                                          plot_prem=plot_prem,
                                          save_figures=save_figures,
                                          pname=pname)
            
    return mass, mi, ri, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary
    
        
def run_single_planet_4layers(pmass, layer_mass_fractions, 
                              pc=330.0e9, ps=1.0e5, tc=5500.0,
                              rc=10.0, delr=100.0,
                              bc_ignore='',
                              pradius=None, mmw=3.6087305699338575e-27, 
                              teff=50.0, teq=300.0,
                              guillot_f=0.5, guillot_gamma=1.0,
                              tmode_core='isothermal', # naming a bit misleading because the mantle layer also uses tmode_core
                              tmode_water ='adiabatic_gradAd',
                              maxsteps=1000000,
                              eos_core="tab_Zeng21 Zeng2021_core",
                              eos_mantle="tab_Zeng21 Zeng2021_mantle",
                              eos_water="tab_AQUA", # it would be better to replace this with eos_upperMantle, because for e.g., carbon planet, this is a layer of C instead of H2O
                              eos_hhe="tab_CD21_HHe",
                              fully_adiabatic_hhe=False, mc=None, rhoc=None):
    """
    Run a single planet with 4 layers, iron, mantle, water, and H/He atmosphere,
    assuming we know its mass.
    
    Parameters
    ----------
    pmass : float
        Total planetary mass in kg.
    layer_mass_fractions : list of int
        A list of 4 integers that sums to 1,000,000 that represent the mass
        fraction of each layer. Using int instead of float here to avoid
        float issues (such as not summing exactly to 1). Parts per million
        precision should be more than enough and can be increased if necessary.
        Note that len(layer_mass_fractions) must equal len(layers).
    pc : float, optional
        Central pressure in Pa. The default is 330.0e9 Pa (estimated Earth central pressure).
    ps : float, optional
        Surface pressure outer boundary condition in Pa. The default is 1.0e5 Pa (1 bar).
    tc : float, optional
        Central temperature. The default is 5500 K (estimated Earth central temperature).
    rc : float, optional
        Central radius. The default is 10.0 m.
    delr : float, optional
        Starting step size of radius. The default is 100.0 m.
    pname : string, optional
        Name of the planet.
    bc_ignore : string, optional
        If empty '', all provided boundary conditions must be met for the code
        to stop (either surface pressure, total mass, or max steps); 
        if bc_ignore == 'mass', the total mass boundary condition will be 
        ignored for the most outer (H-He) layer;
        if bc_ignore == 'mass_mgsio3', the total mass BC will be ignored for
        the MgSiO3 layer (useful when generating M-R curves for iron+silicate
        2-layer planets)
        if bc_ignore == 'mass_h2o', the total mass BC will be ignored for
        the H2O layer (useful when generating M-R curves for 3-layer planets without
        H/He atmosphere)
    
    Several input parameters were added for H/He atmosphere layer:
    pradius : float, optional (will give error if not supplied when including H/He layer)
        Total radius of the planet. Used to calculate outer pressure boundary condition.
    mmw : float, optional
        Mean molecular weight of the atmosphere. By default is 3.6087305699338575e-27 kg
        (H/He mixture with Sun-like ratios). 
    teff, teq : float, optional
        Effective (interal) and equilibrium (due to irradiation from star) temperatures
        of the planet. By default are 50 K and 300 K.
    guillot_f, guillot_gamma : float, optional
        Coefficients required to calculate the Guillot (2010) analytical temperature profile.
        See the paper for details.
        
    tmode_core : string, optional
        Either 'isothermal' or 'adiabatic_alpha_cp' (note that Fe and MgSiO3 EOS has no adiabatic gradient),
        the temperature mode of the core (including both iron and silicate in this case).
        'adiabatic_Boujibar_2020' should also work.

    eos_core and eos_mantle offer choice of iron and silicate EOS. The default EOSs come from Zeng et al.
    2021 (Holzapfel polynomial EOS).

    Returns
    -------
    - total mass of the planet (float, same as input mass)
    - calculated total mass of the planet (float, to see if there's deviation)
    - total radius of the planet (float)
    - pressure, temperature, density at surface
    - list of M(r)
    - list of radius at each step
    - list of P(r)
    - list of T(r)
    - list of rho(r)
    - radius at the boundaries
    - masses at the boundaries
    - pressures at the boundaries
    - temperatures at the boundaries
    
    density at the boundary are not returned due to discontinuity

    """
    assert sum(layer_mass_fractions) == 1000000
    assert len(layer_mass_fractions) == 4
    
    mboundary = []
    rboundary = []
    pboundary = []
    tboundary = []
    
    # calculate the iron core, skip if mass fraction is 0 (no core)
    if layer_mass_fractions[0] != 0:
        m_profile0, r_profile0, p_profile0, t_profile0, rho_profile0 = \
            solve_mprhot_iter(pc, tc, eos_core, tmode_core,
                              rc=rc, delr=delr, ps=ps, mc=mc, rhoc=rhoc,
                              max_mass=pmass*layer_mass_fractions[0]/1000000.0,
                              maxsteps=maxsteps)
        # update inner boundary conditions for the next layer
        mc = m_profile0[-1]
        rc = r_profile0[-1]
        pc = p_profile0[-1]
        tc = t_profile0[-1]
        # add values at layer boundary to the corresponding lists
        mboundary.append(mc)
        rboundary.append(rc)
        pboundary.append(pc)
        tboundary.append(tc)
    else:
        # mc = None # avoid problem when mass fraction of iron core is 0, commented out because mc is None by default
        # rc = 10.0 # commented out because rc is 10.0 by default
        m_profile0 = []
        r_profile0 = []
        p_profile0 = []
        t_profile0 = [] 
        rho_profile0 = []
    
    mlist = m_profile0
    rlist = r_profile0
    plist = p_profile0
    tlist = t_profile0
    rholist = rho_profile0
    taulist = [None] * len(m_profile0) # initialize profile for optical depth, which won't be used until calculating H/He atmosphere layer
                                       # Nones keep the taulist aligned with other lists
                                       # Nones also essentially means infinite for the solid layers
    layer_name_list = ['core'] * len(m_profile0)
    
    # calculate the mantle, skip if mass fraction is 0 (no mantle)
    if layer_mass_fractions[1] != 0:
        if bc_ignore == 'mass_mgsio3':
            m_profile1, r_profile1, p_profile1, t_profile1, rho_profile1 = \
            solve_mprhot_iter(pc, tc, eos_mantle, tmode_core,
                              rc=rc, delr=delr, ps=ps, mc=mc, 
                              max_mass=None,
                              maxsteps=maxsteps)
        else:
            m_profile1, r_profile1, p_profile1, t_profile1, rho_profile1 = \
                solve_mprhot_iter(pc, tc, eos_mantle, tmode_core,
                                  rc=rc, delr=delr, ps=ps, mc=mc, 
                                  max_mass=pmass*sum(layer_mass_fractions[:2])/1000000.0,
                                  maxsteps=maxsteps)
        # update inner boundary conditions for the next layer
        mc = m_profile1[-1]
        rc = r_profile1[-1]
        pc = p_profile1[-1]
        tc = t_profile1[-1]
        # add values at layer boundary to the corresponding lists
        mboundary.append(mc)
        rboundary.append(rc)
        pboundary.append(pc)
        tboundary.append(tc)
    else:
        m_profile1 = []
        r_profile1 = []
        p_profile1 = []
        t_profile1 = [] 
        rho_profile1 = []
    
    mlist = mlist + m_profile1
    rlist = rlist + r_profile1
    plist = plist + p_profile1
    tlist = tlist + t_profile1
    rholist = rholist + rho_profile1
    taulist = taulist + [None] * len(m_profile1)
    layer_name_list = layer_name_list + ['mantle'] * len(m_profile1)
    
    # print(type(rc), type(pc))
    # print('Water layer')
    # print(pc, tc, rc, delr, mc, pmass*sum(layer_mass_fractions[:3])/1000000.0)
    # repeat for water layer
    if layer_mass_fractions[2] != 0:
        if bc_ignore == 'mass_h2o':
            m_profile2, r_profile2, p_profile2, t_profile2, rho_profile2 = \
                solve_mprhot_iter(pc, tc, eos_water, tmode_water,
                                rc=rc, delr=delr, ps=ps, mc=mc, 
                                max_mass=None,
                                maxsteps=maxsteps)
        else:
            m_profile2, r_profile2, p_profile2, t_profile2, rho_profile2 = \
                solve_mprhot_iter(pc, tc, eos_water, tmode_water,
                                rc=rc, delr=delr, ps=ps, mc=mc, 
                                max_mass=pmass*sum(layer_mass_fractions[:3])/1000000.0,
                                maxsteps=maxsteps)
        # update inner boundary conditions for the next layer
        mc = m_profile2[-1]
        rc = r_profile2[-1]
        pc = p_profile2[-1]
        tc = t_profile2[-1]
        # print(r_profile2[-10:], p_profile2[-10:])
        # print(type(rc), type(pc))
        # print(rc, pc)
        # add values at layer boundary to the corresponding lists
        mboundary.append(mc)
        rboundary.append(rc)
        pboundary.append(pc)
        tboundary.append(tc)
    else:
        m_profile2 = []
        r_profile2 = []
        p_profile2 = []
        t_profile2 = [] 
        rho_profile2 = []
    # print(m_profile2[:10], r_profile2[:10], p_profile2[:10], t_profile2[:10],
    #       rho_profile2[:10])
    # water_outputs = np.array([m_profile2, r_profile2, p_profile2,
    #                           t_profile2, rho_profile2])
    # water_outputs = np.transpose(water_outputs)
    # pd.DataFrame(water_outputs).to_csv("K2-18b_test_water_layer.csv",
    #                                    sep='\t')
    
    mlist = mlist + m_profile2
    rlist = rlist + r_profile2
    plist = plist + p_profile2
    tlist = tlist + t_profile2
    rholist = rholist + rho_profile2
    taulist = taulist + [None] * len(rho_profile2) # for steam atmosphere we may encounter physical tau value, space for improvement
    layer_name_list = layer_name_list + ['water'] * len(m_profile2)
    
    # repeat for H/He atmosphere layer
    # several new variables are introduced by H/He atmosphere layer, including:
    # pradius, mmw, teff, teq, guillot_f, guillot_gamma
    if layer_mass_fractions[3] != 0 and not fully_adiabatic_hhe:
        if bc_ignore == 'mass':
            m_profile3, r_profile3, p_profile3, t_profile3, rho_profile3, tau_profile3 = \
                hhe_gas_find_rcb_solve(pc=pc, tc=tc, eos_mode=eos_hhe,
                                       Mp=None, # ignore total mass outer BC
                                       Rp=pradius, rc=rc, delr=delr, ps=ps, mc=mc, #rhoc=rholist[-1], # default rhoc=None, will be calculated from EOS
                                       mmw=mmw, maxsteps=maxsteps, teff=teff, teq=teq, 
                                       f=guillot_f, gamma=guillot_gamma,
                                       hhe_bc_mode='opacity') # parameter defining the BC mode of H/He layer, 'opacity' will ignore total mass and calculate until tau=1
        else:
            m_profile3, r_profile3, p_profile3, t_profile3, rho_profile3, tau_profile3 = \
                hhe_gas_find_rcb_solve(pc=pc, tc=tc, eos_mode=eos_hhe,
                                       Mp=pmass, # note that hhe_gas_find_rcb_solve needs total Mp while other layers need layer mass only
                                       Rp=pradius, rc=rc, delr=delr, ps=ps, mc=mc, #rhoc=rholist[-1], # default rhoc=None, will be calculated from EOS
                                       mmw=mmw, maxsteps=maxsteps, teff=teff, teq=teq, 
                                       f=guillot_f, gamma=guillot_gamma,
                                       hhe_bc_mode='mass')
        # update inner boundary conditions for the next layer
        mc = m_profile3[-1]
        rc = r_profile3[-1]
        pc = p_profile3[-1]
        tc = t_profile3[-1]
        # add values at layer boundary to the corresponding lists
        mboundary.append(mc)
        rboundary.append(rc)
        pboundary.append(pc)
        tboundary.append(tc)
    elif layer_mass_fractions[3] != 0 and fully_adiabatic_hhe:
        # fully adiabatic H/He envelope, call solve_mprhot_iter() instead of hhe_gas_find_rcb_solve()
        m_profile3, r_profile3, p_profile3, t_profile3, rho_profile3 =\
            solve_mprhot_iter(pc=pc, tc=tc, eos_mode=eos_hhe, t_mode="adiabatic_gradAd",
                              rc=rc, delr=delr, ps=ps, # 10 bar surface pressure
                              mc=mc, maxsteps=maxsteps)
        tau_profile3 = [] # no opacity calculated for fully adiabatic H/He envelope
        # update inner boundary conditions for the next layer
        mc = m_profile3[-1]
        rc = r_profile3[-1]
        pc = p_profile3[-1]
        tc = t_profile3[-1]
        # add values at layer boundary to the corresponding lists
        mboundary.append(mc)
        rboundary.append(rc)
        pboundary.append(pc)
        tboundary.append(tc)
    else:
        m_profile3 = []
        r_profile3 = []
        p_profile3 = []
        t_profile3 = [] 
        rho_profile3 = []
        tau_profile3 = []
    
    mlist = mlist + m_profile3
    rlist = rlist + r_profile3
    plist = plist + p_profile3
    tlist = tlist + t_profile3
    rholist = rholist + rho_profile3
    taulist = taulist + tau_profile3
    layer_name_list = layer_name_list + ['H/He'] * len(m_profile3)
    
    pmass_calculated = mlist[-1]
    rs = rlist[-1]
    ps = plist[-1]
    ts = tlist[-1]
    rhos = rholist[-1]
    
    # Return 17 parameters in total. Note that this number has changed before and is a potential source of error,
    # be careful when calling run_single_planet_4layers()
    return pmass, pmass_calculated, rs, ps, ts, rhos, mlist, rlist, plist, tlist, rholist, \
        taulist, mboundary, rboundary, pboundary, tboundary, layer_name_list


def run_single_planet_AVL(pmass, mass_fraction_functions, pc=330.0e9, ps=1.0e5, tc=5500.0,
                          rc=10.0, delr=100.0, bc_ignore='', pradius=None, mmw=3.6087305699338575e-27, 
                          teff=50.0, teq=300.0, guillot_f=0.5, guillot_gamma=1.0,
                          maxsteps=1000000):
    """
    Given inner boundary conditions (pc, tc), outer boundary conditions (Mp, Rp, teq, ps), and three mass
    fraction profiles X_rock, X_water, and X_hhe, generate the internal structure and adiabatic temperature 
    profiles for a mixture planet.
    
    The code uses a shooting method to find pc and tc that produces the desired Mp and Rp. Mp and ps decide
    termination, so it is possible that r > Rp. For r > Rp, assume a purely H/He atmosphere and use 
    hhe_gas_find_rcb_solve() to solve for both the radiative and convective layers, and generate either adiabatic
    or Guillot+2010 analytical temperature profile.
    """
    mboundary = []
    rboundary = []
    pboundary = []
    tboundary = []
    
    mlist, rlist, plist, tlist, rholist = \
        solve_mprhot_iter(pc, tc, "AVL", "AVL", rc=rc, delr=delr, ps=ps, max_mass=pmass,
                          maxsteps=maxsteps, mass_fraction_functions=mass_fraction_functions,
                          pradius=pradius)
        
    taulist = [None] * len(mlist) # initialize profile for optical depth, which won't be used until calculating 
                                  # H/He atmosphere layer; Nones keep the taulist aligned with other lists;
                                  # Nones also essentially means infinite for the solid layers
    layer_name_list = ['mixed'] * len(mlist) # generate this list to make the outputs of this function match run_single_planet_4layers()
    
    # update inner boundary conditions for H/He layer
    mc = mlist[-1]
    rc = rlist[-1]
    pc = plist[-1]
    tc = tlist[-1]
    rhoc = rholist[-1]
    # this "boundary" is either the surface of the planet (if r < Rp) or the bottom of the pure H/He atmosphere
    # (if r > Rp)
    mboundary.append(mc)
    rboundary.append(rc)
    pboundary.append(pc)
    tboundary.append(tc)
    
    # At this stage, we have a planet ran up to Rp; mass and surface pressure outer boundary conditions may not be
    # met yet, so run a pure H/He atmosphere for r > Rp
    if bc_ignore == 'mass':
        mlist_hhe, rlist_hhe, plist_hhe, tlist_hhe, rholist_hhe, taulist_hhe = \
            hhe_gas_find_rcb_solve(pc=pc, tc=tc, eos_mode="tab_CD21_HHe",
                                   Mp=None, # ignore total mass outer BC
                                   Rp=pradius, rc=rc, delr=delr, ps=ps, mc=mc, rhoc=rhoc, 
                                   mmw=mmw, maxsteps=maxsteps, teff=teff, teq=teq, 
                                   f=guillot_f, gamma=guillot_gamma,
                                   hhe_bc_mode='opacity') # parameter defining the BC mode of H/He layer, 'opacity' will ignore total mass and calculate until tau=1
    else:
        mlist_hhe, rlist_hhe, plist_hhe, tlist_hhe, rholist_hhe, taulist_hhe = \
            hhe_gas_find_rcb_solve(pc=pc, tc=tc, eos_mode="tab_CD21_HHe",
                                   Mp=pmass, # note that hhe_gas_find_rcb_solve needs total Mp while other layers need layer mass only
                                   Rp=pradius, rc=rc, delr=delr, ps=ps, mc=mc, rhoc=rhoc,
                                   mmw=mmw, maxsteps=maxsteps, teff=teff, teq=teq, 
                                   f=guillot_f, gamma=guillot_gamma,
                                   hhe_bc_mode='mass')
    
    mlist = mlist + mlist_hhe
    rlist = rlist + rlist_hhe
    plist = plist + plist_hhe
    tlist = tlist + tlist_hhe
    rholist = rholist + rholist_hhe
    taulist = taulist + taulist_hhe
    layer_name_list = layer_name_list + ['H/He'] * len(mlist_hhe)
    
    pmass_calculated = mlist[-1]
    rs = rlist[-1]
    ps = plist[-1]
    ts = tlist[-1]
    rhos = rholist[-1]
    
    return pmass, pmass_calculated, rs, ps, ts, rhos, \
        mlist, rlist, plist, tlist, rholist, \
        taulist, \
        mboundary, rboundary, pboundary, tboundary, layer_name_list
    

def test_single_planet_4layers():
    """ Compare run_single_planet_multilayer() with run_single_planet_4layers() 
        to see if the temperature dependent code is working. 
    """
    # # first run, Earth-like 32% iron core + 68% silicate mantle
    # begin_time = datetime.datetime.now()
    # _, m_earth, r_earth, mlist_earth, rlist_earth, plist_earth, rholist_earth, _, _, _ = \
    #     run_single_planet_multilayer(1.0*Me, ['tabulated fe', 'tabulated mgsio3'], 
    #                                  [320000, 680000], pc=380.0e9, plot_profiles=False)
    # tlist_earth = [300.0] * len(rlist_earth)  # isothermal model assuming 300 K
    # print("Isothermal model for Earth done.")
    # print("M, R calculated are: %f M_Earth, %f R_Earth" % (m_earth/Me, r_earth/Re))
    # print("Execution time: ", datetime.datetime.now() - begin_time)
    # print()
    
    # begin_time = datetime.datetime.now()
    # _, m_earth4l, r_earth4l, _, _, _, mlist_earth4l, rlist_earth4l, plist_earth4l, \
    #     tlist_earth4l, rholist_earth4l, _, _, _, _ = \
    #         run_single_planet_4layers(1.0*Me, [320000, 680000, 0, 0], pc=380.0e9,
    #                                   tc=5500.0)
    # pltmr.overplot_multiple_planets([mlist_earth, mlist_earth4l],
    #                                 [rlist_earth, rlist_earth4l],
    #                                 [plist_earth, plist_earth4l],
    #                                 [tlist_earth, tlist_earth4l],
    #                                 [rholist_earth, rholist_earth4l],
    #                                 ['Isothermal', 'Adiabatic'])
    # print("Adiabatic model for Earth done.")
    # print("M, R calculated are: %f M_Earth, %f R_Earth" % (m_earth4l/Me, r_earth4l/Re))
    # print("Execution time: ", datetime.datetime.now() - begin_time)
    # print()
    
    # second run, K2-18 b model (3), 10% Earth-like core (3% Fe, 7% silicates),
    # 89.994% H2O, 0.006% H/He (ignore H/He for now and add mass to the water
    # layer).
    begin_time = datetime.datetime.now()
    _, m_k218b, r_k218b, mlist_k218b, rlist_k218b, plist_k218b, rholist_k218b, _, _, _ = \
        run_single_planet_multilayer(8.63*Me, ['tabulated fe', 'tabulated mgsio3', 'tabulated h2o'], 
                                      [30000, 70000, 900000], pc=1250.0e9, plot_profiles=False)
    tlist_k218b = [300.0] * len(rlist_k218b)  # isothermal model assuming 300 K
    print("Isothermal model for K2-18 b done.")
    print("M, R calculated are: %f M_Earth, %f R_Earth" % (m_k218b/Me, r_k218b/Re))
    print("Execution time: ", datetime.datetime.now() - begin_time)
    print()
    
    begin_time = datetime.datetime.now()
    _, m_k218b4l, r_k218b4l, _, _, _, mlist_k218b4l, rlist_k218b4l, plist_k218b4l, \
        tlist_k218b4l, rholist_k218b4l, _, _, _, _ = \
            run_single_planet_4layers(8.63*Me, [30000, 70000, 900000, 0], pc=1250.0e9,
                                      tc=7000.0)
    print("Adiabatic model for K2-18 b done.")
    print("M, R calculated are: %f M_Earth, %f R_Earth" % (m_k218b4l/Me, r_k218b4l/Re))
    print("Execution time: ", datetime.datetime.now() - begin_time)
    print()
    pltmr.overplot_multiple_planets([mlist_k218b, mlist_k218b4l],
                                    [rlist_k218b, rlist_k218b4l],
                                    [plist_k218b, plist_k218b4l],
                                    [tlist_k218b, tlist_k218b4l],
                                    [rholist_k218b, rholist_k218b4l],
                                    ['Isothermal', 'Adiabatic'],
                                    save_figures=True,
                                    savename='K2-18b_model2')
    
    # third run, K2-18 b model (2), 45% Earth-like core (30% Fe, 70% silicates),
    # 54.97% H2O, 0.03% H/He; include the H/He layer
    # Radius of K2-18 b is 2.61+-0.087 Re (Benneke et al. 2019)
    # begin_time = datetime.datetime.now()
    # _, m_k218b, r_k218b, mlist_k218b, rlist_k218b, plist_k218b, rholist_k218b, _, _, _ = \
    #     run_single_planet_multilayer(8.63*Me, ['tabulated fe', 'tabulated mgsio3', 'tabulated h2o'], 
    #                                   [135000, 315000, 550000], pc=2170.0e9, plot_profiles=False)
    # tlist_k218b = [300.0] * len(rlist_k218b)  # isothermal model assuming 300 K
    # print("Isothermal model for K2-18 b done.")
    # print("M, R calculated are: %f M_Earth, %f R_Earth" % (m_k218b/Me, r_k218b/Re))
    # print("Surface pressure is: %f bar" % (plist_k218b[-1] / 1.0e5))
    # print("Execution time: ", datetime.datetime.now() - begin_time)
    # print()
    
    # # input conditions
    # mt = 9.95 * Me
    # # mass_fractions = [382748, 617252, 0, 0]
    # mass_fractions = [274471, 557261, 168268, 0]
    # pc = 4.968750e+12
    # tc = 300.0
    # bc_ignore='mass'
    # ps = 1.0e4
    # print('Input conditions are:')
    # print('Total mass = %f M_Earth' % (mt / Me))
    # print('Mass fractions:', mass_fractions)
    # print('P(0) and T(0): %f GPa, %f K' % (pc/1.0e9, tc))
    # print('Outer boundary conditions: P_surf = %f bar, ignore %s' % (ps/1.0e5, bc_ignore))
    # print()
    # begin_time = datetime.datetime.now()
    # _, m_toi1075b, r_toi1075b, _, _, _, mlist_toi1075b, rlist_toi1075b, plist_toi1075b, \
    #     tlist_toi1075b, rholist_toi1075b, mb, rb, pb, tb = \
    #         run_single_planet_4layers(mt, mass_fractions, pc=pc, tc=tc, 
    #                                   bc_ignore=bc_ignore, ps=ps)
    # print("Adiabatic model for TOI-1075 b done.")
    # print("M, R calculated are: %f M_Earth, %f R_Earth" % (m_toi1075b/Me, r_toi1075b/Re))
    # # print(rb, pb, tb)
    # print("M, R, P, T at water-H/He boundaries are: %f M_Earth, %f R_Earth, %f GPa, %f K" \
    #       % (mb[-2]/Me, rb[-2]/Re, pb[-2]/1.0e9, tb[-2]))
    # print("Surface pressure is: %f bar" % (plist_toi1075b[-1] / 1.0e5))
    # print("Execution time: ", datetime.datetime.now() - begin_time)
    # print()
    # pltmr.overplot_multiple_planets([mlist_toi1075b],
    #                                 [rlist_toi1075b],
    #                                 [plist_toi1075b],
    #                                 [tlist_toi1075b],
    #                                 [rholist_toi1075b],
    #                                 ['Adiabatic'],
    #                                 save_figures=False,
    #                                 savename='TOI1075b_mean_core_radius_2layer')


def main():    
    test_single_planet_4layers()
    
    # # # generate a M vs R plot for three homogeneous planets made of pure
    # # # Fe, MgSiO3, and H2O
    # # run_plot_homogeneous()

    # # # plot the EOS (P vs rho) for Fe, MgSiO3, and H2O
    # # plot_EOS()
    
    # # block for running run_single_planet_multilayer()
    # begin_time = datetime.datetime.now()
    # # # Earth-like 32% iron core + 68% silicate mantle
    # # mass, m, r, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary =\
    # #     run_single_planet_multilayer(1.0*Me, ['tabulated fe', 'tabulated mgsio3'], 
    # #                                   [320000, 680000], pc=380.0e9, plot_profiles=True,
    # #                                   plot_prem=True)
    # # CoRoT-7b, 4.8 Me, 1.68 Re, assuming 30% core mass fraction
    # # mass, m, r, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary =\
    # #     run_single_planet_multilayer(4.8*Me, ['tabulated fe', 'tabulated mgsio3'], 
    # #                                   [300000, 700000], pc=1747.65625e9, plot_profiles=True)
        
    # # K2-18 b, 8.63 Me, 2.61 Re, assuming three end members from Madhusudhan et al. 2020
    # # (1) 94.7% Fe, 0.3% H2O, 5% H/He
    # # (2) 45% Earth-like core (13.5% Fe, 31.5% silicates), 54.97% H2O, 0.03% H/He
    # # (3) 10% Earth-like core (3% Fe, 7% silicates), 89.994% H2O, 0.006% H/He
    # mass, m, r, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary =\
    #     run_single_planet_multilayer(8.63*Me*0.95, ['tabulated fe', 'tabulated h2o'], 
    #                                   [996842, 3158], pc=5500.0e9, 
    #                                   plot_profiles=True,
    #                                   save_figures=True,
    #                                   pname='K2-18b_Madhu20-1')
        
    # # mass, m, r, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary =\
    #     # run_single_planet_multilayer(1.0*Me, ['tabulated fe'], [1000000])
    # # mass, m, r, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary =\
    # #     run_single_planet_multilayer(1.0*Me, 
    # #                                   ['tabulated fe', 'tabulated mgsio3', 'tabulated h2o'], 
    # #                                   [300000, 400000, 300000], pc=390.0e9)
    # print("Input mass: ",mass)
    # print("Calculated mass: ", m)
    # print("Calculated radius: ", r)
    # # print(mlist[-10:])
    # # print(rlist[-10:])
    # # print(plist[-10:])
    # print("M at layer boundaries: ", mboundary)
    # print("R at layer boundaries: ", rboundary)
    # print("P at layer boundaries: ", pboundary)
    # print("Execution time: ", datetime.datetime.now() - begin_time)

    
if __name__ == '__main__':
    main()
    

