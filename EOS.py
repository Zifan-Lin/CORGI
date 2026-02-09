#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:41:09 2020

@author: linzifan

Equation of state (EOS) equations for M-R relationship calculation.
"""

#=========================================================================
#====== import necessary packages
#=========================================================================

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator

#=========================================================================
#====== useful constants, in SI
#=========================================================================

G = 6.67408e-11  # gravitational constant
Re = 6371000.0  # Earth radius
Me = 5.97e24  # Earth mass

me = 9.1093837015e-31  # electron mass
qe = 1.60217662e-19  # electron charge
hbar = 6.62607004e-34 / (2.0 * np.pi)  # Planck's constant / 2pi
a0 = 5.29177210903e-11  # Bohr radius
amu = 1.66053906660e-27  # atmoic mass unit
c = 2.99792458e8  # speed of light

#=========================================================================
#====== load tabulated EOS
#=========================================================================

# water
print('Loading Seager et al. 2007 H2O EOS ...')
tabulated_eos_h2o = np.genfromtxt('Data/EOS/water.txt')
tabulated_eos_h2o = np.transpose(tabulated_eos_h2o)
tabulated_p_h2o = tabulated_eos_h2o[0]
tabulated_rho_h2o = tabulated_eos_h2o[1]
tabulated_p_to_rho_h2o = interp1d(tabulated_p_h2o, tabulated_rho_h2o, kind='linear',
                                  fill_value="extrapolate")
print('Completed')
print()

print('Loading Zeng et al. 2021 H2O EOS ...')
Zeng2021_eos_water = np.genfromtxt('Data/EOS/Zeng2021_Holzapfel_water.txt')
Zeng2021_eos_water = np.transpose(Zeng2021_eos_water)
Zeng2021_eos_water_p = Zeng2021_eos_water[0]
Zeng2021_eos_water_rho = Zeng2021_eos_water[1]
Zeng2021_eos_water_p_to_rho = interp1d(Zeng2021_eos_water_p, Zeng2021_eos_water_rho, kind='linear',
                                       fill_value="extrapolate")
print('Completed')
print()

# iron
print('Loading Seager et al. 2007 Fe EOS ...')
tabulated_eos_fe = np.genfromtxt('Data/EOS/iron.txt')
tabulated_eos_fe = np.transpose(tabulated_eos_fe)
tabulated_p_fe = tabulated_eos_fe[0]
tabulated_rho_fe = tabulated_eos_fe[1]
tabulated_p_to_rho_fe = interp1d(tabulated_p_fe, tabulated_rho_fe, kind='linear', fill_value="extrapolate")
print('Completed')
print()

print('Loading Zeng et al. 2016 core EOS ...')
Zeng2016_eos_core = np.genfromtxt('Data/EOS/Zeng2016_core.txt')
Zeng2016_eos_core = np.transpose(Zeng2016_eos_core)
Zeng2016_eos_core_p = Zeng2016_eos_core[0]
Zeng2016_eos_core_rho = Zeng2016_eos_core[1]
Zeng2016_eos_core_p_to_rho = interp1d(Zeng2016_eos_core_p, Zeng2016_eos_core_rho, kind='linear',
                                      fill_value="extrapolate")
print('Completed')
print()

print('Loading Zeng et al. 2021 core EOS ...')
Zeng2021_eos_core = np.genfromtxt('Data/EOS/Zeng2021_Holzapfel_core.txt')
Zeng2021_eos_core = np.transpose(Zeng2021_eos_core)
Zeng2021_eos_core_p = Zeng2021_eos_core[0]
Zeng2021_eos_core_rho = Zeng2021_eos_core[1]
Zeng2021_eos_core_p_to_rho = interp1d(Zeng2021_eos_core_p, Zeng2021_eos_core_rho, kind='linear',
                                      fill_value="extrapolate")
print('Completed')
print()

print('Loading Smith et al. 2018 solid iron EOS ...')
Smith2018_eos_core = np.genfromtxt('Data/EOS/Smith2018_solid_iron.txt')
Smith2018_eos_core = np.transpose(Smith2018_eos_core)
Smith2018_eos_core_p = Smith2018_eos_core[0]
Smith2018_eos_core_rho = Smith2018_eos_core[1]
Smith2018_eos_core_p_to_rho = interp1d(Smith2018_eos_core_p, Smith2018_eos_core_rho, kind='linear',
                                      fill_value="extrapolate")
print('Completed')
print()

print('Loading Grant et al. 2021 liquid iron EOS ...')
Grant2021_eos_core = np.genfromtxt('Data/EOS/Grant2021_liquid_iron.txt')
Grant2021_eos_core = np.transpose(Grant2021_eos_core)
Grant2021_eos_core_p = Grant2021_eos_core[0]
Grant2021_eos_core_rho = Grant2021_eos_core[1]
Grant2021_eos_core_p_to_rho = interp1d(Grant2021_eos_core_p, Grant2021_eos_core_rho, kind='linear',
                                      fill_value="extrapolate")
print('Completed')
print()

# silicate
print('Loading Seager et al. 2007 MgSiO3 (en) EOS ...')
tabulated_eos_silicate = np.genfromtxt('Data/EOS/silicate.txt')
tabulated_eos_silicate = np.transpose(tabulated_eos_silicate)
tabulated_p_silicate = tabulated_eos_silicate[0]
tabulated_rho_silicate = tabulated_eos_silicate[1]
tabulated_p_to_rho_silicate = interp1d(tabulated_p_silicate, 
                                       tabulated_rho_silicate, kind='linear', fill_value="extrapolate")
print('Completed')
print()

print('Loading Seager et al. 2007 MgSiO3 (pv) EOS ...')
tabulated_eos_silicate_pv = np.genfromtxt('Data/EOS/Seager2007_silicates_pv.txt')
tabulated_eos_silicate_pv = np.transpose(tabulated_eos_silicate_pv)
tabulated_eos_silicate_pv_p = tabulated_eos_silicate_pv[0]
tabulated_eos_silicate_pv_rho = tabulated_eos_silicate_pv[1]
tabulated_silicate_pv_p_to_rho = interp1d(tabulated_eos_silicate_pv_p, 
                                          tabulated_eos_silicate_pv_rho, 
                                          kind='linear', fill_value="extrapolate")
print('Completed')
print()

print('Loading Hubbard & Marley 1989 analytical rock EOS ...')
tabulated_eos_silicate_pv = np.genfromtxt('Data/EOS/HubbardMarley1989_rock_EOS.txt')
tabulated_eos_silicate_pv = np.transpose(tabulated_eos_silicate_pv)
tabulated_eos_silicate_pv_p = tabulated_eos_silicate_pv[0]
tabulated_eos_silicate_pv_rho = tabulated_eos_silicate_pv[1]
tabulated_hm89_rock = interp1d(tabulated_eos_silicate_pv_p, 
                               tabulated_eos_silicate_pv_rho, 
                               kind='linear', fill_value="extrapolate")
print('Completed')
print()

print('Loading Seager et al. 2007 MgSiO3 (iron-rich pv) EOS ...')
tabulated_eos_silicate_pvi = np.genfromtxt('Data/EOS/Seager2007_iron_rich_silicates_pv.txt')
tabulated_eos_silicate_pvi = np.transpose(tabulated_eos_silicate_pvi)
tabulated_eos_silicate_pvi_p = tabulated_eos_silicate_pvi[0]
tabulated_eos_silicate_pvi_rho = tabulated_eos_silicate_pvi[1]
tabulated_silicate_pvi_p_to_rho = interp1d(tabulated_eos_silicate_pvi_p, 
                                           tabulated_eos_silicate_pvi_rho, 
                                           kind='linear', fill_value="extrapolate")
print('Completed')
print()

print('Loading Zeng et al. 2016 mantle EOS ...')
Zeng2016_eos_mantle = np.genfromtxt('Data/EOS/Zeng2016_mantle.txt')
Zeng2016_eos_mantle = np.transpose(Zeng2016_eos_mantle)
Zeng2016_eos_mantle_p = Zeng2016_eos_mantle[0]
Zeng2016_eos_mantle_rho = Zeng2016_eos_mantle[1]
Zeng2016_eos_mantle_p_to_rho = interp1d(Zeng2016_eos_mantle_p, Zeng2016_eos_mantle_rho, kind='linear',
                                        fill_value="extrapolate")
print('Completed')
print()

print('Loading Zeng et al. 2021 mantle EOS ...')
Zeng2021_eos_mantle = np.genfromtxt('Data/EOS/Zeng2021_Holzapfel_mantle.txt')
Zeng2021_eos_mantle = np.transpose(Zeng2021_eos_mantle)
Zeng2021_eos_mantle_p = Zeng2021_eos_mantle[0]
Zeng2021_eos_mantle_rho = Zeng2021_eos_mantle[1]
Zeng2021_eos_mantle_p_to_rho = interp1d(Zeng2021_eos_mantle_p, Zeng2021_eos_mantle_rho, kind='linear',
                                        fill_value="extrapolate")
print('Completed')
print()

print('Loading Oganov & Ono 2004 MgSiO3 (ppv) EOS ...')
OgaOno2004_eos_mantle = np.genfromtxt('Data/EOS/OganovOno_2004_MgSiO3_ppv.txt')
OgaOno2004_eos_mantle = np.transpose(OgaOno2004_eos_mantle)
OgaOno2004_eos_mantle_p = OgaOno2004_eos_mantle[0]
OgaOno2004_eos_mantle_rho = OgaOno2004_eos_mantle[1]
OgaOno2004_eos_mantle_p_to_rho = interp1d(OgaOno2004_eos_mantle_p, OgaOno2004_eos_mantle_rho, kind='linear',
                                        fill_value="extrapolate")
print('Completed')
print()

# carbon and carbide EOSs
print('Loading Seager et al. (2007) carbon (graphite) EOS ...')
s07_eos_graphite = np.genfromtxt('Data/EOS/Seager2007_graphite_BME3.txt')
s07_eos_graphite = np.transpose(s07_eos_graphite)
s07_eos_graphite_p = s07_eos_graphite[0]
s07_eos_graphite_rho = s07_eos_graphite[1]
s07_eos_graphite_p_to_rho = interp1d(s07_eos_graphite_p, s07_eos_graphite_rho, kind='linear',
                                     fill_value="extrapolate")
print('Completed')
print()

print('Loading Swift et al. (2022) carbon (diamond) EOS ...')
sw22_eos_diamond = np.genfromtxt('Data/EOS/Swift2022_C_diamond_EOS.txt')
sw22_eos_diamond = np.transpose(sw22_eos_diamond)
sw22_eos_diamond_p = sw22_eos_diamond[0]
sw22_eos_diamond_rho = sw22_eos_diamond[1]
sw22_eos_diamond_p_to_rho = interp1d(sw22_eos_diamond_p, sw22_eos_diamond_rho, kind='linear',
                                     fill_value="extrapolate")
print('Completed')
print()

# SiC, low-pressure zinc-blende (B3) structure
print('Loading SiC zinc-blende structure EOS ...')
sic_eos_zincBlende = np.genfromtxt('Data/EOS/SiC_zincBlende_EOS.txt')
sic_eos_zincBlende = np.transpose(sic_eos_zincBlende)
sic_eos_zincBlende_p = sic_eos_zincBlende[0]
sic_eos_zincBlende_rho = sic_eos_zincBlende[1]
sic_eos_zincBlende_p_to_rho = interp1d(sic_eos_zincBlende_p, sic_eos_zincBlende_rho, kind='linear',
                                     fill_value="extrapolate")
print('Completed')
print()

# SiC, high-pressure rock-salt (B1) structure
print('Loading SiC rock-salt structure EOS ...')
sic_eos_rockSalt = np.genfromtxt('Data/EOS/SiC_rockSalt_EOS.txt')
sic_eos_rockSalt = np.transpose(sic_eos_rockSalt)
sic_eos_rockSalt_p = sic_eos_rockSalt[0]
sic_eos_rockSalt_rho = sic_eos_rockSalt[1]
sic_eos_rockSalt_p_to_rho = interp1d(sic_eos_rockSalt_p, sic_eos_rockSalt_rho, kind='linear',
                                     fill_value="extrapolate")
print('Completed')
print()

# French et al. (2009) QMD simulation EOS for H2O
print('Loading French et al. (2009) H2O EOS ...')
# read original EOS data
french09_h2o_data = np.genfromtxt('Data/French2009_H2O_EOS/French2009_H2O_EOS_extended.txt')
french09_h2o_data = np.transpose(french09_h2o_data)
french09_h2o_p = french09_h2o_data[3] # pressure in log10 Pa
french09_h2o_t = french09_h2o_data[0]
french09_h2o_rho = french09_h2o_data[1] * 1.0e3 # convert g/cm3 into kg/m3
#===== version 1 using griddata + RectBivariateSpline ==============================
# french09_h2o_pt_points = (french09_h2o_p, french09_h2o_t)
# french09_h2o_pt_points = np.transpose(french09_h2o_pt_points)
# # define new rectangular grid to map original data onto
# grid_p, grid_t = np.mgrid[min(french09_h2o_p):max(french09_h2o_p):100j, min(french09_h2o_t):max(french09_h2o_t):200j]
# # RectBivariateSpline only works for data on a rectangular grid, but the French+2009 EOS is on irregular grid, so we use griddata instead
# french09_grid_rho = griddata(french09_h2o_pt_points, french09_h2o_rho, (grid_p, grid_t), method='cubic')
# grid_p = grid_p[:,0]
# grid_t = grid_t[0]
# # now french09_grid_rho is a 2D array representing rho values on rectangular grid point (grid_p, grid_t)
# # therefore, we can call RectBivariateSpline for interpolation
# french09_interp_spline_rho = RectBivariateSpline(grid_p, grid_t, french09_grid_rho, kx=1, ky=1)
#===== version 2 using LinearNDInterpolator ========================================
# french09_interp_nd_rho = LinearNDInterpolator(list(zip(french09_h2o_p, french09_h2o_t)), french09_h2o_rho)
#===== version 3 using CloughTocher2DInterpolator ==================================
# french09_interp_ct2d_rho = CloughTocher2DInterpolator(list(zip(french09_h2o_p, french09_h2o_t)), french09_h2o_rho)
#===================================================================================
print('Completed')
print()

# Fiducial EOS interpolating rho(p) from N13 U1
n13_u1 = np.genfromtxt('Data/Nettelmann2013_table_U1.dat')
n13_u1 = np.transpose(n13_u1)
n13_u1_p = n13_u1[1] * 1.0e9 # convert GPa to Pa
n13_u1_rho = n13_u1[4] * 1.0e3 # convert g/cm3 to kg/m3
n13_u1_interp1d_rho = interp1d(n13_u1_p, n13_u1_rho, fill_value="extrapolate")

# The same for N13 U2
n13_u2 = np.genfromtxt('Data/Nettelmann2013_table_U2.dat')
n13_u2 = np.transpose(n13_u2)
n13_u2_p = n13_u2[1] * 1.0e9 # convert GPa to Pa
n13_u2_rho = n13_u2[4] * 1.0e3 # convert g/cm3 to kg/m3
n13_u2_interp1d_rho = interp1d(n13_u2_p, n13_u2_rho, fill_value="extrapolate")

# AQUA EOS for H2O
print('Loading AQUA H2O EOS ...')
aqua_fn = 'Data/AQUA_Haldemann2020/aqua_eos_pt_v1_0.dat'
aqua_data = pd.read_table(aqua_fn, skiprows=19, delim_whitespace=True, header=None)
aqua_p = aqua_data[0]
aqua_t = aqua_data[1]
aqua_rho2d = []
aqua_ad_grad2d = []
aqua_s2d = [] # specific entropy in J/(kg*K)
aqua_u2d = [] # specific internal energy in J/kg
for pressure_group, sub_df in aqua_data.groupby(aqua_p):
    aqua_rho2d.append(sub_df[2])
    aqua_ad_grad2d.append(sub_df[3])
    aqua_s2d.append(sub_df[4])
    aqua_u2d.append(sub_df[5])
aqua_rho2d = np.array(aqua_rho2d, dtype=object)
aqua_ad_grad2d = np.array(aqua_ad_grad2d, dtype=object)
aqua_s2d = np.array(aqua_s2d, dtype=object)
aqua_u2d = np.array(aqua_u2d, dtype=object)
aqua_p = np.array(aqua_p.unique())
aqua_t = np.array(aqua_t.unique())
# print(np.shape(aqua_p), np.shape(aqua_t))
# print(np.shape(aqua_rho2d))
# RegularGridInterpolator for accurate values at phase boundaries
# aqua_interp_grid_rho = RegularGridInterpolator((aqua_p, aqua_t), aqua_rho2d)
# aqua_interp_grid_ad_grad = RegularGridInterpolator((aqua_p, aqua_t), aqua_ad_grad2d)
# RectBivariateSpline for speed
# print(np.shape(aqua_p), np.shape(aqua_t), np.shape(aqua_rho2d))
aqua_interp_spline_rho = RectBivariateSpline(aqua_p, aqua_t, aqua_rho2d, kx=1, ky=1)
aqua_interp_spline_ad_grad = RectBivariateSpline(aqua_p, aqua_t, aqua_ad_grad2d, kx=1, ky=1)
aqua_interp_spline_s = RectBivariateSpline(aqua_p, aqua_t, aqua_s2d, kx=1, ky=1)
aqua_interp_spline_u = RectBivariateSpline(aqua_p, aqua_t, aqua_u2d, kx=1, ky=1)
# rho_interp = aqua_interp_spline_rho(np.array([7.71876652E+12, 1.81970086E+04]))
# rho_interp2 = aqua_interp_spline_rho(np.array([7.21187761E+05, 4.78630092E+02]))
# print(rho_interp, rho_interp2)
print('Completed')
print()

# Chabrier & Debras 2021 H-He mixture EOS
# only use a subset of the EOS. The full table goes up to 1.0e8 K and 1.0e13 GPa, too high for planets
# currently use a subset up to 1.0e4 K and 1.0e5 GPa (may change in the future)
print('Loading Chabrier & Debras 2021 H-He EOS ...')
cd21_fn = 'Data/H_He_Chabrier2019/TABLEEOS_2021_TP_Y0275_v1'
cd21_data = pd.read_table(cd21_fn, delim_whitespace=True, comment='#', header=None)
cd21_subset = cd21_data[cd21_data[0] <= 4.0] # temperataure up to 10000 K
cd21_subset = cd21_subset[cd21_subset[1] <= 5.0] # pressure up to 100,000 GPa
cd21_t = 10 ** cd21_subset[0]  # convert log10 T to T in K
cd21_p = 10 ** cd21_subset[1] * 1.0e9  # convert log P in GPa to P in Pa
cd21_rho2d = []
cd21_ad_grad2d = []
cd21_s2d = [] # specific entropy
cd21_dlsdlt2d = [] # dlnS/dlnT
for pressure_group, sub_df in cd21_subset.groupby(cd21_p):
    cd21_rho2d.append(10 ** sub_df[2] * 1000.0)  # convert log g/cm3 to kg/m3
    cd21_ad_grad2d.append(sub_df[9]) # unitless
    cd21_s2d.append(10 ** sub_df[4] * 1.0e6) # convert log MJ/kg/K to J/kg/K
    cd21_dlsdlt2d.append(sub_df[7]) # dlnS/dlnT, unit should be log(MJ/kg)/log(K)
cd21_p = np.array(cd21_p.unique())
cd21_t = np.array(cd21_t.unique())
# cd21_t_full = 10 ** cd21_data[0]  # convert log10 T to T in K
# cd21_p_full = 10 ** cd21_data[1] * 1.0e9  # convert log P in GPa to P in Pa
# cd21_t_full = np.array(cd21_t_full.unique())
# cd21_p_full = np.array(cd21_p_full.unique())
# print(np.shape(cd21_rho2d), np.shape(cd21_p_full), np.shape(cd21_t_full))
# cd21_interp_grid_rho = RegularGridInterpolator((cd21_p, cd21_t), cd21_rho2d)
# cd21_interp_grid_ad_grad = RegularGridInterpolator((cd21_p, cd21_t), cd21_ad_grad2d)
# cd21_interp_grid_s = RegularGridInterpolator((cd21_p, cd21_t), cd21_s2d)
cd21_interp_spline_rho = RectBivariateSpline(cd21_p, cd21_t, cd21_rho2d, kx=1, ky=1) # linear interpolation
cd21_interp_spline_ad_grad = RectBivariateSpline(cd21_p, cd21_t, cd21_ad_grad2d, kx=1, ky=1)
cd21_interp_spline_s = RectBivariateSpline(cd21_p, cd21_t, cd21_s2d, kx=1, ky=1)
cd21_interp_spline_dlnsdlnt = RectBivariateSpline(cd21_p, cd21_t, cd21_dlsdlt2d, kx=1, ky=1)
# print(np.shape(cd21_rho2d), np.shape(cd21_p), np.shape(cd21_t))
print('Completed')
print()

# CH4 and NH3 EOS from Bethkenhagen+2017
beth17_nh3 = np.genfromtxt("Data/Bethkenhagen2017_NH3_CH4/Bethkenhagen2017_NH3_EOS.txt")
beth17_nh3 = np.transpose(beth17_nh3)
beth17_nh3_p = beth17_nh3[0] * 1.0e9 # convert GPa to Pa
beth17_nh3_rho = beth17_nh3[1] * 1.0e3 # convert g/cm3 to kg/m3
beth17_nh3_interp = interp1d(beth17_nh3_p, beth17_nh3_rho, fill_value="extrapolate")

beth17_ch4 = np.genfromtxt("Data/Bethkenhagen2017_NH3_CH4/Bethkenhagen2017_CH4_EOS.txt")
beth17_ch4 = np.transpose(beth17_ch4)
beth17_ch4_p = beth17_ch4[0] * 1.0e9 # convert GPa to Pa
beth17_ch4_rho = beth17_ch4[1] * 1.0e3 # convert g/cm3 to kg/m3
beth17_ch4_interp = interp1d(beth17_ch4_p, beth17_ch4_rho, fill_value="extrapolate")

#=========================================================================
#====== numerical functions
#=========================================================================

def eos_Vinet(K0, rho, rho0, K0p):
    """
    Calculate P using the Vinet EOS (eqn 4 of S07).
    
    Parameters
    ----------
    K0 : float
        bulk modulus of a material, in Pa.
    rho : float
        density, in kg m-3
    rho0 : float
        ambient density, in kg m-3
    K0p : float
        derivative of K0, unitless.

    Returns
    -------
    P, float, pressure, in Pa.
    """
    eta = rho/rho0
    
    a = 3 * K0 * eta ** (2.0/3.0) * (1 - eta ** (-1.0/3.0))
    b = (3.0/2.0) * (K0p - 1) * (1 - eta ** (-1.0/3.0))
    return a * np.exp(b)


def eos_BME2(K0, rho, rho0):
    """
    Calculate P using the second-order BME EOS (eqn 1 of Zeng et al. 2016).

    Parameters
    ----------
    K0 : float
        bulk modulus of a material, in Pa.
    rho : float
        density, in kg m-3
    rho0 : float
        ambient density, in kg m-3

    Returns
    -------
    P, float, pressure, in Pa.

    """
    eta = rho / rho0
    p = 1.5 * K0 * (eta ** (7.0/3.0) - eta ** (5.0/3.0))
    return p


def eos_BME3(K0, rho, rho0, K0p):
    """
    Calculate P using the third-order BME EOS (eqn 5 of S07).
    
    Parameters
    ----------
    K0 : float
        bulk modulus of a material, in Pa.
    rho : float
        density, in kg m-3
    rho0 : float
        ambient density, in kg m-3
    K0p : float
        derivative of K0, unitless.

    Returns
    -------
    P, float, pressure, in Pa.
    """
    eta = rho/rho0
    
    a = (3.0/2.0) * K0 * (eta ** (7.0/3.0) - eta ** (5.0/3.0))
    b = 1 + (3.0/4.0) * (K0p - 4) * (eta ** (2.0/3.0) - 1)
    return a * b
    
                               
def eos_BME4(K0, rho, rho0, K0p, K0pp):
    """
    Calculate P using the fourth-order BME EOS (eqn 6 of S07).
    
    Parameters
    ----------
    K0 : float
        bulk modulus of a material, in Pa.
    rho : float
        density, in kg m-3
    rho0 : float
        ambient density, in kg m-3
    K0p : float
        derivative of K0, unitless.
    K0pp : float
        second derivative of K0, in Pa-1.

    Returns
    -------
    P, float, pressure, in Pa.
    """
    eta = rho/rho0
    
    a = (3.0/2.0) * K0 * (eta ** (7.0/3.0) - eta ** (5.0/3.0))
    b = (3.0/8.0) * K0 * (eta ** (2.0/3.0) - 1) ** 2.0
    c = K0p * K0pp + K0p * (K0p - 7.0) + (143.0/9.0)
    return eos_BME3(K0, rho, rho0, K0p) + a * b * c
    

def eos_Vinet_inverse(K0, P, rho0, K0p, rho_guess):
    """
    Calculate rho using the Vinet EOS (eqn 4 of S07) given a P.
    When there are multiple solutions, return the largest positive solution.
    
    Parameters
    ----------
    K0 : float
        bulk modulus of a material, in Pa.
    P : float
        pressure, in Pa
    rho0 : float
        ambient density, in kg m-3
    K0p : float
        derivative of K0, unitless.
    rho_guess : float
        a guess value of the solution that is supposed to be close to the
        solution. Use rho(r) in practice.

    Returns
    -------
    rho, float, density, in kg m-3.
    """    
    func = lambda rho : P - 3 * K0 * (rho/rho0) ** (2.0/3.0) * \
        (1 - (rho/rho0) ** (-1.0/3.0)) * \
        np.exp((3.0/2.0) * (K0p - 1) * (1 - (rho/rho0) ** (-1.0/3.0)))
        
    rho_solution = fsolve(func, rho_guess)
    
    if max(rho_solution) <= 0:
        raise ValueError("Vinet_inverse: No positive solution for rho!")
    else:
        return max(rho_solution)


def eos_BME2_inverse(K0, P, rho0, rho_guess):
    """
    Calculate rho using the second-order BME EOS (eqn 1 of Zeng et al. 2016).
    When there are multiple solutions, return the largest positive solution.
    
    Parameters
    ----------
    K0 : float
        bulk modulus of a material, in Pa.
    P : float
        pressure, in Pa.
    rho0 : float
        ambient density, in kg m-3.
    rho_guess : float
        a guess value of the solution that is supposed to be close to the
        solution. Use rho(r) in practice.

    Returns
    -------
    rho, float, density, in kg m-3.
    """
    func = lambda rho: P - 1.5 * K0 * ((rho/rho0)**(7.0/3.0) - (rho/rho0)**(5.0/3.0))
    
    rho_solution = fsolve(func, rho_guess)
    
    if max(rho_solution) <= 0:
        # return np.nan
        raise ValueError("BME2_inverse: No positive solution for rho!")
    else:
        return max(rho_solution)
    

def eos_BME3_inverse(K0, P, rho0, K0p, rho_guess):
    """
    Calculate rho using the third-order BME EOS (eqn 5 of S07).
    When there are multiple solutions, return the largest positive solution.
    
    Parameters
    ----------
    K0 : float
        bulk modulus of a material, in Pa.
    P : float
        pressure, in Pa.
    rho0 : float
        ambient density, in kg m-3.
    K0p : float
        derivative of K0, unitless.
    rho_guess : float
        a guess value of the solution that is supposed to be close to the
        solution. Use rho(r) in practice.

    Returns
    -------
    rho, float, density, in kg m-3.
    """
    func = lambda rho: P - (3.0/2.0) * \
        K0 * ((rho/rho0) ** (7.0/3.0) - (rho/rho0)  ** (5.0/3.0)) * \
        (1 + (3.0/4.0) * (K0p - 4) * ((rho/rho0)  ** (2.0/3.0) - 1))
    
    rho_solution = fsolve(func, rho_guess)
    
    # print(rho_solution)
    if max(rho_solution) <= 0:
        # return np.nan
        raise ValueError("BME3_inverse: No positive solution for rho!")
    else:
        return max(rho_solution)
    

def eos_BME4_inverse(K0, P, rho0, K0p, K0pp, rho_guess):
    """
    Calculate rho using the third-order BME EOS (eqn 5 of S07).
    When there are multiple solutions, return the largest positive solution.
    
    Parameters
    ----------
    K0 : float
        bulk modulus of a material, in Pa.
    P : float
        pressure, in Pa
    rho0 : float
        ambient density, in kg m-3
    K0p : float
        derivative of K0, unitless.
    K0pp : float
        second derivative of K0, in Pa-1.
    rho_guess : float
        a guess value of the solution that is supposed to be close to the
        solution. Use rho(r) in practice.

    Returns
    -------
    rho, float, density, in kg m-3.
    """
    func = lambda rho: P - \
        ((3.0/2.0) * \
        K0 * ((rho/rho0) ** (7.0/3.0) - (rho/rho0) ** (5.0/3.0)) * \
        (1 + (3.0/4.0) * (K0p - 4) * ((rho/rho0) ** (2.0/3.0) - 1)) + \
        (3.0/2.0) * K0 * ((rho/rho0) ** (7.0/3.0) - (rho/rho0) ** (5.0/3.0)) * \
        (3.0/8.0) * K0 * ((rho/rho0) ** (2.0/3.0) - 1) ** 2.0 * \
        (K0p * K0pp + K0p * (K0p - 7) + (143.0/9.0)))
    
    rho_solution = fsolve(func, rho_guess)
    
    if max(rho_solution) <= 0:
        raise ValueError("BME4_inverse: No positive solution for rho!")
    else:
        return max(rho_solution)
    
    
# def eos_homogeneous(rho, component):
#     """
#     Generate the EOS for homogeneous planets with only one building
#     component.

#     Parameters
#     ----------
#     rho : float
#         density, in kg m-3
#     component : string
#         "h2o", "fe", or "mgsio3".

#     Returns
#     -------
#     P, float, pressure, in Pa.

#     """
#     assert component in ["h2o", "fe", "mgsio3"]  # acceptable components
    
#     if component == "h2o":
#         if rho >= 1543.1416638680932:
#             return eos_BME3(23.7e9, rho, 1.46e3, 4.15)
#         else:
#             return np.nan  # pressure is degenerate in this range
#     elif component == "fe":
#         if rho >= 7916.819159263009:
#             return eos_BME3(162.5e9, rho, 7.86e3, 5.5)
#         else:
#             return np.nan  # pressure is degenerate in this range
#     elif component == "mgsio3":
#         if rho >= 3324.597932270942:
#             return eos_BME3(125.0e9, rho, 3.22e3, 5.0)
#         else:
#             return np.nan  # pressure is degenerate in this range
    

def eos_TFD(A, Z, rho):
    """
    Thomas-Fermi-Dirac EOS based on Salpeter & Zapolsky (1967; 1969).

    Parameters
    ----------
    A : float
        atmoic weight of element.
    Z : float
        charge of element.
    rho : float
        density in kg m-3.

    Returns
    -------
    Pressure, in Pa.

    """
    rho0 = (32.0/3.0) * np.pi ** (-3.0) * A * Z * amu * a0 ** (-3.0)
    
    P0 = Z ** (10.0/3.0) * (2.0 ** 8.0 * (2.0 * np.pi) ** (1.0/3.0))
    P0 /= 15 * np.pi ** 4.0
    P0 *= ((qe ** 2.0) / (hbar * c)) ** 2.0
    P0 *= (me * c ** 2.0) / a0 ** 3.0
    
    phi = (1.0/20.0) * 3.0 ** (1.0/3.0)
    phi += (1.0/8.0) * ((1.0/4.0) * np.pi ** (-2.0) * Z ** (-2.0)) ** (1.0/3.0)
    
    return P0 * ((rho / rho0) ** (1.0/3.0) - phi) ** 5.0


def eos_TFD_inverse(A, Z, P):
    """
    Inversed Thomas-Fermi-Dirac EOS based on Salpeter & Zapolsky (1967; 1969).

    Parameters
    ----------
    A : float
        atmoic weight of element.
    Z : float
        charge of element.
    P : float
        Pressure, in Pa.

    Returns
    -------
    Density in kg m-3.

    """
    rho0 = (32.0/3.0) * np.pi ** (-3.0) * A * Z * amu * a0 ** (-3.0)
    
    P0 = Z ** (10.0/3.0) * (2.0 ** 8.0 * (2.0 * np.pi) ** (1.0/3.0))
    P0 /= 15 * np.pi ** 4.0
    P0 *= ((qe ** 2.0) / (hbar * c)) ** 2.0
    P0 *= (me * c ** 2.0) / a0 ** 3.0
    
    phi = (1.0/20.0) * 3.0 ** (1.0/3.0)
    phi += (1.0/8.0) * ((1.0/4.0) * np.pi ** (-2.0) * Z ** (-2.0)) ** (1.0/3.0)
    
    # P = rho0 * ((P/P0) ** (1.0/5.0) + phi) ** 3.0
    # print(P)
    return rho0 * ((P/P0) ** (1.0/5.0) + phi) ** 3.0


def eos_TFD_SZ67_inverse(A, Z, P):
    """
    Inversed Thomas-Fermi-Dirac EOS based on Salpeter & Zapolsky (1967)
    equation 42-46.

    Parameters
    ----------
    A : float
        atmoic weight of element.
    Z : float
        charge of element.
    P : float
        Pressure, in Pa.

    Returns
    -------
    Density in kg m-3.

    """
    # convert P into dyn/cm2 for consistency
    P *= 10.0
    
    # define parameters epsilon and phi
    epsilon = (3.0 / (32.0 * np.pi ** 2.0 * Z ** 2.0)) ** (1.0/3.0)
    phi = (3.0 ** (1.0/3.0)) / 20.0 + epsilon / (4.0 * 3.0 ** (1.0/3.0))
    
    # define the values of gamma_i_n based on Table 1 in SZ67
    # n = 2
    g02 = 1.512e-2
    g12 = 8.955e-2
    g22 = 1.090e-1
    g32 = 5.089
    g42 = -5.980
    # n = 3
    g03 = 2.181e-3
    g13 = -4.015e-1
    g23 = 1.698
    g33 = -9.566
    g43 = 9.873
    # n = 4
    g04 = -3.328e-4
    g14 = 5.167e-1
    g24 = -2.369
    g34 = 1.349e1
    g44 = -1.427e1
    # n = 5
    g05 = -1.384e-2
    g15 = -6.520e-1
    g25 = 3.529
    g35 = -2.095e1
    g45 = 2.264e1
    
    # calculate x_0(0) based on eqn(43)
    x00 = (8.884e-3 + epsilon ** (1.0/2.0) * 4.988e-1 + epsilon * 5.2604e-1) ** (-1.0)
    
    # calculate alpha based on eqn(45c)
    alpha = 1.941e-2 - epsilon ** (1.0/2.0) * 6.277e-2 + epsilon * 1.076
    alpha = 1.0 / alpha
    
    # calculate beta values based on epsilon and gamma
    beta0 = x00 * phi - 1  # (45a)
    beta1 = beta0 * alpha + ((1 + beta0) / phi)  # (45b)
    
    beta2 = (g02 + g12 * epsilon ** (1.0/2.0) + g22 * epsilon +
             g32 * epsilon ** (3.0/2.0) + g42 * epsilon ** 2.0) ** 2.0
    beta2 = 1.0 / beta2
    
    beta3 = (g03 + g13 * epsilon ** (1.0/2.0) + g23 * epsilon +
             g33 * epsilon ** (3.0/2.0) + g43 * epsilon ** 2.0) ** 3.0
    beta3 = 1.0 / beta3
    
    beta4 = (g04 + g14 * epsilon ** (1.0/2.0) + g24 * epsilon +
            g34 * epsilon ** (3.0/2.0) + g44 * epsilon ** 2.0) ** 4.0
    beta4 = 1.0 / beta4
    
    beta5 = (g05 + g15 * epsilon ** (1.0/2.0) + g25 * epsilon +
            g35 * epsilon ** (3.0/2.0) + g45 * epsilon ** 2.0) ** 5.0
    beta5 = 1.0 / beta5
    
    # calculate xi from pressure P and P0
    P0 = 9.524e13  # in dyn/cm2
    xi = (P / P0) ** (1.0/5.0) * Z ** (-2.0/3.0)
    
    # calculate x0(xi) from xi, phi, and the beta parameters
    x0xi = 1 + np.exp(-alpha * xi) * (beta0 + beta1 * xi + beta2 * xi ** 2.0
                                      + beta3 * xi ** 3.0 + beta4 * xi ** 4.0
                                      + beta5 * xi ** 5.0)
    # x0xi = 1.0
    x0xi /= xi + phi
    
    # finally, calculate rho(xi) based on A, Z, and x0xi
    rho = 3.886 * ((A * Z) / x0xi ** 3.0)  # density in g/cm3
    
    return rho * 1000.0  # convert into kg/m3


def eos_holzapfel_2018(rho, A, Z, rho0, P_FG0, K0, K0p, c0, c2):
    """
    P = f(rho)
    
    Holzapfel (2018) EOS. This is the so-called second-order adpated
    polynomial EOS (AP2 EOS) adpoted in Zeng et al. (2021), as a
    replacement to the PREM + BM2 EOS used in Zeng et al. (2016).
    
    This function is based on Equation (5-7) in Zeng et al. (2021).

    Parameters
    ----------
    rho : float
        Density in kg m-3.
    A : int
        Atomic mass number (total number of neutron + proton).
    Z : int
        Atomic number (number of proton).
    rho0 : float
        Reference density in kg m-3.
    P_FG0 : float
        Fermi-gas pressure at ambient (uncompressed) density in GPa.
    K0 : float
        Bulk modulus in GPa.
    K0p : float
        Pressure derivative of K0, dimensionless.
    c0 : float
        Coefficient determined by K0 and K0p, dimensionless.
    c2 : float
        Coefficient determined by K0 and K0p, dimensionless.

    Returns
    -------
    Pressure in Pa.

    """
    eta = (rho0 / rho) ** (1.0 / 3.0)  # dimensionless variable
    p = 3 * K0 * ((1-eta) / eta**5.0)
    p *= np.exp(c0 * (1-eta))
    p *= 1 + c2 * eta * (1-eta)  # pressure in GPa
    return p * 1.0e9  # convert to Pa
    

def fe_transition_zone(p):
    """
    Given a pressure, return the tabulated density in the transition zone
    of Fe.

    Parameters
    ----------
    p : float
        pressure in Pa.

    Returns
    -------
    rho, float, density, in kg m-3.

    """
    fe_transition = np.genfromtxt('Data/EOS/iron_transition_zone')
    fe_transition = np.transpose(fe_transition)
    transition_p = fe_transition[0]
    transition_rho = fe_transition[1]
    transition_p_to_rho = interp1d(transition_p, transition_rho, kind='linear', fill_value="extrapolate")
    return transition_p_to_rho(p)
    
    
def eos_homogeneous_inverse(p, component, rho_guess):
    """
    Generate the inversed EOS for homogeneous planets with only one building
    component.
    
    Each material has 3 segments: constant density segment (low P),
    experimental fit segment (BME or Vinet, medium P), TFD segment (high P).
    The transition pressures are:
        
        water: 1472639987.7288897 & 1.089441e+11 Pa
        silicate: 5.0e+9 & 5.241702e+11 Pa
        
        Fe is special because it has two phases (epsilon and alpha), the
        two Vinet EOS intersects at 1.433231e+12 Pa
        In addition, inverse Vinet expression fails at low pressures, so
        the transition zone between 1.1938603e9 and 20.0e9 Pa is calculated
        from the forward Vinet formula
        
        So Fe essentially has five segments:
            0 - 1.1938603e9 Pa: constant density
            1.1938603e9 - 20.0e9 Pa: transition zone
            20.0e9 - 1.433231e+12 Pa: Vinet alpha
            1.433231e+12 - 2.062680e+13 Pa: Vinet epsilon
            > 2.062680e+13 Pa: TFD 
        
        The transition zone EOS is tabulated while the Vinet and TFD EOS are
        calculated inversely by solving rho given a known P
        
    Parameters
    ----------
    p : float
        pressure, in Pa.
    component : string
        "h2o", "fe", or "mgsio3".
    rho_guess : float
        a guess of rho solution, in kg m-3.

    Returns
    -------
    rho, float, density, in kg m-3.

    """
    assert component in ["h2o", "fe", "mgsio3"]  # acceptable components
    
    if component == "h2o":
        if p >= 1.089441e+11:
            return eos_TFD_SZ67_inverse(18.0, 10.0, p)
        elif p >= 1472639987.7288897 and p < 1.089441e+11:
            return eos_BME3_inverse(23.7e9, p, 1.46e3, 4.15, rho_guess)
        else:
            return 1543.1416638680932  # constant density at low pressure
    elif component == "fe":
        if p >= 2.062680e+13:
            return eos_TFD_SZ67_inverse(56.0, 26.0, p)
        elif p >= 1.433231e+12 and p < 2.062680e+13:
            return eos_Vinet_inverse(156.2e9, p, 8.30e3, 6.08, rho_guess)
        elif p >= 20.0e9 and p < 1.433231e+12:
            return eos_Vinet_inverse(162.5e9, p, 7.86e3, 5.5, rho_guess)
        elif p >= 1.1938603e9 and p < 20.0e9:
            return fe_transition_zone(p)
        else:
            return 7916.819159263009  # constant density at low pressure
    elif component == "mgsio3":
        if p >= 5.241702e+11:
            return eos_TFD_SZ67_inverse(20.0, 10.0, p)
        elif p >= 5.0e+9 and p < 5.241702e+11:
            return eos_BME3_inverse(125.0e9, p, 3.22e3, 5.0, rho_guess)
        else:
            return 3324.597932270942  # constant density at low pressure


def eos_experimental_inverse(p, component, rho_guess):
    """
    Generate the inversed EOS for homogeneous planets with only one building
    component.
    
    This function is similar to eos_homogeneous_inverse(), but only has
    two segments - constant density and experimental fit.
        
    Parameters
    ----------
    p : float
        pressure, in Pa.
    component : string
        "h2o", "fe", or "mgsio3".
    rho_guess : float
        a guess of rho solution, in kg m-3.

    Returns
    -------
    rho, float, density, in kg m-3.

    """
    assert component in ["h2o", "fe", "mgsio3"]  # acceptable components
    
    if component == "h2o":
        if p >= 1472639987.7288897:
            return eos_BME3_inverse(23.7e9, p, 1.46e3, 4.15, rho_guess)
        else:
            return 1543.1416638680932  # constant density at low pressure
    elif component == "fe":
        if p >= 1.433231e+12:
            return eos_Vinet_inverse(156.2e9, p, 8.30e3, 6.08, rho_guess)
        elif p >= 20.0e9 and p < 1.433231e+12:
            return eos_Vinet_inverse(162.5e9, p, 7.86e3, 5.5, rho_guess)
        elif p >= 1.1938603e9 and p < 20.0e9:
            return fe_transition_zone(p)
        else:
            return 7916.819159263009  # constant density at low pressure
    elif component == "mgsio3":
        if p >= 5.0e+9:
            return eos_BME3_inverse(125.0e9, p, 3.22e3, 5.0, rho_guess)
        else:
            return 3324.597932270942  # constant density at low pressure
        

def eos_carbon(p):
    """ Isothermal EOS of pure carbon. Graphite-diamond phase transition included. """
    graphite_diamond_phase_transition = 10.0 * 1.0e9 # 10 GPa, where phase transition occurs
    if p <= graphite_diamond_phase_transition:
        return s07_eos_graphite_p_to_rho(p)
    else:
        return sw22_eos_diamond_p_to_rho(p)
    

def eos_sic(p):
    """ Isothermal SiC EOS. Phase transition between zinc-blende and rock-salt structures assumed at 67.5 GPa. """
    sic_phase_transition = 67.5 * 1.0e9 # see Miozzi et al. (2018) Fig. 3
    if p <= sic_phase_transition:
        return sic_eos_zincBlende_p_to_rho(p)
    else:
        return sic_eos_rockSalt_p_to_rho(p)

        
def eos_tabulated(p, component):
    """
    Given a pressure p, return the density rho from a tabulated EOS.

    Parameters
    ----------
    p : float
        pressure in Pa.
    component : string
        "h2o", "fe", or "mgsio3".

    Returns
    -------
    rho, float, density, in kg m-3.

    """
    # list acceptable components
    assert component in ["h2o", "fe", "mgsio3", # Seager+2007
                         "mgsio3_pv", "mgsio3_iron-rich_pv", # Seager+2007, different phases of MgSiO3
                         "Zeng2016_mantle", "Zeng2016_core", # Zeng+2016
                         "Zeng2021_mantle", "Zeng2021_core", "Zeng2021_water", # Zeng+2021
                         "Smith2018_solidFe", # Smith+2018, Vinet fit to experiment
                         "Grant2021_liquidFe", # Grant+2021, Vinet fit to experiment
                         "OganovOno2004", # Oganov & Ono 2004, Vinet fit to ab initio calculation of MgSiO3 (ppv)
                         "HM89_rock", # Hubbard & Marley 1989, analytical rock EOS
                         "carbon", "SiC" # pure carbon and SiC EOSs calling eos_carbon() and eos_sic()
                         ], f"<{component}> is not in the acceptable list!"
    
    # Seager et al. 2007 EOS
    if component == "h2o":
        return tabulated_p_to_rho_h2o(p)
    elif component == "fe":
        return tabulated_p_to_rho_fe(p)
    elif component == "mgsio3":
        return tabulated_p_to_rho_silicate(p)
    elif component == "mgsio3_pv":
        return tabulated_silicate_pv_p_to_rho(p)
    elif component == "mgsio3_iron-rich_pv":
        return tabulated_silicate_pvi_p_to_rho(p)
    # Li Zeng's EOS from 2016 and 2021 paper
    elif component == "Zeng2016_mantle":
        return Zeng2016_eos_mantle_p_to_rho(p)
    elif component == "Zeng2016_core":
        return Zeng2016_eos_core_p_to_rho(p)
    elif component == "Zeng2021_mantle":
        return Zeng2021_eos_mantle_p_to_rho(p)
    elif component == "Zeng2021_core":
        return Zeng2021_eos_core_p_to_rho(p)
    elif component == "Zeng2021_water":
        return Zeng2021_eos_water_p_to_rho(p)
    # experimental iron EOS
    elif component == "Smith2018_solidFe":
        return Smith2018_eos_core_p_to_rho(p) 
    elif component == "Grant2021_liquidFe":
        return Grant2021_eos_core_p_to_rho(p)
    # ab initio silicate EOS
    elif component == "OganovOno2004":
        return OgaOno2004_eos_mantle_p_to_rho(p)
    # Hubbard & Marley (1989) rock EOS
    elif component == "HM89_rock":
        return tabulated_hm89_rock(p)
    # carbon and carbide EOSs
    elif component == "carbon":
        return eos_carbon(p)
    elif component == "SiC":
        return eos_sic(p)


def eos_ideal_gas(p, t, Rs):
    """
    Using the ideal gas law: p = rho R_specific T, or alternatively rho = p / RT to calculate ideal gas density.
    
    :param p: float, pressure in Pa
    :param t: float, temperature in K
    :param Rs: float, specific gas constant in SI (J kg-1 K)
    """
    return p / (Rs * t)


def eos_AQUA(p, t):
    """
    Given pressure p and temperature t, return the density rho and adiabatic
    gradient ad_grad at p, t based on the AQUA EOS.

    Parameters
    ----------
    p : float
        Pressure in Pa.
    t : float
        Temperature in K.

    Returns
    -------
    rho, ad_grad. Both are floats

    """
    rho = aqua_interp_spline_rho(p, t)
    ad_grad = aqua_interp_spline_ad_grad(p, t)
    # if rho <= 1500.0: # handles vapor phase boundary & low density regions where RectBivariateSpline gives unphysical negative values
    #     rho = aqua_interp_grid_rho(np.array([p, t]))
    #     ad_grad = aqua_interp_grid_ad_grad(np.array([p, t]))
    return float(rho), float(ad_grad)


def eos_french2009_h2o(p, t):
    """ French et al. (2009) QMD simulation EOS for H2O. Note that their database does not provide adiabatic
    gradient, so only density is returned. Note that this EOS has P, T boundaries as follows:
    - P_min, P_max = 1.8600E+09 Pa, 9.8710E+12 Pa
    - T_min, T_max = 1000 K, 24000 K
    If the given p, t values are above/below these boundaries, replace p, t with the min/max values to avoid
    out of bounds.
    """
    pmin = 1.8600E+09; pmax = 9.8710E+12
    tmin = 1000.0; tmax = 24000.0
    if p < pmin:
        p = pmin
    if p > pmax:
        p = pmax
    if t < tmin:
        t = tmin 
    if t > tmax:
        t = tmax

    # compute rho from interpolation
    # rho = french09_interp_spline_rho(p, t)
    # rho = french09_interp_nd_rho(p, t)
    # rho = french09_interp_ct2d_rho(p, t)
    # rho = n13_u1_interp1d_rho(p) # has no T dependence, simply interpolating rho(P) from N13 U1
    rho = n13_u2_interp1d_rho(p)
    return float(rho)


def eos_cd21_hhe(p, t):
    """
    Given pressure p and temperature t, return the density rho, adiabatic
    gradient ad_grad, and specific entropy s at p, t based on the Chabrier 
    & Debras 2021 H-He mixture EOS.

    Parameters
    ----------
    p : float
        Pressure in Pa.
    t : float
        Temperature in K.

    Returns
    -------
    rho, ad_grad. Both are floats

    """
    # rho = cd21_interp_grid_rho(np.array([p, t]))
    # ad_grad = cd21_interp_grid_ad_grad(np.array([p, t]))
    # s = cd21_interp_grid_s(np.array([p, t]))
    rho = cd21_interp_spline_rho(p, t)
    ad_grad = cd21_interp_spline_ad_grad(p, t)
    s = cd21_interp_spline_s(p, t)
    return float(rho), float(ad_grad), float(s)


def avl_mass_fraction(rp, res=10000):
    """
    Given the radius of a planet, return three scipy.interp1d functions representing X_rock, X_water, and X_hhe of a
    planet. Where X_rock(r) is a monotonically decreasing function from 1 to 0, X_hhe(r) is a monotonically
    increasing function from 0 to 1, and X_water(r) = 1 - X_rock(r) - X_hhe(r).
    
    The X_i(r) functions adopt form of equation (12) from Movshovitz et al. (2020). The curvature is a random number
    between 0 and 5. All negative X_i(r) are replaced with zero. 
    """
    rock_curv = np.random.rand() * 5.0 # curvature a_i
    hhe_curv = np.random.rand() * 5.0
    r = np.linspace(0, 1.0, res) # dimensionless radius between 0 and 1, where 1 represents rp
    X_rock = rock_curv * (r**2.0 - 1.0) + (-1.0 - rock_curv)*(r-1.0) 
    X_rock = np.where(X_rock<0, 0, X_rock)
    X_hhe = 1.0 - (hhe_curv * (1.0 - r**2.0) + (1.0 - hhe_curv)*(1.0-r))
    X_hhe = np.where(X_hhe<0, 0, X_hhe)
    X_water = 1.0 - X_rock - X_hhe
    
    X_rock_interp = interp1d(r*rp, X_rock, kind='linear', bounds_error=False, fill_value=0.0)
    X_water_interp = interp1d(r*rp, X_water, kind='linear', bounds_error=False, fill_value=0.0)
    X_hhe_interp = interp1d(r*rp, X_hhe, kind='linear', bounds_error=False, fill_value=1.0) # when r is out of bound, assume pure H/He atmosphere
    return X_rock_interp, X_water_interp, X_hhe_interp


def avl_planetary_ice(p, t, x_h2o, x_ch4, x_nh3, x_hhe=0.0):
    """ Using AVL, calculate the density of planetary ice mixture composed of H2O, CH4, and NH3. The number fractions
    are defined by x_i. Need to convert x_i to mass fractions before applying AVL. Because no accurate EOSs exist
    for CH4 and NH3, this is done by scaling H2O density from AQUA by molecular_mass_mixture / molecular_mass_water. 
    
    Allow mixing of H/He by using a fourth number mixing fraction parameter x_hhe.
    """
    # calculate new mean molecular weight
    mx_h2o = x_h2o * 18.01528
    mx_ch4 = x_ch4 * 16.04
    mx_nh3 = x_nh3 * 17.031
    # mx_he:mx_h = 0.275:0.715, so x_he:x_h=0.08661417322834647:0.9133858267716535
    mx_hhe = x_hhe * 0.08661417322834647 * 4.0 + x_hhe * 0.9133858267716535 * 1.0
    mx_sum = mx_h2o + mx_ch4 + mx_nh3 + mx_hhe
    mx_h2o /= mx_sum
    mx_ch4 /= mx_sum
    mx_nh3 /= mx_sum
    mx_hhe /= mx_sum
    
    # get the density of each component
    rho_h2o, ad_grad = eos_AQUA(p, t) # use ad_grad of H2O
    rho_ch4 = beth17_ch4_interp(p)
    rho_nh3 = beth17_nh3_interp(p)
    if x_hhe != 0.0:
        rho_hhe, _, _ = eos_cd21_hhe(p, t)
    else:
        rho_hhe = 1.0 # save time by not calling the EOS function

    # calculate mixture rho using LMA/AVL
    one_over_rhomix = mx_h2o/rho_h2o + mx_ch4/rho_ch4 + mx_nh3/rho_nh3 + mx_hhe/rho_hhe
    rho_mix = 1 / one_over_rhomix

    return rho_mix, ad_grad


def avl_h2o_hhe(p, t, mx_h2o, mx_hhe):
    """
    The same as AVL function for planetary ice above, but for H2O and H/He mixture. Note that the fractions as given
    as mass fractions (mx). Because there are 2 H atoms in H2O, corrections are required to convert Z values (mass
    fraction of heavy elemnts, O in this case) into mx values.
    """    
    assert mx_h2o + mx_hhe == 1.0
    
    # if either mass fraction is 1, drop back to pure EOS
    if mx_h2o == 1:
        return eos_AQUA(p, t)
    if mx_hhe == 1:
        return eos_cd21_hhe(p, t)
    
    # convert mass fractions into number fractions
    x_h2o = mx_h2o / 18.01528
    x_hhe = mx_hhe / (0.08661417322834647 * 4.0 + 0.9133858267716535 * 2.0) # roughly 2.17, for comparison, Jupiter's MMW is 2.22
    
    rho_h2o, ad_grad_h2o = eos_AQUA(p, t)
    rho_hhe, ad_grad_hhe, _ = eos_cd21_hhe(p, t)
    rho_AVL_reciprocal = mx_h2o/rho_h2o + mx_hhe/rho_hhe
    # return ad_grad of the dominant species
    if x_h2o >= x_hhe:
        ad_grad = ad_grad_h2o
    else:
        ad_grad = ad_grad_hhe

    return 1/rho_AVL_reciprocal, ad_grad


def avl_eos(p, t, xi):
    """
    Given the EOS of three materials (rock, water, H/He) and the mass fraction of each material, calculate the
    AVL (additive volume law) EOS of the mixture material. The AVL density is calculated using
    
    1/rho_AVL = x1/rho1(P,T) + x2/rho2(P,T) + x3/rho3(P,T) + ...
    
    where xi is the mass fraction and rhoi the density of each individual component.
    
    For rock, use Zeng2021_eos_mantle_p_to_rho(P) # Note: change to a rock EOS with higher iron mass fraction?
    For water, use eos_AQUA(P, T)
    For H/He, use eos_cd21_hhe(P, T)
    
    [p] is pressure in Pa, [t] is temperature in K, [xi] is a list of three floats corresponding to x_rock,
    x_water, and x_hhe.
    """
    assert len(xi) == 3
    
    rho_rock = Zeng2021_eos_mantle_p_to_rho(p)
    rho_water, _ = eos_AQUA(p, t)
    rho_hhe, _, _ = eos_cd21_hhe(p, t)
    rho_AVL_reciprocal = xi[0]/rho_rock + xi[1]/rho_water + xi[2]/rho_hhe
    return 1 / rho_AVL_reciprocal


def is_liquid_Kraus2022(P, T):
    """ Given the P, T of a layer, use the Kraus+2022 iron melt curve to judge if iron should be liquid or
    solid. Return True if is liquid, False otherwise. Note that P is inputted in Pa, but  should be 
    converted to GPa. """
    P /= 1.0e9 # convert Pa to GPa
    Tm = 5530.0 * ((P - 260.0)/293.0 + 1)**0.552 # melting temperature at this pressure
    return T > Tm # True if greater than melting temperature, i.e. melted


def find_eos_intersection():
    """
    Given two EOS, find where they intersect.

    Returns
    -------
    Prints P, pressure at which the two EOS gives the same density.

    """
    P_list = list(np.logspace(7, 19, 10000))
    rho_guess_list = list(np.logspace(2, 8, 10000))
    water_diff_list = []
    
    for i in range(len(P_list)-1):
        # iron, Vinet vs TFD
        iron_rho_Vinet_1 = eos_Vinet_inverse(156.2e9, P_list[i], 8.30e3, 6.08,
                                 rho_guess_list[i])
        iron_rho_Vinet_2 = eos_Vinet_inverse(156.2e9, P_list[i+1], 8.30e3, 6.08,
                                 rho_guess_list[i+1])
        iron_rho_TFD_1 = eos_TFD_SZ67_inverse(56.0, 26.0, P_list[i])
        iron_rho_TFD_2 = eos_TFD_SZ67_inverse(56.0, 26.0, P_list[i+1])
        if ((iron_rho_Vinet_1 < iron_rho_TFD_1) and (iron_rho_Vinet_2 > iron_rho_TFD_2)) or \
            ((iron_rho_Vinet_1 > iron_rho_TFD_1) and (iron_rho_Vinet_2 < iron_rho_TFD_2)):
                avg_P = (P_list[i] + P_list[i+1]) / 2.0
                print("iron TFD vs Vinet", "{:.6e}".format(avg_P))
                
        # iron, Vinet epsilon (hcp) phase vs alpha (bcc) phase
        iron_rho_Vinet_epsilon_1 = eos_Vinet_inverse(156.2e9, P_list[i], 8.30e3, 6.08,
                                                     rho_guess_list[i])
        iron_rho_Vinet_epsilon_2 = eos_Vinet_inverse(156.2e9, P_list[i+1], 8.30e3, 6.08,
                                                     rho_guess_list[i+1])
        iron_rho_Vinet_alpha_1 = eos_Vinet_inverse(162.5e9, P_list[i], 7.86e3, 5.5,
                                                     rho_guess_list[i])
        iron_rho_Vinet_alpha_2 = eos_Vinet_inverse(162.5e9, P_list[i+1], 7.86e3, 5.5,
                                                     rho_guess_list[i+1])
        if ((iron_rho_Vinet_epsilon_1 < iron_rho_Vinet_alpha_1) and (iron_rho_Vinet_epsilon_2 > iron_rho_Vinet_alpha_2)) or \
            ((iron_rho_Vinet_epsilon_1 > iron_rho_Vinet_alpha_1) and (iron_rho_Vinet_epsilon_2 < iron_rho_Vinet_alpha_2)):
                avg_P = (P_list[i] + P_list[i+1]) / 2.0
                print("iron Vinet epsilon vs alpha", "{:.6e}".format(avg_P))
        
        # water, BME3 vs TFD
        water_rho_BME3_1 = eos_BME3_inverse(23.7e9, P_list[i], 1.46e3, 4.15,
                                            rho_guess_list[i])
        water_rho_BME3_2 = eos_BME3_inverse(23.7e9, P_list[i+1], 1.46e3, 4.15,
                                            rho_guess_list[i+1])
        water_rho_TFD_1 = eos_TFD_SZ67_inverse(18.0, 10.0, P_list[i])
        water_rho_TFD_2 = eos_TFD_SZ67_inverse(18.0, 10.0, P_list[i+1])
        if ((water_rho_BME3_1 < water_rho_TFD_1) and (water_rho_BME3_2 > water_rho_TFD_2)) or \
            ((water_rho_BME3_1 > water_rho_TFD_1) and (water_rho_BME3_2 < water_rho_TFD_2)):
                avg_P = (P_list[i] + P_list[i+1]) / 2.0
                print("water", "{:.6e}".format(avg_P))
        water_diff_list.append(np.abs(water_rho_BME3_1 - water_rho_TFD_1))
        
        # mgsio3, BME3 vs TFD
        silicate_rho_BME3_1 = eos_BME3_inverse(125.0e9, P_list[i], 3.22e3, 5.0,
                                            rho_guess_list[i])
        silicate_rho_BME3_2 = eos_BME3_inverse(125.0e9, P_list[i+1], 3.22e3, 5.0,
                                            rho_guess_list[i+1])
        silicate_rho_TFD_1 = eos_TFD_SZ67_inverse(20.0, 10.0, P_list[i])
        silicate_rho_TFD_2 = eos_TFD_SZ67_inverse(20.0, 10.0, P_list[i+1])
        if ((silicate_rho_BME3_1 < silicate_rho_TFD_1) and (silicate_rho_BME3_2 > silicate_rho_TFD_2)) or \
            ((silicate_rho_BME3_1 > silicate_rho_TFD_1) and (silicate_rho_BME3_2 < silicate_rho_TFD_2)):
                avg_P = (P_list[i] + P_list[i+1]) / 2.0
                print("silicate", "{:.6e}".format(avg_P))
    
    # find index of the minimum difference for water
    min_diff = min(water_diff_list)
    min_ind = list.index(water_diff_list, min_diff)
    print("water", "{:.6e}".format(P_list[min_ind]))


def main():
    # find_eos_intersection()
    # print(eos_cd21_hhe(100.0, 4.4668359215096346e-08))
    pass


if __name__ == '__main__':
    main()
    
    
    
    
    
    