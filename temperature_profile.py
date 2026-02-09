#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 17:12:40 2022

@author: linzifan

Compute the temperature profile T(r) or T(m) in the interior of an planet.
"""

#=========================================================================
#====== import necessary packages
#=========================================================================

import numpy as np
import pandas as pd
from EOS import *
from simpleMR import *
import plotMR as pltmr
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy import optimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
hfont = {'fontname':'times'}

#=========================================================================
#====== read in data
#=========================================================================

# Freedman et al. 2008 mean opacities for H/He
print('Loading Freedman et al. 2008 H/He mean opacities ...')
f08_fn = 'Data/Freedman2008_H_opacity/Freedman2008_H_opacity_MH0.0_rect_extended_1.0e13Pa.txt'
f08_data = pd.read_table(f08_fn, skiprows=7, delim_whitespace=True, header=None)
# print(f08_data)
f08_p = f08_data[1] * 0.1 # convert dyne/cm2 to Pa
f08_t = f08_data[0]
f08_kappaR_2d = []
for p_group, sub_df in f08_data.groupby(f08_p):
    f08_kappaR_2d.append(sub_df[3])
f08_kappaR_2d = np.array(f08_kappaR_2d, dtype=object)
f08_p = np.array(f08_p.unique())
f08_t = np.array(f08_t.unique())
f08_interp_grid_kappaR = RegularGridInterpolator((f08_p, f08_t), f08_kappaR_2d) # note that kappaR is in cm2/g
print('Completed')
print()

# read Nettelmann+2013 U1 data, convert to the same (SI) unit, and overplot with my model
n13_u1 = np.genfromtxt('Data/Nettelmann2013_table_U1.dat')
n13_u1 = np.transpose(n13_u1)
n13_u1_r = n13_u1[2] * Re # convert Re to m
n13_u1_t = n13_u1[3] # temperature
n13_u1_rt_interp = interp1d(n13_u1_r, n13_u1_t, kind='linear', fill_value="extrapolate")

# the same for N13 U2
n13_u2 = np.genfromtxt('Data/Nettelmann2013_table_U2.dat')
n13_u2 = np.transpose(n13_u2)
n13_u2_r = n13_u2[2] * Re # convert Re to m
n13_u2_t = n13_u2[3] # temperature
n13_u2_rt_interp = interp1d(n13_u2_r, n13_u2_t, kind='linear', fill_value="extrapolate")

#=========================================================================
#====== useful constants, in SI
#=========================================================================

G = 6.67408e-11  # gravitational constant, SI
Re = 6371000.0  # Earth radius, m
Me = 5.97e24  # Earth mass, kg
R = 8.31446261815324 # universal gas constant, J mol-1 K-1

#=========================================================================
#====== numerical functions
#=========================================================================

def tabulated_tp(r):
    """ Hardcoded T(P) profile for model validation. """
    return n13_u2_rt_interp(r)


def adiabatic_t_r_gradAd(tr, mr, pr, rhor, r, grad_ad, delr=10.0, tmin=300.0):
    """
    Given T(r), compute T(r + delr) using the following equation:
        T(r + delr) = T(r) - (T/P) * rho * (Gm/r**2) * grad_ad * delr
    which is a modified version of adiabatic_t_r() that does not require
    knowledge of alpha and cp - they are incorporated into the adiabatic
    graidient grad_ad, which is given by the EOS.

    Parameters
    ----------
    tr : float
        T(r), temperature at radius r.
    mr : float
        m(r).
    pr : float
        P(r).
    rhor : float
        rho(r).
    r : float
        radius r.
    grad_ad : float
        adiabatic gradient defined as
        grad_ad = (d lnT / d lnP)|S = (alpha T / cp rho) P/T
    delr : float, optional
        radius step in m. The default is 10.0.
    tmin : float, optional
        minimum temperature; if tr-a < tmin, return tmin instead.

    Returns
    -------
    None.

    """
    a = tr / pr
    a *= rhor
    a *= G * mr / r**2.0
    a *= grad_ad * delr
    return max(tr - a, tmin)
    

def adiabatic_t_r(ti, mr, pr, rhor, r, delr=10.0, alpha=210.0E-6, cp=4184.0,
                  tmin=300.0):
    """
    Given T(r), compute T(r + delr) using dT/dr = - T alpha g / c_p

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
    alpha : float
        coefficient of volumetric thermal expansion, which can be calculated 
        from EOS as alpha = 1/V (del V/del T) |_P.
    cp : float
        specific heat capacity at constant pressure.
    tmin : float, optional
        minimum temperature; if tf < tmin, return tmin instead.

    Returns
    -------
    tf, the temperature at radius r + delr.

    """
    g = G * mr / r ** 2.0
    delta_t = - (ti * alpha * g) / cp * delr
    tf = ti + delta_t
    return max(tf, tmin)


def adiabatic_t_r_boujibar20_eq9(tr, rhor, mr, r, delr, gamma0, gamma1, alpha0, rho0,
                                 K0, K0p, tmin=300.0):
    """
    Given T(r), compute T(r + delr) using Eq. 9 from Boujibar et al. (2020). This adiabatic temperature
    profile depends on several constant parameters (avaliable in Table 1 in Boujibar et al. 2020).
    """
    # define the parameters
    K0 *= 1.0e9 # convert GPa to Pa
    alpha0 *= 1.0e-6 # alpha0 is given in 1.0e-6 K-1 unit in Table 1
    
    x = rho0 / rhor
    theta = (3.0/2.0) * (K0p - 1)
    alphar = alpha0 * x**3.0 # thermal expansivity
    gammar = gamma0 * x**gamma1 # Gruneisen parameter
    KTr = K0 * x**(-2.0/3.0) * (1 + (1 + theta * x**(1.0/3.0)) * (1 - x**(1.0/3.0))) * np.exp(theta * (1 - x**(1.0/3.0))) # isothermal bulk modulus
    KSr = KTr * (1 + alphar * gammar * tr) # isentropic bulk modulus
    gr = G * mr / r**2.0 # gravitational acceleration
    
    # finally, calculate delta_t using dT/dr = - rhor * gr * gammar * tr / KSr
    delta_t = - ((rhor * gr * gammar * tr) / KSr) * delr
    tf = tr + delta_t
    return max(tf, tmin)


def two_stream_temp_rs10(teff, teq, tau, mu0=0.5, gamma=1.0):
    """ Two-stream approximation of the temperature profile of the gas envelope. Equation (4) in Rogers & Seager 2010.
    Teq = (L*(1-A) / 16 pi sigma a**2)**(1/4) is defined in Equation (17)
    Teff = (Lint / 4 pi sigma R**2)**(1/4) is the interior temperature of the planet, defined in Equation (18)
    
    mu0 = 1/2 assuming it is averaged over the day hemisphere (following Rogers & Seager 2010)
    gamma = 1 is a fiducial value following Rogers & Seager 2010; can also be 0.1-10 depending on the atmosphere
    """
    t4 = 3.0/4.0 * teff ** 4.0 * (tau + (2.0/3.0))
    t4 += mu0 * teq ** 4.0 * (1 + (3.0/2.0) * (mu0/gamma)**2.0) - \
        (3.0/2.0) * (mu0/gamma)**3.0 * np.log(1 + (gamma/mu0)) - \
            (3.0/4.0) * (mu0/gamma) * np.exp(-(gamma * tau) / mu0)
    return t4 ** (1.0/4.0)


def two_stream_temp_g10(teff, teq, tau, f=0.5, gamma=1.0):
    """ Use this one for H/He layer. two_stream_temp_rs10() referenced a 2008 paper before Guillot 2010.
    Two-stream approximation of the temperature profile of the gas envelope. Equation (29) in Guillot 2010.
    Teq and Teff are defined in the same way as Rogers & Seager 2010.
    Teff = Tint in Guillot 2010, and Teq = Tirr
    
    f = 1/2 assuming day-side average; f=1 means substellar point and f=1/4 means averaging over the whole planet surface.
    gamma = 1 is a fiducial value following Rogers & Seager 2010; can also be 0.1-10 depending on the atmosphere
    """
    t4 = 3.0/4.0 * teff ** 4.0 * (tau + (2.0/3.0))
    t4 += (3.0/4.0) * teq ** 4.0 * f * ((2.0/3.0) + 1/(gamma * np.sqrt(3)) + \
        ((gamma/np.sqrt(3)) - (1/(gamma*np.sqrt(3))) * np.exp(-gamma*tau*np.sqrt(3)))
        )
    return t4 ** (1.0/4.0)


def solve_tau_two_stream_temp_g10(t, teff, teq, f=0.5, gamma=1.0):
    """ The inverse of two_stream_temp_g10(), which solves for optical depth tau given the temperature T. """
    def func(tau):
        return t**4.0 - 3.0/4.0 * teff ** 4.0 * (tau + (2.0/3.0)) - \
            (3.0/4.0) * teq ** 4.0 * f * ((2.0/3.0) + 1/(gamma * np.sqrt(3)) + \
                ((gamma/np.sqrt(3)) - (1/(gamma*np.sqrt(3))) * np.exp(-gamma*tau*np.sqrt(3))))
    
    root = optimize.fsolve(func, [1.0e6]) # 1.0e6 is a guess for tau
    return float(root)


def freedman2008_mean_opacity(p, t):
    """ Return KappaR(P, T), where KappaR is the Rosseland mean opacity of H/He assuming solar metallicity.
    Note that in the file provided by Freedman 2008, kappaR is given in cm2/g, so we need to convert it 
    into SI unit m2/kg. The conversion factor is:
        cm2 g-1 = 1.0e-4 m2 1.0e-3 kg-1 = 0.1 m2 kg-1
        
    Note that the Freedman+2008 opacity table has limited P and T range. 75 K <= T <= 4000 K, 
    and 30 Pa < P < 1.0e13 Pa (was 300 bar, I modified it to include higher pressure and changed the
    table into a rectangular grid). When P, T exceeds these values, use the upper/lower bound values
    to replace them.
    """
    if p > 1.0e13:
        # print("P is out of range of Freedman+2008 opacity table, replaced with 1.0e13 Pa.")
        p = 1.0e13
    if p < 3.0e2:
        # print("P is out of range of Freedman+2008 opacity table, replaced with 300 Pa.")
        p = 3.0e2
    if t > 4000.0:
        # print("T is out of range of Freedman+2008 opacity table, replaced with 4000 K.")
        t = 4000.0
    if t < 75.0:
        # print("T is out of range of Freedman+2008 opacity table, replaced with 75 K.")
        t = 75.0
    return float(f08_interp_grid_kappaR(np.array([p, t]))) * 0.1


def adiabatic_profile_AVL(pr, tr, mr, rhor, r, mass_fractions, delr=10.0, tmin=300.0):
    """
    Given pressure [pr], temperature [tr], mass [mr], density [rhor], radius [r], [mass_fractions] of planet building 
    materials, use Gibbs free energy to calculate volumetric thermal expansion coefficient [alpha] and specific heat 
    capacity at constant pressure [cp], then use alpha and cp to calculate [tf], the adiabatic T(r+delr).
    """
    def calc_drho_dt(pr, tr, rhor, mass_fractions, delt=10.0):
        """ Given the pressure and temperature of the current layer, and mass fractions of each material, calculate
        drho/dt by keeping p constant and varying t by a small amount. Use AVL for mixture density. """
        rhof = avl_eos(pr, tr-delt, mass_fractions)
        return (rhof - rhor) / (-delt)
    
    def calc_ds_dt(pr, tr, mass_fractions, delt=10.0):
        """ Given the pressure and temperature of the current layer, and mass fractions of each material, calculate
        ds/dt by keeping p constant and varying t by a small amount. Use AVL for mixture entropy. """
        # calculate specific entropy at radius r
        sr_rock = 0.0 # rock EOS has no temperature dependence
        sr_water = aqua_interp_spline_s(pr, tr)
        _, _, sr_hhe = eos_cd21_hhe(pr, tr)
        sr_avl = mass_fractions[0] * sr_rock + mass_fractions[1] * sr_water + mass_fractions[2] * sr_hhe
        # calculate specific entropy at a slightly lower T
        sf_rock = 0.0 # rock EOS has no temperature dependence
        sf_water = aqua_interp_spline_s(pr, tr-delt)
        _, _, sf_hhe = eos_cd21_hhe(pr, tr-delt)
        sf_avl = mass_fractions[0] * sf_rock + mass_fractions[1] * sf_water + mass_fractions[2] * sf_hhe
        
        return (sf_avl - sr_avl) / (-delt)
    
    g = G * mr / r ** 2.0
    drho_dt = calc_drho_dt(pr, tr, rhor, mass_fractions)
    ds_dt = calc_ds_dt(pr, tr, mass_fractions)
    return max(tmin, float(tr + (g/rhor) * (drho_dt/ds_dt) * delr)) # adiabatic temperature of r+delr using AVL


def wagner2012_rocky_temp(mi, ri, pi, rhoi, ti, qi, delr, heat_production_rate, layer, cmb_radius, mantle_thickness, 
                          min_mix_len, mix_len_alpha=1.7426, mix_len_beta=1.0771, V_star_diff=6.0e-6, tmin=300.0):
    """ Caculate temperature profile of a rocky planet (iron core + silicate mantle), including a fully adiabatic
    core, a conducting boundary layer, and a mantle that gradually switches from the conduction-dominated boundary
    layer to convection-dominated upper layer. 
    
    Inputs: [m, r, p, rho] profiles calculated by simpleMR at layer with radius [r].
        Note that T calculated by simpleMR is not an input because we are calculating a different T(r) profile here.
        [q] is another input with the boundary conditions that q(0)=0 and q(Rp)=qs.
        [delr] is the radius step size.
        Subscript i denotes parameters at radius r, while subscript f denotes parameters at radius r+delr.
        [heat_production_rate] is the epsilon in the expression for dq/dr. Present-day Earth-like value is
        7.38e-11 W/kg. Unit of input heat_production_rate should be in W/kg as well.
    Outputs: [dT/dr, dq/dr, gamma, kc, kv, eta_eff, Nur, alpha, Ks] at this layer, as well as [T, q] at r + delr
    
    layer = "core" or "mantle". For the core, use parameters for Fe. For the mantle, use eta_diff and eta_disl
    to determine what material to use. If diffusion dominates (eta_diff < eta_disl), we are in the lower mantle,
    and parameters for MgSiO3 (pv or ppv) should be used. Otherwise, we are in the upper mantle, dislocation
    dominates, and we should use Mg2SiO4 (olivine).
    """
    
    def nusselt_number(kv, kc, q, dt_dr_s):
        """ Eq. (18) in Wagner+2012. """
        return (1 + kv/kc) * (1 - (kv/q) * dt_dr_s)**(-1.0)
    
    def alpha_t(t, alpha1, alpha2):
        """ Temperature dependent part of thermal expansivity alpha. """
        return alpha1 + alpha2*t
    
    def alpha_t_p(t, alpha1, alpha2, x, delt):
        """ Eq. (22) """
        return alpha_t(t, alpha1, alpha2) * x**delt
    
    def klat_tp(t, p, b, gamma, K0p, K0, alpha1, alpha2, x, delt, thermal_k0):
        """ Eq. (24)(25), under the assumption that kc is dominated by phonon contribution klat. """
        c1 = thermal_k0 * (298.0/t)**b 
        if t > 298.0:
            c21 = 4*gamma+(1.0/3.0)
            # c22 = quad(alpha_t_p, 298.0, t, args=(alpha1, alpha2, x, delt))[0]
            c22 = quad(alpha_t, 298.0, t, args=(alpha1, alpha2))[0] # in the integral, we should use the T-dependent part of alpha only
            c2 = np.exp(-c21*c22)
        else: # the integral is simply zero, and we have e**0
            c2 = 1.0
        c3 = 1 + K0p * p / K0
        return c1 * c2 * c3
    
    # def kv_func(l, alpha, rho, g, ks, gamma, eta_eff, dt_dr_s, dt_dr):
    #     """ Eq. (12), function for calculating kv. """
    #     if np.abs(dt_dr) > np.abs(dt_dr_s):
    #         c1 = l**4.0 * alpha**2.0 * rho * g * ks 
    #         c2 = 18.0 * gamma * eta_eff
    #         c3 = dt_dr_s - dt_dr
    #         return (c1 / c2) * c3
    #     else: # there is no convection locally, kv = 0
    #         return 0
        
    def kv_const(l, alpha, rho, g, Ks, gamma, eta_eff):
        """ The constant part of kv before dT/dr_s - dT/dr. """
        c1 = l**4.0 * alpha**2.0 * rho * g * Ks 
        c2 = 18.0 * gamma * eta_eff
        return c1/c2
        
    def gamma_lower(gamma0, gamma_inf, x, beta):
        """ Gruneisen parameter, lower mantle and core. """
        return gamma_inf + (gamma0 - gamma_inf) * x**(-beta)
    
    def gamma_upper(gamma0, x, lamb):
        """ Gruneisen parameter, upper mantle. """
        return gamma0 * x**(-lamb)
    
    def eta_diff(B, d, m, Estar, p, Vstar, t):
        eta_exp = np.exp((Estar + p*Vstar)/(R * t))
        return (d**m / (2 * B)) * eta_exp
    
    def eta_disl(B, sigma, n, Estar, p, Vstar, t):
        eta_exp = np.exp((Estar + p*Vstar)/(R * t))
        return (sigma**(1-n) / (2*B)) * eta_exp
    
    def dq_dr(epsilon, rho, q, r):
        """ Eq. (20) """
        return epsilon * rho - 2.0 * q / r 
    
    def bulk_modulus_KSr(alphar, gammar, tr, K0, K0p, rhor, rho0):
        """ Equation (5) and (6) from Boujibar+2020 
        [alphar] is alpha(T, p), thermal expansion coefficient. [gammar] is the Gruneisen parameter.
        [K0] and [K0p] are defined the same way as in Wagner+2012.
        [x] is x(r), defined as rho0/rho(r)
        """
        theta = (3.0/2.0) * (K0p - 1)
        x = rho0 / rhor # note that Boujibar+2020 define x as rho0/rho(r), while Wagner+2012 define x as rho(r)/rho0, not to be confused.
        KTr = K0 * x**(-2.0/3.0) * (1 + (1 + theta * x**(1.0/3.0)) * (1 - x**(1.0/3.0))) * np.exp(theta * (1 - x**(1.0/3.0))) # isothermal bulk modulus
        KSr = KTr * (1 + alphar * gammar * tr) # isentropic bulk modulus
        return KSr
    
    def mixing_length_l_wagner2019(h, D, a_mlt=1.7426, b_mlt=1.0771):
        """ Equation (11), Wagner et al. (2019). alpha and beta are using intermediate values in the mobile-lid regime
        from Table 3. """
        # print('322', h, D)
        if h <= D/2.0 * b_mlt:
            l = a_mlt * h / b_mlt
        else:
            l = a_mlt * ((D-h)/(2-b_mlt))
        return l
    
    # because dT/dr depends on kv, while kv also depends on dT/dr, we need to solve an implicit equation here
    def solve_dt_dr_implicit(q, kc, kv_const, dt_dr_s):
        """ [kv_const] is the constant part of kv before dt/dr_s - dt/dr. 
        [dt_dr_s] is the adiabatic temperature gradient. """
        def func(dt_dr):
            return dt_dr + (q/kc) * (1 + kv_const*(dt_dr_s-dt_dr)/kc)**(-1.0) * (1-(kv_const/q)*(dt_dr_s-dt_dr)*dt_dr_s)
        
        root = optimize.fsolve(func, [dt_dr_s]) # dT/dr is not too far from its adiabatic value should be a good guess
        # if np.abs(root) > 1.0: # > 1 K/m temperature gradient is too large; this line avoids unphysically large values
        #     root = dt_dr_s
        # try:
        #     root = optimize.brentq(func, 1.0e5*dt_dr_s, dt_dr_s) # find root in an interval using Brent’s method
        # except ValueError:
        #     root = dt_dr_s # when -1.0 and 0.0 do not bound a root, fall back to adiabatic gradient
        root = float(root)
        # print(type(root), root)
        
        # check that |dt_dr| > |dt_dr_s|, otherwise fall back to adiabatic temperature gradient
        if np.abs(root) > np.abs(dt_dr_s):
            dt_dr = root
            kv = kv_const * (dt_dr_s - dt_dr)
        else:
            dt_dr = dt_dr_s
            kv = 0
        # print('dT/dr', root, dt_dr_s, q, kc, kv_const)
        return dt_dr, kv
    
    # initialize constants based on material; units in SI unless otherwise specified.
    # parameters come from Table 2 in Wagner+2012
    rho0_fe = 8.2694e3
    k0_fe = 149.4 # GPa
    k0p_fe = 5.65 # dimensionless
    k_inf_p_fe = 2.943 # dimensionless
    gamma0_fe = 1.875 # dimensionless
    gamma_inf_fe = 1.305 # dimensionless
    beta_fe = 3.289 # dimensionless
    # lambda does not apply for the core; only applies to Mg2SiO4
    rho0_mgsio3 = 3.9776e3
    k0_mgsio3 = 204.0e9 # GPa converted to Pa
    k0p_mgsio3 = 4.20 # dimensionless
    k_inf_p_mgsio3 = 2.561 # dimensionless
    gamma0_mgsio3 = 1.553 # dimensionless
    gamma_inf_mgsio3 = 1.114 # dimensionless
    beta_mgsio3 = 4.731 # dimensionless
     # lambda does not apply for MgSiO3; only applies to Mg2SiO4
    rho0_mg2sio4 = 3.2137e3
    k0_mg2sio4 = 127.4e9 # GPa converted to Pa
    k0p_mg2sio4 = 4.20 # dimensionless
    k_inf_p_mg2sio4 = None # does not apply
    gamma0_mg2sio4 = 1.31 # dimensionless
    gamma_inf_mg2sio4 = None # does not apply
    beta_mg2sio4 = None # does not apply
    lamb_mg2sio4 = 3.2 # dimensionless
    
    # initialize creep parameters (Table 3)
    # diffusion (DF)
    B_df = 6.1e-19
    m_df = 2.5
    n_df = 1.0
    Estar_df = 300.0 * 1000.0 # kJ mol-1 converted to J mol-1
    # Vstar_df = 6.0 * 1.0e-6 # cm3 mol-1 converted to m3 mol-1
    # Vstar_df = 3.439 * ((pi/1.30e11) + 0.5736)**(-0.6170) * 1.0e-6 # for diffusion, try parametrization in Wagner+2011 Fig.3(b)
    Vstar_df = V_star_diff # try to make the activation volume a variable
    # dislocation (DL)
    B_dl = 2.4e-16
    m_dl = 0.0
    n_dl = 3.5
    Estar_dl = 540.0 * 1000.0 # kJ mol-1 converted to J mol-1
    # Vstar_dl = 20.0 * 1.0e-6 # cm3 mol-1 converted to m3 mol-1
    Vstar_dl = 5.0 * 1.0e-6
    
    # initialize thermal conductivity parameters (Table 4)
    thermal_k0_mgsio3 = 4.7 # W K-1 m-1, not to be confused with EOS parameter K0
    alpha1_mgsio3 = 2.6e-5 # K-1
    alpha2_mgsio3 = 1.0e-8 # K-2
    delta_t_mgsio3 = 6.5 # dimensionless
    
    thermal_k0_mg2sio4 = 5.2 # W K-1 m-1, not to be confused with EOS parameter K0
    alpha1_mg2sio4 = 2.69e-5 # K-1
    alpha2_mg2sio4 = 2.12e-8 # K-2
    delta_t_mg2sio4 = 8.4 # dimensionless
    
    # first, check if we are in the mantle or in the core; if in core, the T-P profile is simply adiabatic
    # recall that outputs are [dT/dr, dq/dr, gamma, kc, kv, eta_eff, Nur, alpha, Ks] at this layer, as well as 
    # [T, q] at r + delr (i.e. tf, qf)
    layer_name = ""
    if layer == "core":
        layer_name = "core"
        tf = adiabatic_t_r_boujibar20_eq9(ti, rhoi, mi, ri, delr, 1.6, 0.92, 40.0, 7700.0, 125.0, 5.5) # liquid Fe parameters from Boujibar+2020
        dt_dr_layer = (tf - ti) / delr # dT/dr at this layer i
        # dq_dr_layer = dq_dr(heat_production_rate, rhoi, qi, ri)
        # qf = qi + dq_dr_layer
        gruneisen_param = gamma_lower(gamma0_fe, gamma_inf_fe, rhoi/rho0_fe, beta_fe)
        # assume no additional heat source in the core
        dq_dr_layer = 0.0
        k_core = 35.0 # W K-1 m-1
        qf = -k_core * dt_dr_layer # heat flux from the core
        # the other parameters are related to thermal conduction so do not apply for the fully adiabatic core
        Ks_layer = np.nan # bulk modulus
        kc_layer = np.nan # only applies for mantle; the core doesn't have alpha1 and alpha2 parameters (Table 4); use np.nan as a placeholder
        kv_layer = np.nan
        eta_diff_layer = np.nan
        eta_disl_layer = np.nan
        eta_eff_layer = np.nan
        Nur_layer = np.nan
        alpha_layer = np.nan
        mixing_length_l = np.nan
    elif layer == "mantle":
        # if qi is 0 (we just transitioned from core to mantle across the CMB), initiate qi with q_cmb
        # if qi == 0.0:
        #     k_core = 35.0 # W K-1 m-1
        #     tf_core_adiabat = adiabatic_t_r_boujibar20_eq9(ti, rhoi, mi, ri, delr, 1.6, 0.92, 40.0, 7700.0, 125.0, 5.5)
        #     dt_dr_core_adiabat = (tf_core_adiabat - ti) / delr # evaluate dT/dr along the core adiabat across the CMB
        #     qi = -k_core * dt_dr_core_adiabat # heat flux from the core
        #     print('431 qi', qi)
        mixing_length_l = max(mixing_length_l_wagner2019(ri-cmb_radius, mantle_thickness, mix_len_alpha, mix_len_beta), 
                              min_mix_len) # need to tune minimum mixing length to avoid unrealistically large dT/dr
        # mixing_length_l = 500.0e3
        # first calculate the effective viscosity eta, compare eta_diff and eta_disl, to determine if we are in the
        # upper or lower mantle
        eta_diff_layer = eta_diff(B_df, 2.0e-3, # grain size is 2 mm in the lower mantle
                                  m_df, Estar_df, pi, Vstar_df, ti)
        eta_disl_layer = eta_disl(B_dl, 1.0e6, # constant stress sigma is taken as 1 MPa
                                  n_dl, Estar_dl, pi, Vstar_dl, ti)
        # eta_eff_layer = min(eta_diff_layer, eta_disl_layer)
        eta_eff_layer = eta_diff_layer # try using diffusion only
        if eta_diff_layer < eta_disl_layer: # diffusion dominates, we are in the lower mantle with MgSiO3 (pv, ppv)
            layer_name = "lowerMantle"
            gruneisen_param = gamma_lower(gamma0_mgsio3, gamma_inf_mgsio3, rhoi/rho0_mgsio3, beta_mgsio3)
            tf_adiabatic = adiabatic_t_r_boujibar20_eq9(ti, rhoi, mi, ri, delr, 1.48, 1.4, 20.0, 4260.0, 324.0, 3.3, 
                                                        tmin=300.0) # ppv parameters from Boujibar+2020
            dt_dr_adiabatic = (tf_adiabatic - ti) / delr
            alpha_layer = alpha_t_p(ti, alpha1_mgsio3, alpha2_mgsio3, rhoi/rho0_mgsio3, delta_t_mgsio3)
            Ks_layer = bulk_modulus_KSr(alpha_layer, gruneisen_param, ti, k0_mgsio3, k0p_mgsio3, rhoi, rho0_mgsio3)
            kc_layer = klat_tp(ti, pi, 0.33, gruneisen_param, k0p_mgsio3, k0_mgsio3, alpha1_mgsio3, alpha2_mgsio3, 
                               rhoi/rho0_mgsio3, delta_t_mgsio3, thermal_k0_mgsio3)
            kv_const_layer = kv_const(mixing_length_l, alpha_layer, rhoi, G*mi/ri**2, Ks_layer, gruneisen_param, 
                                      eta_eff_layer)
            dt_dr_layer, kv_layer = solve_dt_dr_implicit(qi, kc_layer, kv_const_layer, dt_dr_adiabatic)
            dq_dr_layer = dq_dr(heat_production_rate, rhoi, qi, ri)
            Nur_layer = nusselt_number(kv_layer, kc_layer, qi, dt_dr_adiabatic)
            tf = ti + dt_dr_layer * delr
            qf = qi + dq_dr_layer * delr
        else: # dislocation dominates, we are in the upper mantle with Mg2SiO4 (enstatite)
            layer_name = "upperMantle"
            gruneisen_param = gamma_upper(gamma0_mg2sio4, rhoi/rho0_mg2sio4, lamb_mg2sio4)
            # tf_adiabatic = adiabatic_t_r_boujibar20_eq9(ti, rhoi, mi, ri, delr, 0.99, 2.1, 20.0, 3226.0, 128.0, 4.2, 
                                                        # tmin=300.0) # peridotite (enstatite) parameters from Boujibar+2020
            tf_adiabatic = adiabatic_t_r_boujibar20_eq9(ti, rhoi, mi, ri, delr, 1.48, 1.4, 20.0, 4260.0, 324.0, 3.3, 
                                                        tmin=300.0) # use ppv parameters too; enstatite parameters make dT/dr too small 
            dt_dr_adiabatic = (tf_adiabatic - ti) / delr
            alpha_layer = alpha_t_p(ti, alpha1_mg2sio4, alpha2_mg2sio4, rhoi/rho0_mg2sio4, delta_t_mg2sio4)
            Ks_layer = bulk_modulus_KSr(alpha_layer, gruneisen_param, ti, k0_mg2sio4, k0p_mg2sio4, rhoi, rho0_mg2sio4)
            kc_layer = klat_tp(ti, pi, 0.33, gruneisen_param, k0p_mg2sio4, k0_mg2sio4, alpha1_mg2sio4, alpha2_mg2sio4, 
                               rhoi/rho0_mg2sio4, delta_t_mg2sio4, thermal_k0_mg2sio4)
            kv_const_layer = kv_const(mixing_length_l, alpha_layer, rhoi, G*mi/ri**2, Ks_layer, gruneisen_param, 
                                      eta_eff_layer)
            dt_dr_layer, kv_layer = solve_dt_dr_implicit(qi, kc_layer, kv_const_layer, dt_dr_adiabatic)
            dq_dr_layer = dq_dr(heat_production_rate, rhoi, qi, ri)
            Nur_layer = nusselt_number(kv_layer, kc_layer, qi, dt_dr_adiabatic)
            tf = ti + dt_dr_layer * delr 
            qf = qi + dq_dr_layer * delr
            # print('476 dT/dr', dt_dr_adiabatic, dt_dr_layer)
        # print(layer, eta_diff_layer, eta_disl_layer, eta_eff_layer, dt_dr_layer, dq_dr_layer, gruneisen_param, kc_layer, 
        #       kv_layer, eta_eff_layer, Nur_layer, alpha_layer, tf, qf, mixing_length_l, Ks_layer)
    elif layer == "neither": # we are in water or H/He layer, Wagner+2012 equations don't apply, return NaNs instead
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ''
    else:
        raise ValueError("Layer not valid! Valid inputs are 'core', 'mantle', and 'neither' for water or H/He layer.")

    return dt_dr_layer, dq_dr_layer, gruneisen_param, kc_layer, kv_layer, \
        [eta_diff_layer, eta_disl_layer, eta_eff_layer], Nur_layer, alpha_layer, \
        Ks_layer, max(tf, tmin), qf, mixing_length_l, layer_name


def write_wagner2012_profile(mlist, rlist, plist, rholist, mboundary, rboundary, pboundary, tlist, tboundary, delr,
                             layer_name_list, min_mix_len = 100.0e3, heat_production_rate=7.38e-11, qc=0.0, 
                             tc=5000.0, mix_len_alpha=1.7426, mix_len_beta=1.0771, V_star_diff=6.0e-6,
                             outdir='', pname='Planet'):
    """ Given m(r), r, P(r), rho(r), T(r) profiles of a planet model, calculate dt_dr_layer, dq_dr_layer, 
    gruneisen_param, kc_layer, kv_layer, eta_eff_layer, Nur_layer, alpha_layer, tf, qf using Wagner+2012 equations,
    and write all the quantities to a text file. """
    # calculate thermal parameters using Wagner+2012 equations for each layer
    t_new_list = [tc] # new T(r) profile including both conduction and convection
    q_list = [qc]
    dt_dr_list = []
    dq_dr_list = []
    gruneisen_param_list = []
    kc_list = []
    kv_list = []
    eta_diff_list = []
    eta_disl_list = []
    eta_eff_list = []
    Nur_list = []
    alpha_list = []
    Ks_list = [] # bulk modulus
    mix_len_list = [] # mixing length
    layer_name_list_new = [] # updated layer name, which can tell between upper and lower mantle
    
    # m_core = mboundary[0] # assuming that the first layer is iron core and the second layer is silicate mantle
    # m_core_and_mantle = mboundary[1]
    cmb_radius = rboundary[0]
    mantle_thickness = rboundary[1] - rboundary[0]
    for i in range(len(mlist)):
        if layer_name_list[i] == 'core': # we are in the iron core
            which_layer = 'core'
        elif layer_name_list[i] == 'mantle': # we are in the silicate mantle
            which_layer = 'mantle'
        else: # we are in water or H/He layer, where the Wagner+2012 temperature profile no longer apply
            which_layer = 'neither'
        dt_dr_layer, dq_dr_layer, gruneisen_param, kc_layer, kv_layer, eta_list, Nur_layer, alpha_layer, \
            Ks_layer, tf, qf, mix_len, layer_name = wagner2012_rocky_temp(mlist[i], rlist[i], plist[i], rholist[i], 
                                                                          t_new_list[i], q_list[i], delr, 
                                                                          heat_production_rate, which_layer, cmb_radius, 
                                                                          mantle_thickness, min_mix_len, mix_len_alpha,
                                                                          mix_len_beta, V_star_diff)
        dt_dr_list.append(dt_dr_layer)
        dq_dr_list.append(dq_dr_layer)
        gruneisen_param_list.append(gruneisen_param)
        kc_list.append(kc_layer)
        kv_list.append(kv_layer)
        eta_diff_list.append(eta_list[0])
        eta_disl_list.append(eta_list[1])
        eta_eff_list.append(eta_list[2])
        Nur_list.append(Nur_layer)
        alpha_list.append(alpha_layer)
        Ks_list.append(Ks_layer)
        t_new_list.append(tf)
        q_list.append(qf)
        mix_len_list.append(mix_len)
        layer_name_list_new.append(layer_name)
    # after looping through the entire list, t_new_list and q_list will have two extra elements at the end, remove them
    t_new_list = t_new_list[:-1]
    q_list = q_list[:-1]
    assert len(t_new_list) == len(mlist) and len(q_list) == len(mlist)

    # write everything to text file
    with open(outdir + pname + '.txt', 'a') as outfile:
        outfile.write('# ' + pname)
        outfile.write('\n')
        outfile.write('# Calculated total mass (Me): ')
        outfile.write('{:.10f}'.format(mlist[-1]/ Me))
        outfile.write('\n')
        outfile.write('# Calculated total radius (Re): ')
        outfile.write('{:.10f}'.format(rlist[-1]/ Re))
        outfile.write('\n')
        outfile.write('# Calculated surface pressure (bar): ')
        outfile.write('{:.10f}'.format(plist[-1]/ 1.0e5))
        outfile.write('\n')
        outfile.write('# Calculated surface density (kg m-3): ')
        outfile.write('{:.10f}'.format(rholist[-1]))
        outfile.write('\n')
        outfile.write('# Mass at layer boundaries (Me): ')
        mb = list(map(lambda m: '{:.10f}'.format(m/Me), mboundary))
        outfile.write(' '.join(mb))
        outfile.write('\n')
        outfile.write('# Radius at layer boundaries (Re): ')
        rb = list(map(lambda r: '{:.10f}'.format(r/Re), rboundary))
        outfile.write(' '.join(rb))
        outfile.write('\n')
        outfile.write('# Pressure at layer boundaries (GPa): ')
        pb = list(map(lambda p: '{:.10f}'.format(p/1.0e9), pboundary))
        outfile.write(' '.join(pb))
        outfile.write('\n')
        outfile.write('# Temperature at layer boundaries (K): ')
        if tboundary == []:
            tb = ['300.0'] * len(mboundary)
        else:
            tb = list(map(lambda t: '{:.10f}'.format(t), tboundary))
        outfile.write(' '.join(tb))
        outfile.write('\n')
        outfile.write('# M (kg)\tR (m)\tP (Pa)\trho (kg m-3)\tT (K)\tT_new (K)\tq (W m-2)\tdT/dr\tdq/dr\tgamma\tk_c\tk_v\teta_diff\teta_disl\teta_eff\tNur\talpha\tKs\tmix_length (m)\tlayer')
        outfile.write('\n')
        for i in range(len(mlist)):
            outfile.write('{:.10e}'.format(mlist[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(rlist[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(plist[i]))
            outfile.write('\t')
            outfile.write('{:.10f}'.format(rholist[i]))
            outfile.write('\t')
            outfile.write('{:.10f}'.format(tlist[i]))
            outfile.write('\t')
            outfile.write('{:.10f}'.format(t_new_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(q_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(dt_dr_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(dq_dr_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(gruneisen_param_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(kc_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(kv_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(eta_diff_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(eta_disl_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(eta_eff_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(Nur_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(alpha_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(Ks_list[i]))
            outfile.write('\t')
            outfile.write('{:.10e}'.format(mix_len_list[i]))
            outfile.write('\t')
            outfile.write(layer_name_list_new[i])
            outfile.write('\n')
    

def plot_t_profile():
    # use K2-18 b as an example, case (2) from Madhu et al. 2020
    # 45% Earth-like core (13.5% Fe, 31.5% silicates), 54.97% H2O, 0.03% H/He
    p1 = 2000.0e9; p2 = 4000.0e9  # edicated guess basd on planet mass
    mt = 8.63 * Me * 0.9997
    err_fraction = 1.0e-4
    layers = ['tabulated fe', 'tabulated mgsio3', 'tabulated h2o']
    layer_mass_fractions = [135041, 315095, 549864]
    plot_profiles = True
    save_figures = True
    pname = 'K2-18b_Madhu20-2'
    pc_solution = 3000.0e9
    mass, m, r, mlist, rlist, plist, rholist, mboundary, rboundary, pboundary =\
        run_single_planet_multilayer(mt, layers, 
                                     layer_mass_fractions, pc=pc_solution, 
                                     plot_profiles=plot_profiles,
                                     save_figures=save_figures,
                                     pname=pname)
    
    alpha_fe = 35.4E-6
    alpha_mgsio3 = 24.4E-6
    alpha_h2o = 210.0E-6
    cp_fe = 460.0
    cp_mgsio3 = 845.0  # using MgSiO4 from http://www.minsocam.org/ammin/AM67/AM67_470.pdf as a proxy
    cp_h2o = 4184.0
    ti = 4000.0  # central temperature guess in K
    tlist = []
    for i in range(len(mlist)):
        mr = mlist[i]
        if mr < mboundary[0]:
            alpha = alpha_fe
            cp = cp_fe
        elif mr >= mboundary[0] and mr < mboundary[1]:
            alpha = alpha_mgsio3
            cp = cp_mgsio3
        else:
            alpha = alpha_h2o
            cp = cp_h2o
        tf = adiabatic_t_r(ti, mr, plist[i], rholist[i], rlist[i],
                           alpha=alpha, cp=cp)
        tlist.append(tf)
        ti = tf
    
    rlist = list(map(lambda x: x / 1000.0, rlist))  # convert m to km
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(rlist, tlist, 'k-')
    plt.xlabel('Radius (km)', fontsize=16, **hfont)
    plt.ylabel(r'Adiabatic Temperature (K)', fontsize=16, **hfont)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0, max(rlist)])
    plt.savefig("t_vs_r_" + pname + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    
    print('T surf=', tlist[-1])
    

def main():
    plot_t_profile()
    

if __name__ == '__main__':
    main()
    
    
    

