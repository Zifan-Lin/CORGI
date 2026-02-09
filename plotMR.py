#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:36:20 2022

@author: linzifan

Plot M(r), P(r), T(r), rho(r), ... profiles of simulated planets.

This should be expanded to a I/O module in the future. Currently the function of saving run to txt is included here.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams["font.family"] = "Times New Roman"
rcParams["mathtext.fontset"] = "cm"
from scipy.interpolate import interp1d

#=========================================================================
#====== useful constants, in SI
#=========================================================================

G = 6.67408e-11  # gravitational constant
Re = 6371000.0  # Earth radius
Me = 5.97e24  # Earth mass

# coefficients for calculating temperature
alpha_fe = 35.4E-6
alpha_mgsio3 = 24.4E-6
alpha_h2o = 210.0E-6
cp_fe = 460.0
cp_mgsio3 = 845.0  # using MgSiO4 from http://www.minsocam.org/ammin/AM67/AM67_470.pdf as a proxy
cp_h2o = 4184.0

#=========================================================================
#====== parameters for plotting
#=========================================================================

colors = ['#e6194b', '#4363d8', '#3cb44b', '#ffe119', '#f58231', '#911eb4', 
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', 
    '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
    '#000000']

#=========================================================================
#====== plotting functions
#=========================================================================

def plot_single_planet_multilayer(mlist, rlist, plist, rholist, 
                                  save_figures=False,
                                  plot_prem=False,
                                  pname='single_multilayer_planet'):
    """
    Plot the M(r), P(r), and rho(r) profiles of a multilayer planet.

    Parameters
    ----------
    mlist : list of float
        Mass in kg.
    rlist : list of float
        Radius in m.
    plist : list of float
        Pressure in Pa.
    rholist : list of float
        Density in kg/m3.
    save_figures : bool, optional
        Save figure to PNG or not.
    plot_prem : bool, optional
        Overplot PERM (Preliminary Reference Earth Model) or not.
    pname : string, optional
        Name of the planet. Will be used as saved figure file names.

    Returns
    -------
    None. Generates three figures.

    """
    mlist = list(map(lambda x: x / Me, mlist))  # convert kg to Me
    rlist = list(map(lambda x: x / 1000.0, rlist))  # convert m to km
    plist = list(map(lambda x: x / 1.0e9, plist))  # convert P to GPa
    
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(rlist, mlist, 'k-')
    plt.xlabel('Radius (km)', fontsize=16)
    plt.ylabel(r'Mass (M$_\oplus$)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0, max(rlist)])
    if save_figures:
        plt.savefig("m_vs_r_" + pname + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(rlist, plist, 'k-')
    plt.xlabel('Radius (km)', fontsize=16)
    plt.ylabel('Pressure (GPa)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0, max(rlist)])
    if save_figures:
        plt.savefig("p_vs_r_" + pname + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(rlist, rholist, 'k-')
    plt.xlabel('Radius (km)', fontsize=16)
    plt.ylabel(r'Density (kg m$^{-3}$)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0, max(rlist)])
    # plot PREM density profile optionally
    if plot_prem:
        prem = np.genfromtxt('Data/PREM_1s.txt', comments='#', delimiter=',')
        prem = np.transpose(prem)
        prem_r = prem[0]
        prem_rho = prem[2]
        prem_rho = list(map(lambda x: x * 1000.0, prem_rho)) # convert g cm-3 into kg m-3
        plt.plot(prem_r, prem_rho, 'k--')
    if save_figures:
        plt.savefig("rho_vs_r_" + pname + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    

def overplot_multiple_planets(mll, rll, pll, tll, rholl, pnames, 
                              save_figures=False, plot_prem=False,
                              savename='multiple_planet_overplot'):
    """ Similar to plot_single_planet_multilayer(), but overplots the profiles
        of multiple planets.
    """
    assert len(mll) == len(rll) == len(pll) == len(tll) == len(rholl) == len(pnames)
    
    for i in range(len(mll)):
        mll[i] = list(map(lambda x: x / Me, mll[i]))  # convert kg to Me
        rll[i] = list(map(lambda x: x / 1000.0, rll[i]))  # convert m to km
        pll[i] = list(map(lambda x: x / 1.0e9, pll[i]))  # convert P to GPa
    
    rmax = max(list(map(lambda l: max(l), rll)))
    pmax = max(list(map(lambda l: max(l), pll)))
    
    # plot M(r)
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(mll)):
        plt.plot(rll[i], mll[i], color=colors[i], linestyle='-', label=pnames[i])
    plt.xlabel('Radius (km)', fontsize=16)
    plt.ylabel(r'Mass (M$_\oplus$)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0, rmax])
    plt.legend(fontsize=16)
    if save_figures:
        plt.savefig("m_vs_r_" + savename + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    
    # plot P(r)
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(mll)):
        plt.plot(rll[i], pll[i], color=colors[i], linestyle='-', label=pnames[i])
    plt.xlabel('Radius (km)', fontsize=16)
    plt.ylabel('Pressure (GPa)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0, rmax])
    plt.legend(fontsize=16)
    if save_figures:
        plt.savefig("p_vs_r_" + savename + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    
    # plot T(r)
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(mll)):
        plt.plot(rll[i], tll[i], color=colors[i], linestyle='-', label=pnames[i])
    plt.xlabel('Radius (km)', fontsize=16)
    plt.ylabel('Temperature (K)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0, rmax])
    plt.legend(fontsize=16)
    if save_figures:
        plt.savefig("t_vs_r_" + savename + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    
    # plot rho(r)
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(mll)):
        plt.plot(rll[i], rholl[i], color=colors[i], linestyle='-', label=pnames[i])
    plt.xlabel('Radius (km)', fontsize=16)
    plt.ylabel(r'Density (kg m$^{-3}$)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([0, rmax])
    # plot PREM density profile optionally
    if plot_prem:
        prem = np.genfromtxt('Data/PREM_1s.txt', comments='#', delimiter=',')
        prem = np.transpose(prem)
        prem_r = prem[0]
        prem_rho = prem[2]
        prem_rho = list(map(lambda x: x * 1000.0, prem_rho)) # convert g cm-3 into kg m-3
        plt.plot(prem_r, prem_rho, 'k--', label='PREM')
    plt.legend(fontsize=16)
    if save_figures:
        plt.savefig("rho_vs_r_" + savename + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    
    # plot T(P)
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(len(mll)):
        plt.plot(pll[i], tll[i], color=colors[i], linestyle='-', label=pnames[i])
    plt.xlabel('Pressure (GPa)', fontsize=16)
    plt.ylabel('Temperature (K)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xscale('log')
    plt.xlim([pmax, 1.0e-5])
    plt.legend(fontsize=16)
    if save_figures:
        plt.savefig("t_vs_p_" + savename + ".png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()
    

def write_planet_to_txt(mlist, rlist, plist, rholist,
                        mboundary, rboundary, pboundary, 
                        tlist=[], tboundary=[], layer_name_list=[], outdir='', pname='Planet'):
    """ Write planet data to a text file. 
    Desired format:
    # pname
    # Calculated total mass (Me):
    # Calculated total radius (Re):
    # Calculated surface pressure (bar):
    # Calculated surface density (kg m-3)
    # Mass at layer boundaries (Me): 
    # Radius at layer boundaries (Re): 
    # Pressure at layer boundaries (GPa):
    # Temperature at layer boundaries (K): 300 K if tboundary is empty
    # M (kg)    R (m)   P (Pa)  rho (kg m-3)    T (K) (T is isothermal 300 K if tlist is empty)
    # XX    XX  XX  XX  XX
    """    
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
        outfile.write('# M (kg)\tR (m)\tP (Pa)\trho (kg m-3)\tT (K)\tlayer')
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
            if tlist == []:
                outfile.write('300.0')
            else:
                outfile.write('{:.10f}'.format(tlist[i]))
            outfile.write('\t')
            if layer_name_list == []:
                outfile.write('N/A')
            else:
                outfile.write(layer_name_list[i])
            outfile.write('\n')


def write_planet_to_txt_AVL(mlist, rlist, plist, rholist,
                            mboundary, rboundary, pboundary, 
                            tlist=[], tboundary=[], layer_name_list=[], 
                            x_rock_function=None, x_water_function=None, x_hhe_function=None,
                            outdir='', pname='Planet'):
    """ The same as write_planet_to_txt(), but has three more columns recording X_rock, X_water, and X_hhe.
    """
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
        outfile.write('# M (kg)\tR (m)\tP (Pa)\trho (kg m-3)\tT (K)\tlayer\tX_rock\tX_water\tX_hhe')
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
            if tlist == []:
                outfile.write('300.0')
            else:
                outfile.write('{:.10f}'.format(tlist[i]))
            outfile.write('\t')
            if layer_name_list == []:
                outfile.write('N/A')
            else:
                outfile.write(layer_name_list[i])
            outfile.write('\t')
            outfile.write('{:.10f}'.format(x_rock_function(rlist[i])))
            outfile.write('\t')
            outfile.write('{:.10f}'.format(x_water_function(rlist[i])))
            outfile.write('\t')
            outfile.write('{:.10f}'.format(x_hhe_function(rlist[i])))
            outfile.write('\n')
            

def get_layer_properties(pfile, input, input_type):
    """ Given a text file recording planetary properties by layer (in the same format as written by
    write_planet_to_txt), an input (can be mass, radius, pressure, or density), and an input_type
    (string, can be "mass", "radius", "pressure", or "density"), print the M, R, P, rho, T of this
    layer using interpolation. Note that input should be in SI unit (the same unit as written
    in the files).
    """
    assert input_type in ["mass", "radius", "pressure", "density"]
    
    # read data from pfile
    data = np.genfromtxt(pfile)
    data = np.transpose(data)
    m = data[0]
    r = data[1]
    p = data[2]
    rho = data[3]
    t = data[4]
    
    if input_type == "mass":
        assert input >= min(m) and input <= max(m), "Input mass out of bound!"
        interp_rm = interp1d(m, r)
        interp_pm = interp1d(m, p)
        interp_rhom = interp1d(m, rho)
        interp_tm = interp1d(m, t)
        print("Layer properties at mass = " + str(input) + " kg are (R, depth, P, rho, T):")
        print('{:.10f}'.format(interp_rm(input)) + " m")
        print('{:.10f}'.format(max(r) - interp_rm(input)) + " m")
        print('{:.10f}'.format(interp_pm(input)) + " Pa")
        print('{:.10f}'.format(interp_rhom(input)) + " kg m-3")
        print('{:.10f}'.format(interp_tm(input)) + " K")
    elif input_type == "radius":
        assert input >= min(r) and input <= max(r), "Input radius out of bound!"
        interp_mr = interp1d(r, m)
        interp_pr = interp1d(r, p)
        interp_rhor = interp1d(r, rho)
        interp_tr = interp1d(r, t)
        print("Layer properties at radius = " + str(input) + " m are (M, depth, P, rho, T):")
        print('{:.10f}'.format(interp_mr(input)) + " kg")
        print('{:.10f}'.format(max(r) - input) + " m")
        print('{:.10f}'.format(interp_pr(input)) + " Pa")
        print('{:.10f}'.format(interp_rhor(input)) + " kg m-3")
        print('{:.10f}'.format(interp_tr(input)) + " K")
    elif input_type == "pressure":
        assert input >= min(p) and input <= max(p), "Input pressure out of bound!"
        interp_mp = interp1d(p, m)
        interp_rp = interp1d(p, r)
        interp_rhop = interp1d(p, rho)
        interp_tp = interp1d(p, t)
        print("Layer properties at pressure = " + str(input) + " Pa are (M, R, depth, rho, T):")
        print('{:.10f}'.format(interp_mp(input)) + " kg")
        print('{:.10f}'.format(interp_rp(input)) + " m")
        print('{:.10f}'.format(max(r) - interp_rp(input)) + " m")
        print('{:.10f}'.format(interp_rhop(input)) + " kg m-3")
        print('{:.10f}'.format(interp_tp(input)) + " K")
    elif input_type == "density":
        assert input >= min(rho) and input <= max(rho), "Input density out of bound!"
        interp_mrho = interp1d(rho, m)
        interp_rrho = interp1d(rho, r)
        interp_prho = interp1d(rho, p)
        interp_trho = interp1d(rho, t)
        print("Layer properties at density = " + str(input) + " kg m-3 are (M, R, depth, P, T):")
        print('{:.10f}'.format(interp_mrho(input)) + " kg")
        print('{:.10f}'.format(interp_rrho(input)) + " m")
        print('{:.10f}'.format(max(r) - interp_rrho(input)) + " m")
        print('{:.10f}'.format(interp_prho(input)) + " Pa")
        print('{:.10f}'.format(interp_trho(input)) + " K")