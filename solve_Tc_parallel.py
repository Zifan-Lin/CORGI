#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:50:00 2024

@author: linzifan

Call solmr.solve_tc_bisect() to solve for central temperature of models in parallel.
"""

import numpy as np
import simpleMR as smr
import solveMR as solmr
import datetime
from multiprocessing import Pool

# define some useful constants in SI
G = 6.67408e-11 # gravitational constant
Re = 6371000.0 # Earth radius
Me = 5.97e24 # Earth mass
Msun = 1.989e30 # Solar mass
AU = 149597870700.0 # AU in meters
kb = 1.380649e-23 # Boltzmann constant, J/K
sigma = 5.670374419e-8 # Stefan-Boltzmann constant
massH = 1.66053906660e-27 # mass of H atom in kg


def calc_tau_period(mc, delr, rc, tau_photo=1000.0):
    """ Given core mass mc, envelop radius delr, core radius rc, and photoevaporation timescale tau_photo,
    calculate the orbital period P required to evaporize the planet's atmosphere within tau_photo, which
    is by default 1 Gyr. Note that tau_photo is in unit of Myr. mc, delr, and rc in Earth units. """
    if delr/rc < 1:
        r_rc_term = (delr/rc)**1.57
    else:
        r_rc_term = (delr/rc)**(1.69)
        
    factor = 210.0 * (1/1.2)**(-3.0) * (tau_photo/100.0)**0.37 * (mc/5.0)**1.42 * r_rc_term
    P = (tau_photo / factor) ** (1/1.41) # orbital period in the unit of 10 days
    
    return float(P*10) # return period in days


def kepler_3rd_law_P_to_a(P, m, M=1.0):
    """ Return a in unit of meters given P in unit of day. M and m are in solar and Earth units. """
    P *= 24 * 3600.0 # convert P from day to s
    factor = (G * (M*Msun + m*Me)) / (4 * np.pi**2.0)
    a3 = factor * P**2.0
    a = a3 ** (1/3.0)
    return a # return a in unit of meters
    

def period_to_teq(P, m, M=1.0, Lstar=3.828e26):
    """ P is orbital period in days. Lstar is stellar bolometric luminosity in W, and by default takes the 
    solar value. Assuming zero albedo. Albedo of magma ocean is low (~0.1, refs?), while H/He and water
    atmospheres may have high-albedo clouds. This will introduce a factor of 10x error, which should be small
    compared to other approximations. """
    a = kepler_3rd_law_P_to_a(P, m, M)
    teq = (Lstar / (16.0 * sigma * np.pi * a**2.0)) ** (1/4.0)
    return float(teq)


# melt boundary from Fratanduono+2018, with the assumption that for low pressure < 20.6 GPa, the melt curve
# becomes isothermal at 2316.0 K
def mgsio3_melt_temp_F18(p):
    # note that p is in unit of GPa
    if p > 20.6:
        f18_melt_t = 2316.0 * (p - 20.6) ** 0.1769
    else:
        f18_melt_t = 2316.0 
    return f18_melt_t


def solve_tc_case(input_fn, output_fn, index, envelope, tc1, tc2):
    """" Solve for the central tempearture Tc given input filename (which contains information about e.g., Pc,
    Peq). When Tc is solved, write all input information as well as Teq, Tc, and calculated Tsurf to a text file.
    """
    if envelope == "H2O":
        lmf = [330000, 670000-1, 1, 0]
        psurf = 1.0e5 # 1 bar
        bc_ignore = "mass_h2o"
        delr = 100.0
    elif envelope == "H/He":
        lmf = [330000, 670000-1, 0, 1]
        psurf = 1.0e6 # 10 bar
        bc_ignore = "mass"
        delr = 1000.0
    else:
        raise ValueError("envelope can only be H2O or H/He!")
    
    # read info from input_fn
    input_data = np.genfromtxt(input_fn, skip_header=1)
    input_data = np.transpose(input_data)
    assert index >=0 and index < len(input_data[0]), "Provided index is out of bound!" # make sure index is not out of bound

    # calculate Teq
    pcore = input_data[0][index]
    mcore = input_data[1][index] # iron-silicate mass in Me
    mtot = input_data[6][index] # total mass of the planet
    rcore = input_data[2][index] # iron-silicate radius in Re
    renv = input_data[7][index] - rcore # total mass - iron-silicate mass
    period = calc_tau_period(mcore, renv, rcore)
    teq = period_to_teq(period, mtot)

    # solve for core temperature
    tcore_mid = solmr.solve_tc_bisect(tc1, tc2, mcore*Me, lmf, pc=pcore, teq=teq, ps=psurf, delr=delr, 
                                      bc_ignore=bc_ignore)

    # run forward model with calculated Tc, and find the liquid part of the mantle
    pmass, pmass_calculated, rs, ps, ts, rhos, mlist, rlist, plist, tlist, rholist, \
    taulist, mboundary, rboundary, pboundary, tboundary, layer_name_list =\
    smr.run_single_planet_4layers(mcore*Me, lmf, 
                                  pc=pcore, ps=psurf, tc=tcore_mid,
                                  rc=10.0, delr=delr,
                                  bc_ignore=bc_ignore,
                                  tmode_core='adiabatic_Boujibar_2020',
                                  maxsteps=2000000,
                                  fully_adiabatic_hhe=True)

    # find the depths of MO by finding intersection between P-T in the mantle and the F18 MgSiO3 melt curve
    # boundary[0] is core-mantle boundary (CMB), [1] is mantle-envelope boundary (MEB), and [2] is surface of 
    # the planet
    #
    # Note that we are interested in surface MO, i.e., if there's an edge case that the lower mantle is molten
    # while the upper mantle is not, we consider that the lower mantle can still stay compressed, because the
    # pressure exerted by the upper mantle remains. However, if the upper mantle is molten, due to subsequent
    # lost of H/He or H2O envelope, the upper mantle can decompress and alter the overall density of the planet.
    # Therefore, we find the melt boundary in such a way:
    # - if both CMB and MEB are molten, then the entire mantle is considered to be molten
    # - if both CMB and MEB are solid, then the entire mantle is considered to be solid
    # - if the CMB is molten but the MEB is not, consider the entire mantle as solid
    # - if the CMB is solid but the MEB is molten, iteratively find the inner melt boundary, and calculate the mass
    #   of the liquid part of the mantle / mass of the entire mantle
    mantle_melt_mass_fraction = -1
    cmb_is_liquid = tboundary[0] > mgsio3_melt_temp_F18(pboundary[0]/1.0e9)
    meb_is_liquid = tboundary[1] > mgsio3_melt_temp_F18(pboundary[1]/1.0e9)
    if cmb_is_liquid and meb_is_liquid:
        mantle_melt_mass_fraction = 1.0
    elif not cmb_is_liquid and not meb_is_liquid:
        mantle_melt_mass_fraction = 0.0
    elif cmb_is_liquid and not meb_is_liquid:
        mantle_melt_mass_fraction = 0.0
    elif not cmb_is_liquid and meb_is_liquid:
        inner_melt_index = -1
        # search for inner melt boundary using for loop
        for i in range(len(plist)):
            if plist[i] < pboundary[0] and plist[i] > pboundary[1]: # pressure is between CMB and MEB
               if tlist[i] > mgsio3_melt_temp_F18(plist[i]/1.0e9): # mantle is liquid at this layer
                   inner_melt_index = i
                   break # stop at first encounter of melt boundary
        # calculate the mass fraction of the molten part of the mantle
        mantle_mass = mboundary[1] - mboundary[0]
        liquid_mantle_mass = mboundary[1] - mlist[inner_melt_index]
        assert liquid_mantle_mass < mantle_mass
        mantle_melt_mass_fraction = liquid_mantle_mass / mantle_mass

    # save all above results to text file
    # header should be:
    # P_core (Pa)   M_core (Me) R_core (Re)	P_eq (Pa)	CMF	CRF	Mtot (Me)	Rtot (Re)	P_surf (Pa)	T_eq (K)    T_surf (K)	T_core (K)  rho_surf (kg m-3)   mantle_mmf
    with open(output_fn, "a") as outfile:
        outfile.write("{:.6e}".format(pcore)); outfile.write("\t")
        outfile.write("{:.6f}".format(mcore)); outfile.write("\t")
        outfile.write("{:.6f}".format(rcore)); outfile.write("\t")
        outfile.write("{:.6e}".format(input_data[3][index])); outfile.write("\t") # Peq
        outfile.write("{:.6f}".format(input_data[4][index])); outfile.write("\t") # CMF
        outfile.write("{:.6f}".format(input_data[5][index])); outfile.write("\t") # CRF
        outfile.write("{:.6f}".format(pmass_calculated/Me)); outfile.write("\t") # total mass in Me (updated)
        outfile.write("{:.6f}".format(rs/Re)); outfile.write("\t") # total radius in Re (updated)
        outfile.write("{:.6e}".format(ps)); outfile.write("\t") # surface pressure in Pa (updated)
        outfile.write("{:.6f}".format(teq)); outfile.write("\t") # calculated Teq
        outfile.write("{:.6f}".format(ts)); outfile.write("\t") # calculated Tsurf
        outfile.write("{:.6f}".format(tcore_mid)); outfile.write("\t") # calculated Tcore
        outfile.write("{:.6f}".format(rhos)); outfile.write("\t") # calculated rho at surface (kg m-3)
        outfile.write("{:.6f}".format(mantle_melt_mass_fraction)) # calculated mantle melt mass fraction
        outfile.write("\n")


def solve_tc_parallel_HHe():
    # define forward parameters
    hhe_dir = "MR_curves/high_density_envelope/v2_05-20Me/HHe/"
    hhe_fns = ['MR_curve_nakedCore_HHe_1000GPa.txt', 'MR_curve_nakedCore_HHe_2000GPa.txt', 'MR_curve_nakedCore_HHe_30GPa.txt', 
               'MR_curve_nakedCore_HHe_50GPa.txt', 'MR_curve_nakedCore_HHe_100GPa.txt', 'MR_curve_nakedCore_HHe_3000GPa.txt', 
               'MR_curve_nakedCore_HHe_5000GPa.txt', 'MR_curve_nakedCore_HHe_750GPa.txt', 'MR_curve_nakedCore_HHe_10GPa.txt', 
               'MR_curve_nakedCore_HHe_300GPa.txt', 'MR_curve_nakedCore_HHe_500GPa.txt', 'MR_curve_nakedCore_HHe_75GPa.txt']
    hhe_fns = list(map(lambda fn: hhe_dir + fn, hhe_fns))
    indexes = list(np.arange(0, 100))
    out_fn = "high_density_Tc_parallel_HHe.txt"

    input_fn_list = hhe_fns * len(indexes)
    input_fn_list.sort()
    num_models = len(input_fn_list)

    output_fn_list = [out_fn] * num_models

    index_list = indexes * len(hhe_fns)
    assert len(index_list) == num_models

    envelope_list = ["H/He"] * num_models
    tc1_list = [200.0] * num_models # to avoid any edge case that produces t < 300 K
    tc2_list = [100000.0] * num_models

    args = list(zip(input_fn_list, output_fn_list, index_list, envelope_list, tc1_list, tc2_list))

    # begin parallel runs for H/He first
    with open(out_fn, "a") as outfile:
        outfile.write("#P_core (Pa)\tM_core (Me)\tR_core (Re)\tP_eq (Pa)\tCMF\tCRF\tMtot (Me)\tRtot (Re)\tP_surf (Pa)\tT_eq (K)\tT_surf (K)\tT_core (K)\trho_surf (kg m-3)\tmantle_mmf\n")
    
    start_time = datetime.datetime.now()
    print('Beginning parallel run for H/He envelope...')

    with Pool() as pool:
        pool.starmap(solve_tc_case, args)
        # def solve_tc_case(input_fn, output_fn, index, envelope, tc1, tc2):

    print()
    print('************************ Parallel run ended ************************')
    print()
    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def solve_tc_parallel_H2O():
    h2o_dir = "MR_curves/high_density_envelope/v2_05-20Me/H2O/"
    h2o_fns = ['MR_curve_nakedCore_H2O_1000GPa.txt', 'MR_curve_nakedCore_H2O_2000GPa.txt', 'MR_curve_nakedCore_H2O_30GPa.txt', 
               'MR_curve_nakedCore_H2O_50GPa.txt', 'MR_curve_nakedCore_H2O_100GPa.txt', 'MR_curve_nakedCore_H2O_3000GPa.txt', 
               'MR_curve_nakedCore_H2O_5000GPa.txt', 'MR_curve_nakedCore_H2O_750GPa.txt', 'MR_curve_nakedCore_H2O_10GPa.txt', 
               'MR_curve_nakedCore_H2O_300GPa.txt', 'MR_curve_nakedCore_H2O_500GPa.txt', 'MR_curve_nakedCore_H2O_75GPa.txt']
    h2o_fns = list(map(lambda fn: h2o_dir + fn, h2o_fns))
    indexes = list(np.arange(0, 100))

    out_fn = "high_density_Tc_parallel_H2O.txt"

    input_fn_list = h2o_fns * len(indexes)
    input_fn_list.sort()
    num_models = len(input_fn_list)

    output_fn_list = [out_fn] * num_models

    index_list = indexes * len(h2o_fns)
    assert len(index_list) == num_models

    envelope_list = ["H2O"] * num_models
    tc1_list = [200.0] * num_models # to avoid any edge case that produces t < 300 K
    tc2_list = [50000.0] * num_models

    args = list(zip(input_fn_list, output_fn_list, index_list, envelope_list, tc1_list, tc2_list))

    # begin parallel runs for H/He first
    with open(out_fn, "a") as outfile:
        outfile.write("#P_core (Pa)\tM_core (Me)\tR_core (Re)\tP_eq (Pa)\tCMF\tCRF\tMtot (Me)\tRtot (Re)\tP_surf (Pa)\tT_eq (K)\tT_surf (K)\tT_core (K)\trho_surf (kg m-3)\tmantle_mmf\n")
    
    start_time = datetime.datetime.now()
    print('Beginning parallel run for H2O envelope...')

    with Pool() as pool:
        pool.starmap(solve_tc_case, args)
        # def solve_tc_case(input_fn, output_fn, index, envelope, tc1, tc2):

    print()
    print('************************ Parallel run ended ************************')
    print()
    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def test_solve_tc_case():
    input_fn = "Data/MR_curves/high-density/HHe_envelope/MR_curve_nakedCore_HHe_100GPa.txt"
    solve_tc_case(input_fn, "MR_curve_nakedCore_HHe_100GPa_test.txt", 6, "H/He", 300.0, 100000.0)


def main():
    # test_solve_tc_case()
    # solve_tc_parallel_HHe()
    solve_tc_parallel_H2O()


if __name__ == "__main__":
    main()


    