#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:29:00 2023

@author: linzifan

The goal of this script is to run forward MR models, more specifically, the smr.run_single_planet_4layers() function, 
in parallel to allow rapid exploration of a large compositional parameter space.
"""

# import packages
import numpy as np
import EOS as eos
import simpleMR as smr
import plotMR as pltmr
# import gravity_harmonics as rop
# import matplotlib.pyplot as plt
# from matplotlib import colors
# import matplotlib.cm as cm
# from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['mathtext.fontset'] = 'stix'
# from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator
# import seaborn as sns
import datetime
from multiprocessing import Pool

# define useful constants, in SI
G = 6.67408e-11  # gravitational constant
Re = 6371000.0  # Earth radius
Me = 5.97e24  # Earth mass


def run_forwardMR_case(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                       a20, a21):
    """ This is the function for running a single planet with smr.run_single_planet_4layers().
    Metadata of the planet that are relevant for convergence (total mass, total radius, surface pressure, 
    etc.) are written out to a file.
    """
    input_para = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                  a20, a21]
    assert len(input_para) == 22
    
    # run smr.run_single_planet_4layers() with input parameters
    _, pmass_calc_case, _, _, _, _, _, rlist_case, plist_case, tlist_case, _, _, \
    _, _, _, _, _ = \
    smr.run_single_planet_4layers(pmass=input_para[0], layer_mass_fractions=input_para[1], 
                                  pc=input_para[2], ps=input_para[3], tc=input_para[4],
                                  rc=input_para[5], delr=input_para[6], bc_ignore=input_para[7],
                                  pradius=input_para[8], mmw=input_para[9], teff=input_para[10], 
                                  teq=input_para[11], guillot_f=input_para[12], 
                                  guillot_gamma=input_para[13], tmode_core=input_para[14],
                                  maxsteps=input_para[15], eos_core=input_para[16],
                                  eos_mantle=input_para[17])
    model_index = input_para[18] # integer starting from 0 to distinguish between models
    pmass_measured = input_para[19]
    prad_measured = input_para[20]
    outname = input_para[21]
    
    # write the total mass, mass deviation (abs of calculated mass - pmass_measured), 
    # mass deviation fraction (in percent), total radius, radius deviation, radius deviation fraction,
    # surface pressure, surface temperature to a text file. In units that are intuitive (Me, Re,
    # GPa or bar, K, etc.)
    line = "" # the data line to write into text file
    line += str(model_index) + "\t"
    line += "{:.6f}".format(pmass_calc_case/smr.Me) + " Me\t" # planet mass in Me
    line += "{:.6f}".format(np.abs(pmass_calc_case/smr.Me - pmass_measured)) + "\t" # mass deviation
    line += "{:.6f}".format(np.abs(pmass_calc_case/smr.Me - pmass_measured)/pmass_measured * 100.0) + "\t" # mass deviation %
    line += "{:.6f}".format(rlist_case[-1]/smr.Re) + " Re\t" # planet radius in Re
    line += "{:.6f}".format(np.abs(rlist_case[-1]/smr.Re - prad_measured)) + "\t" # radius deviation
    line += "{:.6f}".format(np.abs(rlist_case[-1]/smr.Re - prad_measured)/prad_measured * 100.0) + "\t" # radius deviation %
    line += "{:.6f}".format(plist_case[-1]/1.0e5) + " bar\t"
    line += "{:.6f}".format(tlist_case[-1]) + " K\t"
    line += "\n"
    
    with open(outname, "a") as outfile:
        outfile.write(line)


def solve_forwardMR_pc_bisect_case(p1, p2, pmass, lmf, ps, tc, rc, delr, bc_ignore, pradius, mmw, teff, teq, 
                                   guillot_f, guillot_gamma, tmode_core, maxsteps, eos_core, eos_mantle,
                                   model_index, pmass_measured, prad_measured, outname, recursion_i, 
                                   recursion_max=10, ps_tol=1.0e7, mass_tol=1.0e-3):
    """ Given forward MR model input parameters and a range for central pressure Pc, using bisection, solve for 
    the central pressure Pc that
        (1) minimizes surface pressure (close to 1 bar), and 
        (2) minimize total mass deviation. 
    A range of Pc is given as p1 < p2 (a tested Pc range for Uranus ice giant models is 500 GPa to 2500 GPa, which
    should be the initial p1 and p2 inputs). An intermediate pressure pm = (p1 + p2) / 2 is calculated and passed
    into smr.run_single_planet_4layers(). If the surface pressure is greater than a certain threshold (say, 100 bar),
    pm is too large, keep p1 unchanged and pass pm to the recursive function as the new p2. If the surface pressure 
    is close to 1 bar, but the total mass is smaller than the given planet mass (say, off by 0.1%), pm is too small,
    keep p2 unchanged and pass pm to the recursive function as the new p1.

    Terminate the recursion and write smr.run_single_planet_4layers(pm) results to text file when
        - Surface pressure < 100 bar = 1.0e7 Pa AND
        - Total mass deviation fraction < 1.0e-3 OR
        - Maximum recursion depth reached (try 10 for testing, increase if needed)

    recursion_i starts from 0. When recursion_i == recursion_max, terminate recursion and write to text file.
    """
    assert p1 < p2

    pm = (p1 + p2) / 2.0

    _, pmass_calc_case, _, _, _, _, mlist_case, rlist_case, plist_case, tlist_case, rholist_case, _, \
    mb_case, rb_case, pb_case, tb_case, lnames_case = \
    smr.run_single_planet_4layers(pmass=pmass, layer_mass_fractions=lmf, pc=pm, ps=ps, tc=tc,
                                  rc=rc, delr=delr, bc_ignore=bc_ignore,
                                  pradius=pradius, mmw=mmw, teff=teff, 
                                  teq=teq, guillot_f=guillot_f, 
                                  guillot_gamma=guillot_gamma, tmode_core=tmode_core,
                                  maxsteps=maxsteps, eos_core=eos_core,
                                  eos_mantle=eos_mantle)
    recursion_i += 1
    
    # Check if surface pressure and total mass outer boundary conditions are met
    mass_dev_frac = np.abs(pmass_calc_case/smr.Me - pmass_measured)/pmass_measured
    # If outer boundary conditions are met, write output to text file, flag it as "converged", and save forward
    # model outputs using pltmr.write_planet_to_txt()
    if plist_case[-1] < ps_tol and mass_dev_frac < mass_tol:
        line = "" # the data line to write into text file
        line += str(model_index) + "\t"
        line += "{:.6f}".format(pm/1.0e9) + " GPa\t" # central pressure in GPa
        line += "{:.6f}".format(pmass_calc_case/smr.Me) + " Me\t" # planet mass in Me
        line += "{:.6f}".format(np.abs(pmass_calc_case/smr.Me - pmass_measured)) + "\t" # mass deviation
        line += "{:.6f}".format(mass_dev_frac * 100.0) + "\t" # mass deviation %
        line += "{:.6f}".format(rlist_case[-1]/smr.Re) + " Re\t" # planet radius in Re
        line += "{:.6f}".format(np.abs(rlist_case[-1]/smr.Re - prad_measured)) + "\t" # radius deviation
        line += "{:.6f}".format(np.abs(rlist_case[-1]/smr.Re - prad_measured)/prad_measured * 100.0) + "\t" # radius deviation %
        line += "{:.6f}".format(plist_case[-1]/1.0e5) + " bar\t"
        line += "{:.6f}".format(tlist_case[-1]) + " K\t"
        line += str(recursion_i) + "\t"
        line += "converged"
        line += "\n"
        
        with open(outname, "a") as outfile:
            outfile.write(line)
    
        # write the converged model to text file
        pltmr.write_planet_to_txt(mlist_case, rlist_case, plist_case, rholist_case, mb_case, 
                                  rb_case, pb_case, tlist_case, tb_case, lnames_case, 
                                  pname='Uranus_forwardPcBisect_model'+str(model_index))

        return # end recursion
    
    # If maximum recursion depth is reached, flag it as "max_recursion"
    if recursion_i >= recursion_max:
        line = "" # the data line to write into text file
        line += str(model_index) + "\t"
        line += "{:.6f}".format(pm/1.0e9) + " GPa\t" # central pressure in GPa
        line += "{:.6f}".format(pmass_calc_case/smr.Me) + " Me\t" # planet mass in Me
        line += "{:.6f}".format(np.abs(pmass_calc_case/smr.Me - pmass_measured)) + "\t" # mass deviation
        line += "{:.6f}".format(mass_dev_frac * 100.0) + "\t" # mass deviation %
        line += "{:.6f}".format(rlist_case[-1]/smr.Re) + " Re\t" # planet radius in Re
        line += "{:.6f}".format(np.abs(rlist_case[-1]/smr.Re - prad_measured)) + "\t" # radius deviation
        line += "{:.6f}".format(np.abs(rlist_case[-1]/smr.Re - prad_measured)/prad_measured * 100.0) + "\t" # radius deviation %
        line += "{:.6f}".format(plist_case[-1]/1.0e5) + " bar\t"
        line += "{:.6f}".format(tlist_case[-1]) + " K\t"
        line += str(recursion_i) + "\t"
        line += "max_recursion"
        line += "\n"
    
        with open(outname, "a") as outfile:
            outfile.write(line)

        return # end recursion
    
    # Else if surface pressure is too large, keep p1 unchanged and pass pm to the recursive function as the new p2
    if plist_case[-1] > ps_tol:
        return solve_forwardMR_pc_bisect_case(p1, pm, pmass, lmf, ps, tc, rc, delr, bc_ignore, pradius, mmw, teff, 
                                              teq, guillot_f, guillot_gamma, tmode_core, maxsteps, eos_core, 
                                              eos_mantle, model_index, pmass_measured, prad_measured, outname, 
                                              recursion_i, recursion_max, ps_tol, mass_tol)
    # Else if total calculated mass is too small, keep p2 unchanged and pass pm to the recursive function as the new p1
    elif mass_dev_frac > mass_tol and pmass_calc_case/smr.Me < pmass_measured:
        return solve_forwardMR_pc_bisect_case(pm, p2, pmass, lmf, ps, tc, rc, delr, bc_ignore, pradius, mmw, teff, 
                                              teq, guillot_f, guillot_gamma, tmode_core, maxsteps, eos_core, 
                                              eos_mantle, model_index, pmass_measured, prad_measured, outname, 
                                              recursion_i, recursion_max, ps_tol, mass_tol)
    # This case should not happen, but included to capture error. Flag it with "err".
    else:
        line = "" # the data line to write into text file
        line += str(model_index) + "\t"
        line += "{:.6f}".format(pm/1.0e9) + " GPa\t" # central pressure in GPa
        line += "{:.6f}".format(pmass_calc_case/smr.Me) + " Me\t" # planet mass in Me
        line += "{:.6f}".format(np.abs(pmass_calc_case/smr.Me - pmass_measured)) + "\t" # mass deviation
        line += "{:.6f}".format(mass_dev_frac * 100.0) + "\t" # mass deviation %
        line += "{:.6f}".format(rlist_case[-1]/smr.Re) + " Re\t" # planet radius in Re
        line += "{:.6f}".format(np.abs(rlist_case[-1]/smr.Re - prad_measured)) + "\t" # radius deviation
        line += "{:.6f}".format(np.abs(rlist_case[-1]/smr.Re - prad_measured)/prad_measured * 100.0) + "\t" # radius deviation %
        line += "{:.6f}".format(plist_case[-1]/1.0e5) + " bar\t"
        line += "{:.6f}".format(tlist_case[-1]) + " K\t"
        line += str(recursion_i) + "\t"
        line += "err"
        line += "\n"

        with open(outname, "a") as outfile:
            outfile.write(line)

        return # end recursion


def run_forwardMR_parallel_test():
    """ The main function. Set up parallel run parameters, and start parallel run using Pool. """
    # set up input parameters, and run run_forwardMR_case() in parallel
    pmass_li = [14.5322753*smr.Me] * 3
    lmf_li = [[0, 70000, 790000, 140000], [0, 60000, 800000, 140000], [0, 80000, 720000, 200000]]
    pc_li = [1200.0e9, 1000.0e9, 1100.0e9]
    ps_li = [1.0e5] * 3
    tc_li = [8500.0] * 3
    rc_li = [10.0] * 3
    delr_li = [100.0] * 3
    bc_ig_li = [''] * 3
    pradius_li = [3.97931*smr.Re] * 3
    mmw_li = [3.8504579876320796e-27] * 3
    teff_li = [50.0] * 3
    teq_li = [76.0] * 3
    gf_li = [0.5] * 3
    gg_li = [1.0] * 3
    tmodec_li = ["isothermal"] * 3
    maxsteps_li = [2000000] * 3
    eosc_li = ["tab_Zeng21 Zeng2021_core"] * 3
    eosm_li = ["tab_Zeng21 Zeng2021_mantle"] * 3
    index_li = [0, 1, 2]
    m_measured_li = [14.53] * 3
    r_measured_li = [4.007] * 3
    outname_li = ["Uranus_test.txt"] * 3

    args = list(zip(pmass_li, lmf_li, pc_li, ps_li, tc_li, rc_li, delr_li, bc_ig_li, pradius_li,
                    mmw_li, teff_li, teq_li, gf_li, gg_li, tmodec_li, maxsteps_li, eosc_li, eosm_li,
                    index_li, m_measured_li, r_measured_li, outname_li))
    # print("110", len(args), len(args[0]))
    
    # begin parallel runs
    with open("Uranus_test.txt", "a") as outfile:
        outfile.write("#index\tmass\tdelM\tdelM%\tradius\tdelR\tdelR%\tPs\tTs\n")
            
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')

    with Pool() as pool:
        pool.starmap(run_forwardMR_case, args)

    print()
    print('************************ Parallel run ended ************************')
    print()
    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def solve_forwardMR_pc_bisect_test():
    """ Test finding Pc using bisection for multiple cases in parallel. """
    # set up input parameters, and run run_forwardMR_case() in parallel
    p1_li = [500.0e9] * 3
    p2_li = [2500.0e9] * 3
    pmass_li = [14.5322753*smr.Me] * 3
    lmf_li = [[0, 70000, 790000, 140000], [0, 60000, 800000, 140000], [0, 80000, 720000, 200000]]
    ps_li = [1.0e5] * 3
    tc_li = [8500.0] * 3
    rc_li = [10.0] * 3
    delr_li = [100.0] * 3
    bc_ig_li = [''] * 3
    pradius_li = [3.97931*smr.Re] * 3
    mmw_li = [3.8504579876320796e-27] * 3
    teff_li = [50.0] * 3
    teq_li = [76.0] * 3
    gf_li = [0.5] * 3
    gg_li = [1.0] * 3
    tmodec_li = ["isothermal"] * 3
    maxsteps_li = [2000000] * 3
    eosc_li = ["tab_Zeng21 Zeng2021_core"] * 3
    eosm_li = ["tab_Zeng21 Zeng2021_mantle"] * 3
    index_li = [0, 1, 2]
    m_measured_li = [14.53] * 3
    r_measured_li = [4.007] * 3
    outname_li = ["Uranus_test_PcBisect.txt"] * 3
    reci_li = [0] * 3
    recm_li = [20] * 3
    ps_tol_li = [1.0e7] * 3
    mass_tol_li = [1.0e-3] * 3

    args = list(zip(p1_li, p2_li, pmass_li, lmf_li, ps_li, tc_li, rc_li, delr_li, bc_ig_li, pradius_li,
                    mmw_li, teff_li, teq_li, gf_li, gg_li, tmodec_li, maxsteps_li, eosc_li, eosm_li,
                    index_li, m_measured_li, r_measured_li, outname_li, reci_li, recm_li, ps_tol_li,    
                    mass_tol_li))
    
    # begin parallel runs
    with open("Uranus_test_PcBisect.txt", "a") as outfile:
        outfile.write("#index\tPc\tmass\tdelM\tdelM%\tradius\tdelR\tdelR%\tPs\tTs\trecursion depth\tflag\n")
    
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')

    with Pool() as pool:
        pool.starmap(solve_forwardMR_pc_bisect_case, args)

    print()
    print('************************ Parallel run ended ************************')
    print()
    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def run_forwardMR_parallel_UranusIG():
    no_models = 318

    pmass_li = [14.5322753*smr.Me] * no_models
    lmf_li = [[0, 5282, 817933, 176785],
[0, 5282, 826052, 168666],
[0, 5282, 834171, 160547],
[0, 5282, 842289, 152429],
[0, 5282, 850408, 144310],
[0, 5282, 858527, 136191],
[0, 5282, 866645, 128073],
[0, 5282, 874764, 119954],
[0, 5282, 882883, 111835],
[0, 5282, 891002, 103716],
[0, 5282, 899120, 95598],
[0, 5282, 907239, 87479],
[0, 5282, 915358, 79360],
[0, 10565, 809814, 179621],
[0, 10565, 817933, 171502],
[0, 10565, 826052, 163383],
[0, 10565, 834171, 155264],
[0, 10565, 842289, 147146],
[0, 10565, 850408, 139027],
[0, 10565, 858527, 130908],
[0, 10565, 866645, 122790],
[0, 10565, 874764, 114671],
[0, 10565, 882883, 106552],
[0, 10565, 891002, 98433],
[0, 10565, 899120, 90315],
[0, 10565, 907239, 82196],
[0, 10565, 915358, 74077],
[0, 15847, 801696, 182457],
[0, 15847, 809814, 174339],
[0, 15847, 817933, 166220],
[0, 15847, 826052, 158101],
[0, 15847, 834171, 149982],
[0, 15847, 842289, 141864],
[0, 15847, 850408, 133745],
[0, 15847, 858527, 125626],
[0, 15847, 866645, 117508],
[0, 15847, 874764, 109389],
[0, 15847, 882883, 101270],
[0, 15847, 891002, 93151],
[0, 15847, 899120, 85033],
[0, 15847, 907239, 76914],
[0, 15847, 915358, 68795],
[0, 21130, 801696, 177174],
[0, 21130, 809814, 169056],
[0, 21130, 817933, 160937],
[0, 21130, 826052, 152818],
[0, 21130, 834171, 144699],
[0, 21130, 842289, 136581],
[0, 21130, 850408, 128462],
[0, 21130, 858527, 120343],
[0, 21130, 866645, 112225],
[0, 21130, 874764, 104106],
[0, 21130, 882883, 95987],
[0, 21130, 891002, 87868],
[0, 21130, 899120, 79750],
[0, 21130, 907239, 71631],
[0, 26412, 793577, 180011],
[0, 26412, 801696, 171892],
[0, 26412, 809814, 163774],
[0, 26412, 817933, 155655],
[0, 26412, 826052, 147536],
[0, 26412, 834171, 139417],
[0, 26412, 842289, 131299],
[0, 26412, 850408, 123180],
[0, 26412, 858527, 115061],
[0, 26412, 866645, 106943],
[0, 26412, 874764, 98824],
[0, 26412, 882883, 90705],
[0, 26412, 891002, 82586],
[0, 26412, 899120, 74468],
[0, 31695, 785458, 182847],
[0, 31695, 793577, 174728],
[0, 31695, 801696, 166609],
[0, 31695, 809814, 158491],
[0, 31695, 817933, 150372],
[0, 31695, 826052, 142253],
[0, 31695, 834171, 134134],
[0, 31695, 842289, 126016],
[0, 31695, 850408, 117897],
[0, 31695, 858527, 109778],
[0, 31695, 866645, 101660],
[0, 31695, 874764, 93541],
[0, 31695, 882883, 85422],
[0, 31695, 891002, 77303],
[0, 31695, 899120, 69185],
[0, 36977, 777340, 185683],
[0, 36977, 785458, 177565],
[0, 36977, 793577, 169446],
[0, 36977, 801696, 161327],
[0, 36977, 809814, 153209],
[0, 36977, 817933, 145090],
[0, 36977, 826052, 136971],
[0, 36977, 834171, 128852],
[0, 36977, 842289, 120734],
[0, 36977, 850408, 112615],
[0, 36977, 858527, 104496],
[0, 36977, 866645, 96378],
[0, 36977, 874764, 88259],
[0, 36977, 882883, 80140],
[0, 42260, 769221, 188519],
[0, 42260, 777340, 180400],
[0, 42260, 785458, 172282],
[0, 42260, 793577, 164163],
[0, 42260, 801696, 156044],
[0, 42260, 809814, 147926],
[0, 42260, 817933, 139807],
[0, 42260, 826052, 131688],
[0, 42260, 834171, 123569],
[0, 42260, 842289, 115451],
[0, 42260, 850408, 107332],
[0, 42260, 858527, 99213],
[0, 42260, 866645, 91095],
[0, 42260, 874764, 82976],
[0, 47542, 761102, 191356],
[0, 47542, 769221, 183237],
[0, 47542, 777340, 175118],
[0, 47542, 785458, 167000],
[0, 47542, 793577, 158881],
[0, 47542, 801696, 150762],
[0, 47542, 809814, 142644],
[0, 47542, 817933, 134525],
[0, 47542, 826052, 126406],
[0, 47542, 834171, 118287],
[0, 47542, 842289, 110169],
[0, 47542, 850408, 102050],
[0, 47542, 858527, 93931],
[0, 47542, 866645, 85813],
[0, 52825, 761102, 186073],
[0, 52825, 769221, 177954],
[0, 52825, 777340, 169835],
[0, 52825, 785458, 161717],
[0, 52825, 793577, 153598],
[0, 52825, 801696, 145479],
[0, 52825, 809814, 137361],
[0, 52825, 817933, 129242],
[0, 52825, 826052, 121123],
[0, 52825, 834171, 113004],
[0, 52825, 842289, 104886],
[0, 52825, 850408, 96767],
[0, 52825, 858527, 88648],
[0, 58107, 752983, 188910],
[0, 58107, 761102, 180791],
[0, 58107, 769221, 172672],
[0, 58107, 777340, 164553],
[0, 58107, 785458, 156435],
[0, 58107, 793577, 148316],
[0, 58107, 801696, 140197],
[0, 58107, 809814, 132079],
[0, 58107, 817933, 123960],
[0, 58107, 826052, 115841],
[0, 58107, 834171, 107722],
[0, 58107, 842289, 99604],
[0, 58107, 850408, 91485],
[0, 63390, 752983, 183627],
[0, 63390, 761102, 175508],
[0, 63390, 769221, 167389],
[0, 63390, 777340, 159270],
[0, 63390, 785458, 151152],
[0, 63390, 793577, 143033],
[0, 63390, 801696, 134914],
[0, 63390, 809814, 126796],
[0, 63390, 817933, 118677],
[0, 63390, 826052, 110558],
[0, 63390, 834171, 102439],
[0, 68672, 744865, 186463],
[0, 68672, 752983, 178345],
[0, 68672, 761102, 170226],
[0, 68672, 769221, 162107],
[0, 68672, 777340, 153988],
[0, 68672, 785458, 145870],
[0, 68672, 793577, 137751],
[0, 68672, 801696, 129632],
[0, 68672, 809814, 121514],
[0, 68672, 817933, 113395],
[0, 68672, 826052, 105276],
[0, 73955, 744865, 181180],
[0, 73955, 752983, 173062],
[0, 73955, 761102, 164943],
[0, 73955, 769221, 156824],
[0, 73955, 777340, 148705],
[0, 73955, 785458, 140587],
[0, 73955, 793577, 132468],
[0, 73955, 801696, 124349],
[0, 73955, 809814, 116231],
[0, 73955, 817933, 108112],
[0, 79237, 736746, 184017],
[0, 79237, 744865, 175898],
[0, 79237, 752983, 167780],
[0, 79237, 761102, 159661],
[0, 79237, 769221, 151542],
[0, 79237, 777340, 143423],
[0, 79237, 785458, 135305],
[0, 79237, 793577, 127186],
[0, 79237, 801696, 119067],
[0, 79237, 809814, 110949],
[0, 84520, 736746, 178734],
[0, 84520, 744865, 170615],
[0, 84520, 752983, 162497],
[0, 84520, 761102, 154378],
[0, 84520, 769221, 146259],
[0, 84520, 777340, 138140],
[0, 84520, 785458, 130022],
[0, 84520, 793577, 121903],
[0, 84520, 801696, 113784],
[0, 84520, 809814, 105666],
[0, 89802, 736746, 173452],
[0, 89802, 744865, 165333],
[0, 89802, 752983, 157215],
[0, 89802, 761102, 149096],
[0, 89802, 769221, 140977],
[0, 89802, 777340, 132858],
[0, 89802, 785458, 124740],
[0, 89802, 793577, 116621],
[0, 89802, 801696, 108502],
[0, 95085, 736746, 168169],
[0, 95085, 744865, 160050],
[0, 95085, 752983, 151932],
[0, 95085, 761102, 143813],
[0, 95085, 769221, 135694],
[0, 95085, 777340, 127575],
[0, 95085, 785458, 119457],
[0, 95085, 793577, 111338],
[0, 95085, 801696, 103219],
[0, 100367, 736746, 162887],
[0, 100367, 744865, 154768],
[0, 100367, 752983, 146650],
[0, 100367, 761102, 138531],
[0, 100367, 769221, 130412],
[0, 100367, 777340, 122293],
[0, 100367, 785458, 114175],
[0, 100367, 793577, 106056],
[0, 105650, 736746, 157604],
[0, 105650, 744865, 149485],
[0, 105650, 752983, 141367],
[0, 105650, 761102, 133248],
[0, 105650, 769221, 125129],
[0, 105650, 777340, 117010],
[0, 105650, 785458, 108892],
[0, 105650, 793577, 100773],
[0, 110932, 728627, 160441],
[0, 110932, 736746, 152322],
[0, 110932, 744865, 144203],
[0, 110932, 752983, 136085],
[0, 110932, 761102, 127966],
[0, 110932, 769221, 119847],
[0, 110932, 777340, 111728],
[0, 110932, 785458, 103610],
[0, 110932, 793577, 95491],
[0, 116215, 720509, 163276],
[0, 116215, 728627, 155158],
[0, 116215, 736746, 147039],
[0, 116215, 744865, 138920],
[0, 116215, 752983, 130802],
[0, 116215, 761102, 122683],
[0, 116215, 769221, 114564],
[0, 116215, 777340, 106445],
[0, 116215, 785458, 98327],
[0, 116215, 793577, 90208],
[0, 121497, 712390, 166113],
[0, 121497, 720509, 157994],
[0, 121497, 728627, 149876],
[0, 121497, 736746, 141757],
[0, 121497, 744865, 133638],
[0, 121497, 752983, 125520],
[0, 121497, 761102, 117401],
[0, 121497, 769221, 109282],
[0, 121497, 777340, 101163],
[0, 121497, 785458, 93045],
[0, 121497, 793577, 84926],
[0, 126780, 704271, 168949],
[0, 126780, 712390, 160830],
[0, 126780, 720509, 152711],
[0, 126780, 728627, 144593],
[0, 126780, 736746, 136474],
[0, 126780, 744865, 128355],
[0, 126780, 752983, 120237],
[0, 126780, 761102, 112118],
[0, 126780, 769221, 103999],
[0, 126780, 777340, 95880],
[0, 126780, 785458, 87762],
[0, 132062, 704271, 163667],
[0, 132062, 712390, 155548],
[0, 132062, 720509, 147429],
[0, 132062, 728627, 139311],
[0, 132062, 736746, 131192],
[0, 132062, 744865, 123073],
[0, 132062, 752983, 114955],
[0, 132062, 761102, 106836],
[0, 132062, 769221, 98717],
[0, 132062, 777340, 90598],
[0, 132062, 785458, 82480],
[0, 137345, 696152, 166503],
[0, 137345, 704271, 158384],
[0, 137345, 712390, 150265],
[0, 137345, 720509, 142146],
[0, 137345, 728627, 134028],
[0, 137345, 736746, 125909],
[0, 137345, 744865, 117790],
[0, 137345, 752983, 109672],
[0, 137345, 761102, 101553],
[0, 137345, 769221, 93434],
[0, 137345, 777340, 85315],
[0, 142628, 696152, 161220],
[0, 142628, 704271, 153101],
[0, 142628, 712390, 144982],
[0, 142628, 720509, 136863],
[0, 142628, 728627, 128745],
[0, 142628, 736746, 120626],
[0, 142628, 744865, 112507],
[0, 142628, 752983, 104389],
[0, 142628, 761102, 96270],
[0, 142628, 769221, 88151],
[0, 147910, 696152, 155938],
[0, 147910, 704271, 147819],
[0, 147910, 712390, 139700],
[0, 147910, 720509, 131581],
[0, 147910, 752983, 99107],
[0, 147910, 761102, 90988]]
    assert len(lmf_li) == no_models
    pc_li = [2000.0e9] * no_models # min 500 GPa, max 2000 GPa
    # assert len(pc_li) == no_models
    ps_li = [1.0e5] * no_models
    tc_li = [8500.0] * no_models
    rc_li = [10.0] * no_models
    delr_li = [1000.0] * no_models
    bc_ig_li = [''] * no_models
    pradius_li = [3.97931*smr.Re] * no_models
    mmw_li = [3.8504579876320796e-27] * no_models
    teff_li = [50.0] * no_models
    teq_li = [76.0] * no_models
    gf_li = [0.5] * no_models
    gg_li = [1.0] * no_models
    tmodec_li = ["isothermal"] * no_models
    maxsteps_li = [2000000] * no_models
    eosc_li = ["tab_Zeng21 Zeng2021_core"] * no_models
    eosm_li = ["tab_Zeng21 Zeng2021_mantle"] * no_models
    index_li = list(range(1, 319))
    assert len(index_li) == no_models
    m_measured_li = [14.53] * no_models
    r_measured_li = [4.007] * no_models
    outname_li = ["Uranus_test.txt"] * no_models

    args = list(zip(pmass_li, lmf_li, pc_li, ps_li, tc_li, rc_li, delr_li, bc_ig_li, pradius_li,
                    mmw_li, teff_li, teq_li, gf_li, gg_li, tmodec_li, maxsteps_li, eosc_li, eosm_li,
                    index_li, m_measured_li, r_measured_li, outname_li))
    # print("468", len(args), len(args[0]))
    
    # begin parallel runs
    with open("Uranus_test.txt", "a") as outfile:
        outfile.write("#index\tmass\tdelM\tdelM%\tradius\tdelR\tdelR%\tPs\tTs\n")
            
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')

    with Pool() as pool:
        pool.starmap(run_forwardMR_case, args)

    print()
    print('************************ Parallel run ended ************************')
    print()
    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def solve_forwardMR_pc_bisect_UranusIG():
    no_models = 318

    p1_li = [500.0e9] * no_models
    p2_li = [2000.0e9] * no_models
    pmass_li = [14.5322753*smr.Me] * no_models
    lmf_li = [[0, 5282, 817933, 176785],
[0, 5282, 826052, 168666],
[0, 5282, 834171, 160547],
[0, 5282, 842289, 152429],
[0, 5282, 850408, 144310],
[0, 5282, 858527, 136191],
[0, 5282, 866645, 128073],
[0, 5282, 874764, 119954],
[0, 5282, 882883, 111835],
[0, 5282, 891002, 103716],
[0, 5282, 899120, 95598],
[0, 5282, 907239, 87479],
[0, 5282, 915358, 79360],
[0, 10565, 809814, 179621],
[0, 10565, 817933, 171502],
[0, 10565, 826052, 163383],
[0, 10565, 834171, 155264],
[0, 10565, 842289, 147146],
[0, 10565, 850408, 139027],
[0, 10565, 858527, 130908],
[0, 10565, 866645, 122790],
[0, 10565, 874764, 114671],
[0, 10565, 882883, 106552],
[0, 10565, 891002, 98433],
[0, 10565, 899120, 90315],
[0, 10565, 907239, 82196],
[0, 10565, 915358, 74077],
[0, 15847, 801696, 182457],
[0, 15847, 809814, 174339],
[0, 15847, 817933, 166220],
[0, 15847, 826052, 158101],
[0, 15847, 834171, 149982],
[0, 15847, 842289, 141864],
[0, 15847, 850408, 133745],
[0, 15847, 858527, 125626],
[0, 15847, 866645, 117508],
[0, 15847, 874764, 109389],
[0, 15847, 882883, 101270],
[0, 15847, 891002, 93151],
[0, 15847, 899120, 85033],
[0, 15847, 907239, 76914],
[0, 15847, 915358, 68795],
[0, 21130, 801696, 177174],
[0, 21130, 809814, 169056],
[0, 21130, 817933, 160937],
[0, 21130, 826052, 152818],
[0, 21130, 834171, 144699],
[0, 21130, 842289, 136581],
[0, 21130, 850408, 128462],
[0, 21130, 858527, 120343],
[0, 21130, 866645, 112225],
[0, 21130, 874764, 104106],
[0, 21130, 882883, 95987],
[0, 21130, 891002, 87868],
[0, 21130, 899120, 79750],
[0, 21130, 907239, 71631],
[0, 26412, 793577, 180011],
[0, 26412, 801696, 171892],
[0, 26412, 809814, 163774],
[0, 26412, 817933, 155655],
[0, 26412, 826052, 147536],
[0, 26412, 834171, 139417],
[0, 26412, 842289, 131299],
[0, 26412, 850408, 123180],
[0, 26412, 858527, 115061],
[0, 26412, 866645, 106943],
[0, 26412, 874764, 98824],
[0, 26412, 882883, 90705],
[0, 26412, 891002, 82586],
[0, 26412, 899120, 74468],
[0, 31695, 785458, 182847],
[0, 31695, 793577, 174728],
[0, 31695, 801696, 166609],
[0, 31695, 809814, 158491],
[0, 31695, 817933, 150372],
[0, 31695, 826052, 142253],
[0, 31695, 834171, 134134],
[0, 31695, 842289, 126016],
[0, 31695, 850408, 117897],
[0, 31695, 858527, 109778],
[0, 31695, 866645, 101660],
[0, 31695, 874764, 93541],
[0, 31695, 882883, 85422],
[0, 31695, 891002, 77303],
[0, 31695, 899120, 69185],
[0, 36977, 777340, 185683],
[0, 36977, 785458, 177565],
[0, 36977, 793577, 169446],
[0, 36977, 801696, 161327],
[0, 36977, 809814, 153209],
[0, 36977, 817933, 145090],
[0, 36977, 826052, 136971],
[0, 36977, 834171, 128852],
[0, 36977, 842289, 120734],
[0, 36977, 850408, 112615],
[0, 36977, 858527, 104496],
[0, 36977, 866645, 96378],
[0, 36977, 874764, 88259],
[0, 36977, 882883, 80140],
[0, 42260, 769221, 188519],
[0, 42260, 777340, 180400],
[0, 42260, 785458, 172282],
[0, 42260, 793577, 164163],
[0, 42260, 801696, 156044],
[0, 42260, 809814, 147926],
[0, 42260, 817933, 139807],
[0, 42260, 826052, 131688],
[0, 42260, 834171, 123569],
[0, 42260, 842289, 115451],
[0, 42260, 850408, 107332],
[0, 42260, 858527, 99213],
[0, 42260, 866645, 91095],
[0, 42260, 874764, 82976],
[0, 47542, 761102, 191356],
[0, 47542, 769221, 183237],
[0, 47542, 777340, 175118],
[0, 47542, 785458, 167000],
[0, 47542, 793577, 158881],
[0, 47542, 801696, 150762],
[0, 47542, 809814, 142644],
[0, 47542, 817933, 134525],
[0, 47542, 826052, 126406],
[0, 47542, 834171, 118287],
[0, 47542, 842289, 110169],
[0, 47542, 850408, 102050],
[0, 47542, 858527, 93931],
[0, 47542, 866645, 85813],
[0, 52825, 761102, 186073],
[0, 52825, 769221, 177954],
[0, 52825, 777340, 169835],
[0, 52825, 785458, 161717],
[0, 52825, 793577, 153598],
[0, 52825, 801696, 145479],
[0, 52825, 809814, 137361],
[0, 52825, 817933, 129242],
[0, 52825, 826052, 121123],
[0, 52825, 834171, 113004],
[0, 52825, 842289, 104886],
[0, 52825, 850408, 96767],
[0, 52825, 858527, 88648],
[0, 58107, 752983, 188910],
[0, 58107, 761102, 180791],
[0, 58107, 769221, 172672],
[0, 58107, 777340, 164553],
[0, 58107, 785458, 156435],
[0, 58107, 793577, 148316],
[0, 58107, 801696, 140197],
[0, 58107, 809814, 132079],
[0, 58107, 817933, 123960],
[0, 58107, 826052, 115841],
[0, 58107, 834171, 107722],
[0, 58107, 842289, 99604],
[0, 58107, 850408, 91485],
[0, 63390, 752983, 183627],
[0, 63390, 761102, 175508],
[0, 63390, 769221, 167389],
[0, 63390, 777340, 159270],
[0, 63390, 785458, 151152],
[0, 63390, 793577, 143033],
[0, 63390, 801696, 134914],
[0, 63390, 809814, 126796],
[0, 63390, 817933, 118677],
[0, 63390, 826052, 110558],
[0, 63390, 834171, 102439],
[0, 68672, 744865, 186463],
[0, 68672, 752983, 178345],
[0, 68672, 761102, 170226],
[0, 68672, 769221, 162107],
[0, 68672, 777340, 153988],
[0, 68672, 785458, 145870],
[0, 68672, 793577, 137751],
[0, 68672, 801696, 129632],
[0, 68672, 809814, 121514],
[0, 68672, 817933, 113395],
[0, 68672, 826052, 105276],
[0, 73955, 744865, 181180],
[0, 73955, 752983, 173062],
[0, 73955, 761102, 164943],
[0, 73955, 769221, 156824],
[0, 73955, 777340, 148705],
[0, 73955, 785458, 140587],
[0, 73955, 793577, 132468],
[0, 73955, 801696, 124349],
[0, 73955, 809814, 116231],
[0, 73955, 817933, 108112],
[0, 79237, 736746, 184017],
[0, 79237, 744865, 175898],
[0, 79237, 752983, 167780],
[0, 79237, 761102, 159661],
[0, 79237, 769221, 151542],
[0, 79237, 777340, 143423],
[0, 79237, 785458, 135305],
[0, 79237, 793577, 127186],
[0, 79237, 801696, 119067],
[0, 79237, 809814, 110949],
[0, 84520, 736746, 178734],
[0, 84520, 744865, 170615],
[0, 84520, 752983, 162497],
[0, 84520, 761102, 154378],
[0, 84520, 769221, 146259],
[0, 84520, 777340, 138140],
[0, 84520, 785458, 130022],
[0, 84520, 793577, 121903],
[0, 84520, 801696, 113784],
[0, 84520, 809814, 105666],
[0, 89802, 736746, 173452],
[0, 89802, 744865, 165333],
[0, 89802, 752983, 157215],
[0, 89802, 761102, 149096],
[0, 89802, 769221, 140977],
[0, 89802, 777340, 132858],
[0, 89802, 785458, 124740],
[0, 89802, 793577, 116621],
[0, 89802, 801696, 108502],
[0, 95085, 736746, 168169],
[0, 95085, 744865, 160050],
[0, 95085, 752983, 151932],
[0, 95085, 761102, 143813],
[0, 95085, 769221, 135694],
[0, 95085, 777340, 127575],
[0, 95085, 785458, 119457],
[0, 95085, 793577, 111338],
[0, 95085, 801696, 103219],
[0, 100367, 736746, 162887],
[0, 100367, 744865, 154768],
[0, 100367, 752983, 146650],
[0, 100367, 761102, 138531],
[0, 100367, 769221, 130412],
[0, 100367, 777340, 122293],
[0, 100367, 785458, 114175],
[0, 100367, 793577, 106056],
[0, 105650, 736746, 157604],
[0, 105650, 744865, 149485],
[0, 105650, 752983, 141367],
[0, 105650, 761102, 133248],
[0, 105650, 769221, 125129],
[0, 105650, 777340, 117010],
[0, 105650, 785458, 108892],
[0, 105650, 793577, 100773],
[0, 110932, 728627, 160441],
[0, 110932, 736746, 152322],
[0, 110932, 744865, 144203],
[0, 110932, 752983, 136085],
[0, 110932, 761102, 127966],
[0, 110932, 769221, 119847],
[0, 110932, 777340, 111728],
[0, 110932, 785458, 103610],
[0, 110932, 793577, 95491],
[0, 116215, 720509, 163276],
[0, 116215, 728627, 155158],
[0, 116215, 736746, 147039],
[0, 116215, 744865, 138920],
[0, 116215, 752983, 130802],
[0, 116215, 761102, 122683],
[0, 116215, 769221, 114564],
[0, 116215, 777340, 106445],
[0, 116215, 785458, 98327],
[0, 116215, 793577, 90208],
[0, 121497, 712390, 166113],
[0, 121497, 720509, 157994],
[0, 121497, 728627, 149876],
[0, 121497, 736746, 141757],
[0, 121497, 744865, 133638],
[0, 121497, 752983, 125520],
[0, 121497, 761102, 117401],
[0, 121497, 769221, 109282],
[0, 121497, 777340, 101163],
[0, 121497, 785458, 93045],
[0, 121497, 793577, 84926],
[0, 126780, 704271, 168949],
[0, 126780, 712390, 160830],
[0, 126780, 720509, 152711],
[0, 126780, 728627, 144593],
[0, 126780, 736746, 136474],
[0, 126780, 744865, 128355],
[0, 126780, 752983, 120237],
[0, 126780, 761102, 112118],
[0, 126780, 769221, 103999],
[0, 126780, 777340, 95880],
[0, 126780, 785458, 87762],
[0, 132062, 704271, 163667],
[0, 132062, 712390, 155548],
[0, 132062, 720509, 147429],
[0, 132062, 728627, 139311],
[0, 132062, 736746, 131192],
[0, 132062, 744865, 123073],
[0, 132062, 752983, 114955],
[0, 132062, 761102, 106836],
[0, 132062, 769221, 98717],
[0, 132062, 777340, 90598],
[0, 132062, 785458, 82480],
[0, 137345, 696152, 166503],
[0, 137345, 704271, 158384],
[0, 137345, 712390, 150265],
[0, 137345, 720509, 142146],
[0, 137345, 728627, 134028],
[0, 137345, 736746, 125909],
[0, 137345, 744865, 117790],
[0, 137345, 752983, 109672],
[0, 137345, 761102, 101553],
[0, 137345, 769221, 93434],
[0, 137345, 777340, 85315],
[0, 142628, 696152, 161220],
[0, 142628, 704271, 153101],
[0, 142628, 712390, 144982],
[0, 142628, 720509, 136863],
[0, 142628, 728627, 128745],
[0, 142628, 736746, 120626],
[0, 142628, 744865, 112507],
[0, 142628, 752983, 104389],
[0, 142628, 761102, 96270],
[0, 142628, 769221, 88151],
[0, 147910, 696152, 155938],
[0, 147910, 704271, 147819],
[0, 147910, 712390, 139700],
[0, 147910, 720509, 131581],
[0, 147910, 752983, 99107],
[0, 147910, 761102, 90988]]
    assert len(lmf_li) == no_models
    # assert len(pc_li) == no_models
    ps_li = [1.0e5] * no_models
    tc_li = [8500.0] * no_models
    rc_li = [10.0] * no_models
    delr_li = [100.0] * no_models
    bc_ig_li = [''] * no_models
    pradius_li = [3.97931*smr.Re] * no_models
    mmw_li = [3.8504579876320796e-27] * no_models
    teff_li = [50.0] * no_models
    teq_li = [76.0] * no_models
    gf_li = [0.5] * no_models
    gg_li = [1.0] * no_models
    tmodec_li = ["isothermal"] * no_models
    maxsteps_li = [2000000] * no_models
    eosc_li = ["tab_Zeng21 Zeng2021_core"] * no_models
    eosm_li = ["tab_Zeng21 Zeng2021_mantle"] * no_models
    index_li = list(range(1, 319))
    assert len(index_li) == no_models
    m_measured_li = [14.53] * no_models
    r_measured_li = [4.007] * no_models
    outname_li = ["Uranus_IG_PcBisect_Tc8500K.txt"] * no_models
    reci_li = [0] * no_models
    recm_li = [20] * no_models
    ps_tol_li = [1.0e7] * no_models
    mass_tol_li = [1.0e-3] * no_models

    args = list(zip(p1_li, p2_li, pmass_li, lmf_li, ps_li, tc_li, rc_li, delr_li, bc_ig_li, pradius_li,
                    mmw_li, teff_li, teq_li, gf_li, gg_li, tmodec_li, maxsteps_li, eosc_li, eosm_li,
                    index_li, m_measured_li, r_measured_li, outname_li, reci_li, recm_li, ps_tol_li,    
                    mass_tol_li))
    
    # begin parallel runs
    with open("Uranus_IG_PcBisect_Tc8500K.txt", "a") as outfile:
        outfile.write("#index\tPc\tmass\tdelM\tdelM%\tradius\tdelR\tdelR%\tPs\tTs\trecursion depth\tflag\n")
    
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')

    with Pool() as pool:
        pool.starmap(solve_forwardMR_pc_bisect_case, args)

    print()
    print('************************ Parallel run ended ************************')
    print()
    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def main():
    # run_forwardMR_parallel_test()
    # solve_forwardMR_pc_bisect_test()
    # run_forwardMR_parallel_UranusIG()
    solve_forwardMR_pc_bisect_UranusIG()


if __name__ == "__main__":
    main()

