#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:32:00 2023

@author: linzifan

This script is used for running CMS simulation using rotating_planet.py
"""

# Import packages
import numpy as np
import datetime
import simpleMR as smr
import gravity_harmonics as grav
from multiprocessing import Pool

# Hardcoded Gaussian quadrature weights and abscissae
quadw_8 = [0.2025782419255610, 0.1984314853271110, 0.1861610000155620, 0.1662692058169930, 0.1395706779261540, 
           0.1071592204671710, 0.0703660474881081, 0.0307532419961173]
quadmu_8 = [0.0000000000000000, 0.2011940939974340, 0.3941513470775630, 0.5709721726085380, 0.7244177313601700, 
            0.8482065834104270, 0.9372733924007060, 0.9879925180204850]

quadw_25 = [0.0634632814047906, 0.0633355092964917, 0.0629527074651957, 0.0623164173200573, 0.0614292009791929,
            0.0602946309531520, 0.0589172757600273, 0.0573026815301875, 0.0554573496748036, 0.0533887107082590, 
            0.0511050943301446, 0.0486156958878282, 0.0459305393555959, 0.0430604369812596, 0.0400169457663730, 
            0.0368123209630007, 0.0334594667916222, 0.0299718846205838, 0.0263636189270660, 0.0226492015874467, 
            0.0188435958530895, 0.0149621449356247, 0.0110205510315936, 0.0070350995900865, 0.0030272789889229]
quadmu_25 = [0.0000000000000000, 0.0634206849826868, 0.1265859972696720, 0.1892415924618130, 0.2511351786125770, 
             0.3120175321197480, 0.3716435012622840, 0.4297729933415760, 0.4861719414524920, 0.5406132469917260, 
             0.5928776941089000, 0.6427548324192370, 0.6900438244251320, 0.7345542542374020, 0.7761068943454460, 
             0.8145344273598550, 0.8496821198441650, 0.8814084455730080, 0.9095856558280730, 0.9341002947558100, 
             0.9548536586741370, 0.9717622009015550, 0.9847578959142130, 0.9937886619441670, 0.9988201506066350]

quadw_32 = [0.0494723666239310, 0.0494118330399182, 0.0492303804237476, 0.0489284528205120, 0.0485067890978838, 
            0.0479664211379951, 0.0473086713122689, 0.0465351492453837, 0.0456477478762926, 0.0446486388259414, 
            0.0435402670830276, 0.0423253450208158, 0.0410068457596664, 0.0395879958915441, 0.0380722675843496, 
            0.0364633700854573, 0.0347652406453559, 0.0329820348837793, 0.0311181166222198, 0.0291780472082805, 
            0.0271665743590979, 0.0250886205533450, 0.0229492710048899, 0.0207537612580391, 0.0185074644601613, 
            0.0162158784103383, 0.0138846126161156, 0.0115193760768800, 0.0091259686763267, 0.0067102917659601, 
            0.0042785083468638, 0.0018398745955771]
quadmu_32 = [0.0000000000000000, 0.0494521871161596, 0.0987833564469453, 0.1478727863578720, 0.1966003467915060, 
             0.2448467932459530, 0.2924940585862510, 0.3394255419745840, 0.3855263942122470, 0.4306837987951110, 
             0.4747872479948040, 0.5177288132900330, 0.5594034094862850, 0.5997090518776250, 0.6385471058213650, 
             0.6758225281149860, 0.7114440995848450, 0.7453246483178470, 0.7773812629903720, 0.8075354957734560, 
             0.8357135543195020, 0.8618464823641230, 0.8858703285078530, 0.9077263027785310, 0.9273609206218430, 
             0.9447261340410090, 0.9597794497589410, 0.9724840346975700, 0.9828088105937270, 0.9907285468921890, 
             0.9962240127779700, 0.9992829840291230]

def read_file_after_last_target_string(filename, target_string):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

            # Find the last occurrence of target_string
            last_index = -1
            for i, line in enumerate(lines):
                if target_string in line:
                    last_index = i

            if last_index == -1:
                print(f"Target string '{target_string}' not found in the file.")
                return None

            # Get lines after the last occurrence
            lines_after_last_target = lines[last_index + 1:]

            return ''.join(lines_after_last_target)

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def run_cms_case(forward_model_filename, layer_boundaries, layers, omega, NL, NM, r_core, 
                 Req, Mtot, Nsteps, savename, del_zeta=1.0e-3, rho_p_func=None):
    """ Given information of a planet's forward structure model, rotating rate, NL, NM, etc., run CMS simulation. """
    assert NM in (8, 25, 32) # only some weights and abscissae are included, in the future should write a function to generate weights and abscissae
    if NM == 8:
        quadmu = quadmu_8; quadw = quadw_8
    elif NM == 25:
        quadmu = quadmu_25; quadw = quadw_25
    elif NM == 32:
        quadmu = quadmu_32; quadw = quadw_32

    # Initialize planet profile
    start_time = datetime.datetime.now()
    arr_i, mu, arr_zeta, arr_r_i0, arr_r, arr_rho_si, arr_rho_pu, arr_p_si, arr_p_pu, arr_t, arr_lambda, \
    arr_delta, arr_Jin, arr_Jinp, arr_Jinpp, arr_u, arr_f, arr_quadw, li_layers = \
    grav.initialize_planet_array(forward_model_filename, # forward model output
                                 layer_boundaries, layers, NL, r_core, omega, quadmu, quadw)
    print('Duration (initialization): {}'.format(datetime.datetime.now() - start_time))
    print()
    
    # Run the first (Newton) step
    start_time = datetime.datetime.now()

    arr_i_new, mu_new, arr_zeta_new, arr_r_i0_new, arr_r_new, arr_rho_si_new, arr_rho_pu_new, \
    arr_p_si_new, arr_p_pu_new, arr_t_new, arr_lambda_new, arr_delta_new, arr_Jin_new, arr_Jinp_new, \
    arr_Jinpp_new, arr_u_new, arr_f_new, arr_quadw_new, li_layers_new = \
    grav.step_planet_array(arr_i, mu, arr_zeta, arr_r_i0, arr_r, arr_rho_si, arr_rho_pu, arr_p_si, arr_p_pu,
                           arr_t, arr_lambda, arr_delta, arr_Jin, arr_Jinp, arr_Jinpp, arr_u, arr_f, arr_quadw,
                           li_layers, omega, Req, Mtot, mode="initial", rho_p_func=rho_p_func)

    print('Duration (first step): {}'.format(datetime.datetime.now() - start_time))

    # Begin the main iteration loop
    input_params = [arr_i_new, mu_new, arr_zeta_new, arr_r_i0_new, arr_r_new, arr_rho_si_new, arr_rho_pu_new,
                arr_p_si_new, arr_p_pu_new, arr_t_new, arr_lambda_new, arr_delta_new, arr_Jin_new, 
                arr_Jinp_new, arr_Jinpp_new, arr_u_new, arr_f_new, arr_quadw_new, li_layers_new,
                omega, Req, Mtot]
    _ = grav.iterate_planet_array(input_params, Nsteps, savename, savefile=True, del_zeta=del_zeta, 
                                  rho_p_func=rho_p_func)


def continue_cms_case(allout_filename, Nsteps, savename, omega, Rp, Mp, del_zeta=1.0e-3, rho_p_func=None):
    """ Given an output file from previous CMS run, continue the iteration by reading in the last step from
    before as initial parameters. """
    # Read in input parameters from output file of a previous run
    input_str = read_file_after_last_target_string(allout_filename, 'Iteration step')
    input_str = input_str.replace("array", "np.array")
    input_params = eval(input_str)
    input_params = input_params + (omega, Rp, Mp)

    # Begin iteration
    _ = grav.iterate_planet_array(input_params, Nsteps, savename, savefile=True, del_zeta=del_zeta, 
                                  rho_p_func=rho_p_func)
    

def run_cms_case_precise(forward_model_filename, layer_boundaries, layers, omega, NL, NM, r_core, 
                         Req, Mtot, Nsteps, savename, del_zeta=1.0, rho_p_func=None):
    """ Given information of a planet's forward structure model, rotating rate, NL, NM, etc., run CMS simulation. """
    assert NM in (8, 25, 32) # only some weights and abscissae are included, in the future should write a function to generate weights and abscissae
    if NM == 8:
        quadmu = quadmu_8; quadw = quadw_8
    elif NM == 25:
        quadmu = quadmu_25; quadw = quadw_25
    elif NM == 32:
        quadmu = quadmu_32; quadw = quadw_32

    # Initialize planet profile
    start_time = datetime.datetime.now()
    arr_i, mu, arr_zeta, arr_r_i0, arr_r, arr_rho_si, arr_rho_pu, arr_p_si, arr_p_pu, arr_t, arr_lambda, \
    arr_delta, arr_Jin, arr_Jinp, arr_Jinpp, arr_u, arr_f, arr_quadw, li_layers = \
    grav.initialize_planet_array_precise(forward_model_filename, # forward model output
                                         layer_boundaries, layers, NL, r_core, omega, quadmu, quadw)
    print('Duration (initialization): {}'.format(datetime.datetime.now() - start_time))
    print()
    
    # Run the first (Newton) step
    start_time = datetime.datetime.now()

    arr_i_new, mu_new, arr_zeta_new, arr_r_i0_new, arr_r_new, arr_rho_si_new, arr_rho_pu_new, \
    arr_p_si_new, arr_p_pu_new, arr_t_new, arr_lambda_new, arr_delta_new, arr_Jin_new, arr_Jinp_new, \
    arr_Jinpp_new, arr_u_new, arr_f_new, arr_quadw_new, li_layers_new = \
    grav.step_planet_array_precise(arr_i, mu, arr_zeta, arr_r_i0, arr_r, arr_rho_si, arr_rho_pu, arr_p_si, 
                                   arr_p_pu, arr_t, arr_lambda, arr_delta, arr_Jin, arr_Jinp, arr_Jinpp, 
                                   arr_u, arr_f, arr_quadw, li_layers, omega, Req, Mtot, mode="initial", 
                                   del_zeta=del_zeta, rho_p_func=rho_p_func)
    
    print('Duration (first step): {}'.format(datetime.datetime.now() - start_time))

    # Begin the main iteration loop
    input_params = [arr_i_new, mu_new, arr_zeta_new, arr_r_i0_new, arr_r_new, arr_rho_si_new, arr_rho_pu_new,
                arr_p_si_new, arr_p_pu_new, arr_t_new, arr_lambda_new, arr_delta_new, arr_Jin_new, 
                arr_Jinp_new, arr_Jinpp_new, arr_u_new, arr_f_new, arr_quadw_new, li_layers_new,
                omega, Req, Mtot]
    _ = grav.iterate_planet_array_precise(input_params, Nsteps, savename, savefile=True, del_zeta=del_zeta, 
                                          rho_p_func=rho_p_func)


def continue_cms_case_precise(allout_filename, Nsteps, savename, omega, Rp, Mp, del_zeta=1.0, rho_p_func=None):
    """ Given an output file from previous CMS run, continue the iteration by reading in the last step from
    before as initial parameters. """
    # Read in input parameters from output file of a previous run
    input_str = read_file_after_last_target_string(allout_filename, 'Iteration step')
    input_str = input_str.replace("array", "np.array")
    input_params = eval(input_str)
    input_params = input_params + (omega, Rp, Mp)

    # Begin iteration
    _ = grav.iterate_planet_array_precise(input_params, Nsteps, savename, savefile=True, del_zeta=del_zeta, 
                                          rho_p_func=rho_p_func)
    

def run_cms_case_parallel(args):
    """ Use the multiprocessing package to run multiple CMS cases at once. Calls run_cms_case(). All arguments are 
    zipped into [args], including model filename, steps, NL, NM, etc. """
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')
    
    with Pool() as pool:
        pool.starmap(run_cms_case, args)

    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def continue_cms_case_parallel(args):
    """ Use the multiprocessing package to continue running multiple CMS cases at once. Calls continue_cms_case().
    All arguments are zipped into [args]. """
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')

    with Pool() as pool:
        pool.starmap(continue_cms_case, args)

    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def run_cms_case_parallel_precise(args):
    """ Use the multiprocessing package to run multiple CMS cases at once. Calls run_cms_case(). All arguments are 
    zipped into [args], including model filename, steps, NL, NM, etc. """
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')
    
    with Pool() as pool:
        pool.starmap(run_cms_case_precise, args)

    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


def continue_cms_case_parallel_precise(args):
    """ Use the multiprocessing package to continue running multiple CMS cases at once. Calls continue_cms_case().
    All arguments are zipped into [args]. """
    start_time = datetime.datetime.now()
    print('Beginning parallel run...')

    with Pool() as pool:
        pool.starmap(continue_cms_case_precise, args)

    print('Duration (parallel run): {}'.format(datetime.datetime.now() - start_time))
    print()


if __name__ == "__main__":
    # Change parameters here for each run
    run_cms_case("runs/23.9.1_CMS_Uranus/forward_models_IG/Uranus_model_11.txt",
                 [0.5980238581*smr.Re, 1.0540590174*smr.Re, 2.5704771621*smr.Re, 4.0377036572*smr.Re],
                 ["core", "mantle", "water", "H/He"], 
                 2 * np.pi / (0.71832 * 24 * 3600), # omega of Uranus
                 20, 32, # NL and NM
                 1000.0e3, # R_core, which is arbitrary
                 25559.0e3, 8.6663789767e+25, # total radius and mass of Uranus in SI
                 10, # Nsteps
                 "CMS_UranusM11_NL20_NM32_step1-10",
                 del_zeta=[1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5] + [1.0e-6]*6 # shape function step size
                 )