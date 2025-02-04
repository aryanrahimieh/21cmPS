import scipy
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os

import py21cmfast as p21c # To run 21cmFirstCLASS (21cmFAST)
from py21cmfast import plotting # For plotting global signals, coeval boxes and lightcone boxes
import py21cmfast.power_spectrum as ps # Calculate power spectrum from the lightcone
from py21cmfast.inputs import global_params # Useful in this tutorial to plot the initial conditions for the simulation

import time

#################functions########################################
def derivative(lightcone_fid, lightcone_chngd, param_fid, param_chngd):
    T_fid = lightcone_fid.global_quantities['brightness_temp']
    T_par = lightcone_chngd.global_quantities['brightness_temp']
    deriv = (T_par-T_fid) / (param_chngd-param_fid)
    return deriv

def ps_derivative(ps_dict_fid, ps_dict_chngd, param_fid, param_chngd):
    PS_fid = ps_dict_fid.ps_values[:, 0]
    PS_par = ps_dict_chngd.ps_values[:, 0]
    deriv = (PS_par-PS_fid) / (param_chngd-param_fid)
    return deriv

def run_lcone_sigma(sigma_value, wr_path, 
              user_params, astro_params, flag_options, cosmo_params, global_quantities, lightcone_quantities):
    cosmo_params['sigma_SDM'] = sigma_value
    start_time = time.time()
    lightcone = p21c.run_lightcone(redshift = 6., # minimu
                                        random_seed = 1, # numerical seed -- 
                                        regenerate = False, # create new data even if cached are found
                                        write = True, # whether or no to save cached files
                                        direc= wr_path,
                                        user_params = user_params,
                                        astro_params = astro_params,
                                        flag_options = flag_options,
                                        cosmo_params = cosmo_params,
                                        global_quantities = global_quantities,
                                        lightcone_quantities = lightcone_quantities)
    end_time = time.time();elapsed_time = end_time - start_time;print(f"Elapsed time: {elapsed_time} seconds")
    return lightcone
###############################################################


##########################input_params##########################
user_params = {"BOX_LEN": 800, 
               "HII_DIM": 100,
               "N_THREADS": 6} 

flag_options = {"USE_MINI_HALOS": False, } # if False, popIII stars are not included - Note: if set to True, the runtime increases significantly!

cosmo_params = {"hlittle": 0.6736, # hubble parameter
                "OMb": 0.0493, # baryon density
                "OMm": 0.3153, # matter (CDM+baryon) density
                "A_s": 2.1e-9, # amplitude of the primordial fluctuations
                "POWER_INDEX": 0.9649, # spectral index of the primordial spectrum
                "tau_reio": 0.0544, # optical depth to reionization
                }
                
astro_params = {"F_STAR10": -1.25, # star formation efficiency (atomic cooling galaxies) for pivot mass 1e10 Msun (log10)
                "ALPHA_STAR": 0.5, # slope of the dependency of star formation efficiency on the host halo mass 
                "F_ESC10": -1.35, # escape fraction of Lyman photons into the IGM for pivot mass 1e10 Msun (log10)
                "ALPHA_ESC": -0.3, # slope of the dependency of escape fraction on the host halo mass 
                "L_X": 40.5, # X-ray luminosity (log10)
               }

global_quantities = ("brightness_temp", # brightness temperature
                     "J_Lya_box", # Lyman alpha flux 
                     "Tk_box", # baryon kinetic temperature
                     "T_chi_box", # SDM temperature
                     "V_chi_b_box") # Relative velocity between baryons and SDM 

lightcone_quantities = ("brightness_temp",) # brightness temperature

# Include scattering dark matter in the simulation
user_params['SCATTERING_DM'] = True
user_params['MANY_Z_SAMPLES_AT_COSMIC_DAWN'] = True

# Specify the SDM properties
cosmo_params['m_chi'] = 7. # mass of the SDM particle. This is in fact log_10(m_chi/eV), thus we simulate here SDM with a mass of m_chi=1 MeV.
cosmo_params['f_chi'] = 0. # fraction of SDM. This is in fact -log_10(f_chi), thus we simulate here SDM with a fraction of 100%.
cosmo_params['SDM_INDEX'] = -4. # index of the SDM cross-section. Note that it doesn't have to be an integer!

# We also need to specify the target particles. 
# We have several options to do so. Valid input is either an integer or a string (lower case letters can be used as well)
# 1: BARYONS (all the baryons)
# 2: IONIZED (free protons and electrons)
# 3: HYDROGEN (hydrogen nuclei, neutralized or not)
# 4: PROTONS (free protons)
# 5: ELECTRONS (free electrons)
user_params['SDM_TARGET_TYPE'] = 'BARYONS'
###############################################################


##########running and savinglightcones for diff sigmas#########
sigma_steps = np.linspace(45,50,6)
sigma_fid   = 0
sigma_vals = sigma_steps + sigma_fid
print(sigma_vals)

lightcones = {}
freq_bands_boundaries = np.arange(50.,225.+1.,1.); freq_bands_boundaries[-1] = 225.

subdir_org_data = r"/scratch1/rahimieh/21cmfirstclass_outputs/idm_with_save_manually/" 
subdir_small_data = r"./outputs_m_%1.0f_BOXLEN_%1.0f_HIIDIM_%1.0f/" %(cosmo_params['m_chi'],user_params['BOX_LEN'], user_params['HII_DIM'])

if not os.path.exists(subdir_small_data):
    os.makedirs(subdir_small_data)

for sigma in sigma_vals:
    wr_path = subdir_org_data + r'idm_m_%1.0f_sigma_%1.2f_BOXLEN_%1.0f_HIIDIM_%1.0f/' %(cosmo_params['m_chi'], sigma,
                                                                                         user_params['BOX_LEN'], user_params['HII_DIM'])
    print("\n", wr_path)
    ### 1-running lightcones
    lightcones[wr_path] = run_lcone_sigma(sigma, wr_path, 
              user_params, astro_params, flag_options, cosmo_params, global_quantities, lightcone_quantities)
    ### 2-saving B_temp and redshifts
    B_temps = lightcones[wr_path].global_quantities['brightness_temp']
    z_arr = np.array(lightcones[wr_path].node_redshifts)
    global_temp_filename = subdir_small_data + r"Tb_m_%1.0f_sigma_%1.2f.csv" %(cosmo_params['m_chi'], sigma)
    np.savetxt(global_temp_filename, np.column_stack((B_temps, z_arr)), 
               delimiter=',', fmt='%.6f', header='B_temp,Redshift')
    ### 3-run PS(k,z)
    power_spectrum = ps.lightcone_power_spectrum(lightcones[wr_path], freq_bands_boundaries=freq_bands_boundaries)
    ### 4-save PS(k,z), k, z
    ps_vals_to_save = power_spectrum.ps_values
    kk_vals_to_save = power_spectrum.k_values
    zz_vals_to_save = power_spectrum.z_values
    ps_filename = subdir_small_data +  r"ps_m_%1.0f_sigma_%1.2f.npz" %(cosmo_params['m_chi'], sigma)
    np.savez(ps_filename, ps_2d_arr=ps_vals_to_save, kk_1d_arr=kk_vals_to_save, zz_1d_arr=zz_vals_to_save)
    ### 5-save input params
    user_params_dict =  lightcones[wr_path].user_params.__dict__
    astro_params_dict = lightcones[wr_path].astro_params.__dict__
    astro_params_dict.pop('_StructWrapper__cstruct', None)
    cosmo_params_dict = lightcones[wr_path].cosmo_params.__dict__
    flag_options_dict = lightcones[wr_path].flag_options.__dict__
    flag_options_dict.pop('_StructWrapper__cstruct', None)
    data_w = [user_params_dict, astro_params_dict,cosmo_params_dict, flag_options_dict]
    input_filename = subdir_small_data +  r"input_m_%1.0f_sigma_%1.2f.json" %(cosmo_params['m_chi'], sigma)
    with open(input_filename, 'w') as file:
        json.dump(data_w, file, indent=4)

###############################################################
