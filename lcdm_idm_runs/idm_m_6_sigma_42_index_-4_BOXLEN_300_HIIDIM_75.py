import numpy as np 

import py21cmfast as p21c # To run 21cmFirstCLASS (21cmFAST)
from py21cmfast import plotting # For plotting global signals, coeval boxes and lightcone boxes
import py21cmfast.power_spectrum as ps # Calculate power spectrum from the lightcone
from py21cmfast.inputs import global_params # Useful in this tutorial to plot the initial conditions for the simulation

import dmeff_classy
print("class_dmeff_version:", dmeff_classy.__version__)

import classy
print("class_version:", classy.__version__)

# ## 1st run: &Lambda;CDM
# The first simulation we shall perform is similar to the one we had in Notebook #1, that is under the assumption of &Lambda;CDM.
# 
# Before running the simulation we need to set its parameters. We begin with setting 'user_params'.
# 
# ### Important Note!
# To speed up the computation we use a rather small box with low resolution. Altough these settings are fine to obtain the correct global history, they are inadequate for studying the fluctuations of the box, or the power spectrum. Make sure to increase BOX_LEN and HII_DIM for reliable simulation of the power spectrum!

# Parameters related with the size of the simulation itself and with the kind of outputs required

user_params = {"BOX_LEN": 300, # size of the simulated box (in comoving Mpc) 
               "HII_DIM": 75, # number of cells along each axis of the coeval box - Note: more cells means longer runtime! 
               "N_THREADS": 6} # whether or not to run CLASS prior to the 21cmFAST simulation.

# Now we set the other parameters for the simulation.

# Parameters that set the physical processes the code accounts for

flag_options = {"USE_MINI_HALOS": False, } # if False, popIII stars are not included - Note: if set to True, the runtime increases significantly!

# Cosmological parameters in LCDM

cosmo_params = {"hlittle": 0.6736, # hubble parameter
                "OMb": 0.0493, # baryon density
                "OMm": 0.3153, # matter (CDM+baryon) density
                "A_s": 2.1e-9, # amplitude of the primordial fluctuations
                "POWER_INDEX": 0.9649, # spectral index of the primordial spectrum
                "tau_reio": 0.0544, # optical depth to reionization
                }

# Astrophysical parameters in a standard scenario
# some parameters require to input the log10 value

astro_params = {"F_STAR10": -1.25, # star formation efficiency (atomic cooling galaxies) for pivot mass 1e10 Msun (log10)
                "ALPHA_STAR": 0.5, # slope of the dependency of star formation efficiency on the host halo mass 
                "F_ESC10": -1.35, # escape fraction of Lyman photons into the IGM for pivot mass 1e10 Msun (log10)
                "ALPHA_ESC": -0.3, # slope of the dependency of escape fraction on the host halo mass 
                "L_X": 40.5, # X-ray luminosity (log10)
        
# popIII stars parameters -- use only if USE_MINI_HALOS = True    
                #"F_STAR7_MINI": -2.5, # star formation efficiency (molecular cooling galaxies) for pivot mass 1e7 Msun (log10)
                #"ALPHA_STAR_MINI": 0., # slope of the dependency of star formation efficiency on the host halo mass 
                #"F_ESC7_MINI": -1.35, # escape fraction of Lyman photons into the IGM for molecular cooling galaxies, assumed constant as function of the halo mass (log10)
                # "L_X_MINI": 40.5, # Xray luminosity 
}

global_quantities = ("brightness_temp", # brightness temperature
                     "J_Lya_box", # Lyman alpha flux 
                     "Tk_box", # baryon kinetic temperature
                     "T_chi_box", # SDM temperature
                     "V_chi_b_box") # Relative velocity between baryons and SDM 

lightcone_quantities = ("brightness_temp",) # brightness temperature

# ## 3rd run: scattering dark matter (SDM)
# Let's now explore a beyond LCDM model that affects the 21cm signal both due to changes in the initial power spectrum shape and because it affects the evolution of the thermal and reionization histories. For this, we choose to rely on the scattering dark matter (SDM) model.
# 
# In the scattering dark matter model, dark matter particles (often denoted by $\chi$) elastically scatter off standard model particles. As a consequence, heat flows from the hotter baryons to the colder SDM. In addition, the friction between the fluids results in a drag force that tends to lower the relative velocity as well as heating up both fluids.
# 
# The cross-section of the interaction is usually modeled as $\sigma=\sigma_{n}\left(v/c\right)^n$. Thus, the SDM model has four extra parameters to be determined:
# 1. The SDM particle mass $m_\chi$.
# 2. The SDM fraction $f_\chi$ from the total dark matter.
# 3. The power-law index of the cross section $n$.
# 4. The amplitude of the cross section $\sigma_n$.
# 5. In addition, different SDM models can differ by the target particles that interact with the SDM particles. This choice can change the interaction rate by orders of magnitude, depending on the mass and number density of the target particles.

# 21cmFirstCLASS allows you to perform simulations that include the SDM component; to do so, it relies on dmeff_CLASS, which you installed at the beginning of this notebook. When the SCATTERING_DM flag is set on True, the code runs dmeff_classy instead of the standard classy to compute the initial condition. Moreover, the implementation of the temperature evolution inside the modules of 21cmFirstCLASS runs differently when SDM is included. In order to simulate SDM in 21cmFirstCLASS, you need to set the SCATTERING_DM flag on true, as well as to specify all the above SDM parameters.

# Include scattering dark matter in the simulation
user_params['SCATTERING_DM'] = True
user_params['MANY_Z_SAMPLES_AT_COSMIC_DAWN'] = False
# user_params['MANY_Z_SAMPLES_AT_COSMIC_DAWN'] = True

# Specify the SDM properties
cosmo_params['m_chi'] = 6. # mass of the SDM particle. This is in fact log_10(m_chi/eV), thus we simulate here SDM with a mass of m_chi=1 MeV.
cosmo_params['f_chi'] = 0. # fraction of SDM. This is in fact -log_10(f_chi), thus we simulate here SDM with a fraction of 100%.
cosmo_params['SDM_INDEX'] = -4. # index of the SDM cross-section. Note that it doesn't have to be an integer!
cosmo_params['sigma_SDM'] = 42. # amplitude of SDM cross-section. This is in fact -log_10(sigma_SDM/cm^2), thus we simulate here SDM with a cross section of sigma = 1e-42*(v/c)^-4 cm^2.

# We also need to specify the target particles. 
# We have several options to do so. Valid input is either an integer or a string (lower case letters can be used as well)
# 1: BARYONS (all the baryons)
# 2: IONIZED (free protons and electrons)
# 3: HYDROGEN (hydrogen nuclei, neutralized or not)
# 4: PROTONS (free protons)
# 5: ELECTRONS (free electrons)
user_params['SDM_TARGET_TYPE'] = 'BARYONS'

print("idm run starts")
wr_path = '/scratch1/rahimieh/21cmfirstclass_outputs/idm_m_6_sigma_42_index_-4_BOXLEN_300_HIIDIM_75/'
lightcone_SDM = p21c.run_lightcone(redshift = 6., # minimum redshift at which the simulation will stop
                                   random_seed = 1, # numerical seed -- if None, each run will produce different initial conditions, with the same cosmological power spectrum but different spatial realization. You need to specify the random seed to coherently compare two different runs, due to cosmic variance. 
                                   regenerate = True, # create new data even if cached are found
                                   write = True, # whether or no to save cached files
                                   direc= wr_path,
                                   user_params = user_params,
                                   astro_params = astro_params,
                                   flag_options = flag_options,
                                   cosmo_params = cosmo_params,
                                   global_quantities = global_quantities,
                                   lightcone_quantities = lightcone_quantities)

print("idm run ends")
