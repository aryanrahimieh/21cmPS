import numpy as np 

import py21cmfast as p21c # To run 21cmFirstCLASS (21cmFAST)
from py21cmfast import plotting # For plotting global signals, coeval boxes and lightcone boxes
import py21cmfast.power_spectrum as ps # Calculate power spectrum from the lightcone
from py21cmfast.inputs import global_params # Useful in this tutorial to plot the initial conditions for the simulation

# ## 1st run: &Lambda;CDM
# The first simulation we shall perform is similar to the one we had in Notebook #1, that is under the assumption of &Lambda;CDM.
# 
# Before running the simulation we need to set its parameters. We begin with setting 'user_params'.
# 
# ### Important Note!
# To speed up the computation we use a rather small box with low resolution. Altough these settings are fine to obtain the correct global history, they are inadequate for studying the fluctuations of the box, or the power spectrum. Make sure to increase BOX_LEN and HII_DIM for reliable simulation of the power spectrum!

# Parameters related with the size of the simulation itself and with the kind of outputs required

wr_path = '/scratch1/rahimieh/21cmfirstclass_outputs/lcdm_200_25/'

user_params = {"BOX_LEN": 200, # size of the simulated box (in comoving Mpc) 
               "HII_DIM": 25, # number of cells along each axis of the coeval box - Note: more cells means longer runtime! 
               "N_THREADS": 6} # whether or not to run CLASS prior to the 21cmFAST simulation.

# Now we set the other parameters for the simulation.

# In[6]:


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


# We now run 21cmFirstCLASS with the above settings.

print("lcdm run start")

lightcone_LCDM = p21c.run_lightcone(redshift = 6., # minimum redshift at which the simulation will stop
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

print("lcdm run end")