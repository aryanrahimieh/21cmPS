import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt

import py21cmfast as p21c # To run 21cmFirstCLASS (21cmFAST)
from py21cmfast import plotting # For plotting global signals, coeval boxes and lightcone boxes
import py21cmfast.power_spectrum as ps # Calculate power spectrum from the lightcone
from py21cmfast.inputs import global_params # Useful in this tutorial to plot the initial conditions for the simulation

import dmeff_classy
print("class_dmeff_version:", dmeff_classy.__version__)

import classy
print("class_version:", classy.__version__)

# mpl.rcParams.update({"text.usetex": False, "font.family": "Times new roman"}) # Use latex fonts

colors =  ['#377eb8', '#ff7f00', '#4daf4a',
           '#f781bf', '#a65628', '#984ea3',
           '#999999', '#e41a1c', '#dede00']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) # Set the color palette as default


# ## 1st run: &Lambda;CDM
# The first simulation we shall perform is similar to the one we had in Notebook #1, that is under the assumption of &Lambda;CDM.
# 
# Before running the simulation we need to set its parameters. We begin with setting 'user_params'.
# 
# ### Important Note!
# To speed up the computation we use a rather small box with low resolution. Altough these settings are fine to obtain the correct global history, they are inadequate for studying the fluctuations of the box, or the power spectrum. Make sure to increase BOX_LEN and HII_DIM for reliable simulation of the power spectrum!

# Parameters related with the size of the simulation itself and with the kind of outputs required

user_params = {"BOX_LEN": 200, # size of the simulated box (in comoving Mpc) 
               "HII_DIM": 50, # number of cells along each axis of the coeval box - Note: more cells means longer runtime! 
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


# We now run 21cmFirstCLASS with the above settings.

print("lcdm run start")

lightcone_LCDM = p21c.run_lightcone(redshift = 6., # minimum redshift at which the simulation will stop
                                    random_seed = 1, # numerical seed -- if None, each run will produce different initial conditions, with the same cosmological power spectrum but different spatial realization. You need to specify the random seed to coherently compare two different runs, due to cosmic variance. 
                                    regenerate = True, # create new data even if cached are found
                                    write = False, # whether or no to save cached files
                                    user_params = user_params,
                                    astro_params = astro_params,
                                    flag_options = flag_options,
                                    cosmo_params = cosmo_params,
                                    global_quantities = global_quantities,
                                    lightcone_quantities = lightcone_quantities)

print("lcdm run end")

fig0, ax0 = plotting.plot_global_history(lightcone_LCDM,kind='Tk_box',color='k',label=r'$\Lambda$CDM')
fig0.savefig('./triple_output/Tk_lcdm.png',dpi=100)

# Remember that in the default settings of 21cmFirstCLASS, CLASS runs before the 21cmFAST simulation, in order to generate the initial conditions for 21cmFAST (if you want to know more on new features of 21cmFirstCLASS in &Lambda;CDM - check out Notebook #2!).
# 
# We extract in the cell below the matter power spectrum for the above &Lambda;CDM simulation.

# Extract transfer functions and associated wavenumbers
k = pow(10.,np.array(global_params.LOG_K_ARR_FOR_TRANSFERS))[1:] # 1/Mpc
T_m0_LCDM = np.array(global_params.T_M0_TRANSFER)[1:]

# Primordial curvature power spectrum
k_pivot = 0.05 # 1/Mpc
P_R = lightcone_LCDM.cosmo_params.A_s * pow(k/k_pivot, lightcone_LCDM.cosmo_params.POWER_INDEX-1.)

# Matter density power spectrum (at z=0)
P_m0_LCDM = 2.*np.pi**2 * P_R * T_m0_LCDM**2/k**3


# define the frequency range (50-250 MHz) and resolution (8 MHz) 
freq_bands_boundaries = np.arange(50.,225.+8.,8.); freq_bands_boundaries[-1] = 225.

# 21cm power spectrum in LCDM
power_spectrum_LCDM = ps.lightcone_power_spectrum(lightcone_LCDM,
                                                  freq_bands_boundaries = freq_bands_boundaries)

# And now we compare the power spectra.
# ### Important Note!
# The following power spectrum plots are imprecise, as we have set a small box with very low resolution!
# In order to achieve the correct power spectrum, you must increase BOX_LEN and HII_DIM!

# Here we see that the 21cm power spectrum in FDM has shifted to higher frequencies (lower redshifts). In particular, the &Lambda;CDM signal is stronger at $\nu\sim100\,\mathrm{MHz}$ while the FDM signal is stonger at $\nu\sim125\,\mathrm{MHz}$ due to the delay in the signal.
# 
# Remember that from now on the FUZZY_DM flag is set on True inside this notebook. If you want to disable the FDM computation, set it back to False.

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
# user_params['FUZZY_DM'] = False # We need to tun off FDM as 21cmFirstCLASS currently doesn't support a mixed model of SDM and FDM. We can also leave that on True, and then an automatic logic will set FUZZY_DM to False.

# Specify the SDM properties
cosmo_params['m_chi'] = 6. # mass of the SDM particle. This is in fact log_10(m_chi/eV), thus we simulate here SDM with a mass of m_chi=1 MeV.
cosmo_params['f_chi'] = 0. # fraction of SDM. This is in fact -log_10(f_chi), thus we simulate here SDM with a fraction of 100%.
cosmo_params['SDM_INDEX'] = -4. # index of the SDM cross-section. Note that it doesn't have to be an integer!
# cosmo_params['sigma_SDM'] = 42. # amplitude of SDM cross-section. This is in fact -log_10(sigma_SDM/cm^2), thus we simulate here SDM with a cross section of sigma = 1e-42*(v/c)^-4 cm^2.
# cosmo_params['N_dm'] = 1

# We also need to specify the target particles. 
# We have several options to do so. Valid input is either an integer or a string (lower case letters can be used as well)
# 1: BARYONS (all the baryons)
# 2: IONIZED (free protons and electrons)
# 3: HYDROGEN (hydrogen nuclei, neutralized or not)
# 4: PROTONS (free protons)
# 5: ELECTRONS (free electrons)
user_params['SDM_TARGET_TYPE'] = 'BARYONS'

print("idm run1 starts")
cosmo_params['sigma_SDM'] = 41.69897
lightcone_IDM_nm4_m6_s41p7 = p21c.run_lightcone(redshift = 6., # minimum redshift at which the simulation will stop
                                   random_seed = 1, # numerical seed -- if None, each run will produce different initial conditions, with the same cosmological power spectrum but different spatial realization. You need to specify the random seed to coherently compare two different runs, due to cosmic variance. 
                                   regenerate = True, # create new data even if cached are found
                                   write = False, # whether or no to save cached files
                                   user_params = user_params,
                                   astro_params = astro_params,
                                   flag_options = flag_options,
                                   cosmo_params = cosmo_params,
                                   global_quantities = global_quantities,
                                   lightcone_quantities = lightcone_quantities)

print("idm run1 ends")

print("idm run2 starts")
cosmo_params['sigma_SDM'] = 42. 
lightcone_IDM_nm4_m6_s42 = p21c.run_lightcone(redshift = 6., # minimum redshift at which the simulation will stop
                                   random_seed = 1, # numerical seed -- if None, each run will produce different initial conditions, with the same cosmological power spectrum but different spatial realization. You need to specify the random seed to coherently compare two different runs, due to cosmic variance. 
                                   regenerate = True, # create new data even if cached are found
                                   write = False, # whether or no to save cached files
                                   user_params = user_params,
                                   astro_params = astro_params,
                                   flag_options = flag_options,
                                   cosmo_params = cosmo_params,
                                   global_quantities = global_quantities,
                                   lightcone_quantities = lightcone_quantities)

print("idm run2 ends")

print("idm run3 starts")
cosmo_params['sigma_SDM'] = 43. 
lightcone_IDM_nm4_m6_s43= p21c.run_lightcone(redshift = 6., # minimum redshift at which the simulation will stop
                                   random_seed = 1, # numerical seed -- if None, each run will produce different initial conditions, with the same cosmological power spectrum but different spatial realization. You need to specify the random seed to coherently compare two different runs, due to cosmic variance. 
                                   regenerate = True, # create new data even if cached are found
                                   write = False, # whether or no to save cached files
                                   user_params = user_params,
                                   astro_params = astro_params,
                                   flag_options = flag_options,
                                   cosmo_params = cosmo_params,
                                   global_quantities = global_quantities,
                                   lightcone_quantities = lightcone_quantities)

print("idm run3 ends")

# Although higher cross section are possible, we note that large enough cross-sections may yield inconsistent results, mostly during the dark ages, due to uncalibrated collison coupling for temperatures below 1K (see more details in https://arxiv.org/pdf/2309.03942).

# Notice the warning regarding the MANY_Z_SAMPLES_AT_COSMIC_DAWN flag. We will address it below.

# Remember that when we specified the global quantities at the output, we also specified 'T_chi_box', for the SDM temperature $T_\chi$, and 'V_chi_b_box' for the bulk relative velocity between baryons and SDM $V_{\chi b}$. You can access these new global quantities from the usual global_quantities dictionary.

print(f'Stored global histories are {list(lightcone_IDM_nm4_m6_s41p7.global_quantities.keys())}')

# The direct interaction between SDM and baryons causes the temperature of the latter to significantly decrease.

fig1, ax1 = plotting.plot_global_history(lightcone_LCDM,kind='Tk_box',color='k',label='LCDM')
fig1, ax1 = plotting.plot_global_history(lightcone_IDM_nm4_m6_s41p7,kind='Tk_box',ax=ax1,label='IDM_nm4_m6_s41p7')
fig1, ax1 = plotting.plot_global_history(lightcone_IDM_nm4_m6_s42,  kind='Tk_box',ax=ax1,label='nm4_m6_s42')
fig1, ax1 = plotting.plot_global_history(lightcone_IDM_nm4_m6_s43,  kind='Tk_box',ax=ax1,label='nm4_m6_s43')
fig1.savefig('./triple_output/Tk_n-4_m6_s-42.png',dpi=100)


# It is also interesting to see how SDM temperature reacts to the interaction with the baryons.
# 
# In the settings we have chosen, there is a little effect on the SDM temperature as it is mostly cooled adiabatically by the expansion of the Universe. This is because the SDM particles outnumber the baryons: we chose a mass of $m_\chi=1\,\mathrm{MeV}$, while the typical baryons mass is $m_b\sim1\,\mathrm{GeV}\gg m_\chi$. Since we chose $f_\chi=100\%$ and $\Omega_c/\Omega_b$ is not too large, then $\rho_\chi=f_\chi\rho_c\sim\rho_b$ and thus $n_\chi=\rho_\chi/m_\chi\gg\rho_b/m_b=n_b$. Hence, increasing $m_\chi$ or decreasing $f_\chi$ is expected to have a larger effect on $T_\chi$.

# fig2, ax2 = plotting.plot_global_history(lightcone_SDM,kind='Tk_box',label='$T_k$')
# fig2, ax2 = plotting.plot_global_history(lightcone_SDM,kind='T_chi_box',ax=ax2,label='$T_\chi$')
# ax2.set_ylabel('T[K]',fontsize=20)
# fig2.savefig('temps_n-4_m6_s-42.png',dpi=100)


# Because $T_k$ reaches lower values when SDM is present, the deep in the brightness temperature can reach now much lower values, compared to &Lambda;CDM.


fig3, ax3 = plotting.plot_global_history(lightcone_LCDM,color='k',label='$\Lambda$CDM')
fig3, ax3 = plotting.plot_global_history(lightcone_IDM_nm4_m6_s41p7,ax=ax3,label='IDM_nm4_m6_s41p7')
fig3, ax3 = plotting.plot_global_history(lightcone_IDM_nm4_m6_s42,  ax=ax3,label='nm4_m6_s42')
fig3, ax3 = plotting.plot_global_history(lightcone_IDM_nm4_m6_s43,  ax=ax3,label='nm4_m6_s43')
fig3.savefig('./triple_output/global_n-4_m6_s-42.png',dpi=100)


# Notice that the blue curve at $z\sim20$ seems to be bit 'stiff'. We will fix that in the next run.

# Here we see that not only the brightness temperature looks smoother (as we have more samples), but we also get a <b>different</b> curve when we have more samples. The reason for this is that errors in $T_k$ (due to insufficient $\Delta z$) manifest themselves mostly at the minimum of the brightness temperature (see a detailed explanation for that in Notebook #2).
# 
# Thus, we learn that large SDM cross sections require working with more redshift samples, thereby increasing the overall runtime. Smaller SDM cross sections though do not require having more redshift samples during cosmic dawn, which is why the default value of MANY_Z_SAMPLES_AT_COSMIC_DAWN is False.

# #### Note
# If you inspect the total number of redshift samples in the global quantitites, you will find it is not 173+291=464, but rather 173+84=257.

# The reason for this is because when MANY_Z_SAMPLES_AT_COSMIC_DAWN = True, 21cmFirstCLASS does not compute $x_\mathrm{HI}$ and $T_{21}$ at every iteration during cosmic dawn for which $T_k$ (or $T_s$) is computed. This is because the computation of $x_\mathrm{HI}$ can take a significant amount of time when USE_MINI_HALOS = True, and we only need the finer redshift sampling for getting a precise solution for $T_k$, not for the other fields.

# Like in FDM, we can compute the 21cm power spectrum and see how it is affected by SDM.
# ### Important Note!
# The following power spectrum plots are imprecise, as we have set a small box with very low resolution!
# In order to achieve the correct power spectrum, you must increase BOX_LEN and HII_DIM!
# define the frequency range (50-250 MHz) and resolution (8 MHz) 
freq_bands_boundaries = np.arange(50.,225.+8.,8.); freq_bands_boundaries[-1] = 225.

kk = 0.3

power_spectrum_SDM1 = ps.lightcone_power_spectrum(lightcone_IDM_nm4_m6_s41p7,
                                                  freq_bands_boundaries = freq_bands_boundaries)

power_spectrum_SDM2 = ps.lightcone_power_spectrum(lightcone_IDM_nm4_m6_s42,
                                                  freq_bands_boundaries = freq_bands_boundaries)

power_spectrum_SDM3 = ps.lightcone_power_spectrum(lightcone_IDM_nm4_m6_s43,
                                                  freq_bands_boundaries = freq_bands_boundaries)

# Plot the power spectra
fig5, ax5 = plotting.plot_1d_power_spectrum(power_spectrum_LCDM,
                                           k=kk, # scale in units 1/Mpc
                                           smooth=True,
                                           x_kind = 'frequency',
                                           color='k',
                                           label = '$\Lambda$CDM')

fig5, ax5 = plotting.plot_1d_power_spectrum(power_spectrum_SDM1,
                                           k=kk, # scale in units 1/Mpc
                                           smooth=True,
                                           x_kind = 'frequency',
                                           color = colors[1],
                                           label = 'IDM_nm4_m6_s41p7',
                                           redshift_axis_on_top = True,
                                           ax=ax5) 

fig5, ax5 = plotting.plot_1d_power_spectrum(power_spectrum_SDM2,
                                           k=kk, # scale in units 1/Mpc
                                           smooth=True,
                                           x_kind = 'frequency',
                                           color = colors[1],
                                           label = 'nm4_m6_s42',
                                           redshift_axis_on_top = True,
                                           ax=ax5) 

fig5, ax5 = plotting.plot_1d_power_spectrum(power_spectrum_SDM3,
                                           k=kk, # scale in units 1/Mpc
                                           smooth=True,
                                           x_kind = 'frequency',
                                           color = colors[1],
                                           label = 'nm4_m6_s43',
                                           redshift_axis_on_top = True,
                                           ax=ax5) 

fig5.savefig('./triple_output/PS_n-4_m6_s-42.png',dpi=100)