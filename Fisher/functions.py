#!/usr/bin/env python
# coding: utf-8

import numpy as np
import ares
import scipy
from scipy import interpolate
import scipy.stats as stats
import matplotlib.pyplot as plt
import time
import pickle
import os

def data_matrix(nu, params):
    ####################################################################
    # 1- A matrix for saving the data: M.shape= (1+3*npar, lennu)      #
    #         first row: nu, len(nu) = lennu                           #
    #         next 2: first parameter's 2 neighbors                    #
    #         next 2: second parameter's 2 neighbors ...               #
    # returns an array of interpolated dTb as a func. of nu            #
    ####################################################################
    sim = ares.simulations.Global21cm(**params, verbose=False)     # Initializing a simulation object
    sim.run()
    x = sim.history['nu']
    y = sim.history['dTb']
    f = interpolate.interp1d(x, y)
    return f(nu)


def fisher_matrix(der_mat, input_cov):
    ##################################################################
    #3- A function to calculate the Fisher elements using            #
    ### the 21-cm formula.                                           #
    ##################################################################
    fish = der_mat @ np.linalg.inv(input_cov) @ der_mat.T
    return fish

def DataDict(nu, fid_dict, step_dict):
    #################################################################
    # returns a dictionary with these rows:                         #
    # 1,2: frequency and the fiducial temp                          #
    # 2i+1,2i+2: temp(par+) and temp(par-), e.g. T(fX+-step)        #
    #################################################################
    time0 = time.time()
    DD = {}
    DD["nu"] = nu                                 #row1
    DD["Fiducial"] = data_matrix(nu, fid_dict)    #row2
    for name in fid_dict:
        parss = fid_dict.copy()
#         print(parss)
#         dlabel = name+"+"                                     # +run row 2i+1
#         parss.update({name: fid_dict[name]+step_dict[name]})
#         DD[dlabel] = data_matrix(nu, parss)
#         print(parss)
        dlabel = name+"-"                                     # -run row 2i+2
        parss.update({name: fid_dict[name]-step_dict[name]})
        DD[dlabel] = data_matrix(nu, parss)
#         print(parss, "\n")
    print("Data Dictionary Runtime = %1.2f seconds." % (time.time() - time0))
#     print(DD.keys())
    return DD

def DerivDict(datadict, fid_dict, step_dict):
    ##########################################################
    #                                                        #
    ##########################################################
    time0 = time.time()
    partial = {}
    partial["nu"] = datadict["nu"]

    for name in fid_dict:
        parss = fid_dict.copy()
#         print(parss)
#         ul = name + "+"                                        # +label
        dl = name + "-"                                        # -label
#         print(name, ul,dl,step_dict[name])
#         deriv = (datadict[ul] - datadict[dl]) / 2. / step_dict[name]
        deriv = (datadict['Fiducial'] - datadict[dl]) / step_dict[name]
        derivlog10 = deriv * np.log(10) * fid_dict[name]       # correction: derivative with respect to log10(param)
#         print(fid_dict[name])
        partial[name] = deriv
#         print("derivative finished", "\n")
    print("Derivative Dictionary Runtime = %1.8f seconds." % (time.time() - time0))
    print(partial.keys())
    return partial

def radiometer_noise(nu, T408=20, nu408=408, beta=-2.55, dnu=1000, tobs=10*3600, Trec=300):
    #########################################################
    # dnu in Hz, tobs in s, nu in MHz                       #
    # default values: nu408=408 MHz, beta=-2.55,            #
    # T408=20K, dnu=1KHz, tobs=10hr, Trec = 300k            #
    # Output is in [K]                                      #
    #########################################################
    Tsky = T408 * (nu / nu408)**(beta)
    sigma = (Tsky+Trec) / np.sqrt(tobs*dnu)
    # print(Tsky)
    return sigma


def StepFinder(steps,  nu_arr, fid, param='fX'):
    #########################################################
    # This function is designed to find the interval of     #
    # stepsizes for each parameter with the right derivative#
    # Input: param(string)                                  #
    #########################################################
    print('steps array: ',steps,'\n parameter: ',param,'\n frequencies: ', min(nu_arr),',', max(nu_arr))
    print('fid:', fid)
    time0 = time.time()

    fid_dict = {param: fid}
    step_dict = {param: 0.0}
    stability = {}
    for i, step in enumerate(steps):
        step_dict[param] = step
        print(step_dict)
        data_dict  = DataDict(nu_arr,    fid_dict, step_dict)
        deriv_dict = DerivDict(data_dict, fid_dict, step_dict)
        if i==0:
            stability['nu'] = deriv_dict['nu']
        stability[str(step)] = deriv_dict[param]

    print("--- %s seconds ---" % (time.time() - time0))
    return stability


def Plot_der_step(stability_, param):
    #########################################################
    #                                                       #
    #########################################################
    plt.subplots(figsize=(12,6))

    for i,step in enumerate(stability_):
        if i>0:
            plt.plot(stability_['nu'], stability_[str(step)], label= param+"=" + str(step))
    label = "$\partial T / \partial $"+param
    plt.grid();plt.xlabel("$\mathcal{V}$",fontsize=15);plt.ylabel(label,fontsize=15)
    fname = "deriv_nu_"+param+".pdf"
    plt.legend(loc='upper right');plt.savefig(fname)
    return None

def residual_plot(name, fid, std, nu_arr, noise, ylim):
    #########################################################
    #noise input in mK unit                                 #
    #########################################################
    fid_dict = {name: fid}
    step_dict = {name: std}

    data_dict_res = DataDict(nu_arr, fid_dict, step_dict)
    namep = name+"-"

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
    axes1 = plt.subplot(111)
    plt.plot(nu_arr, data_dict_res['Fiducial']-data_dict_res[namep], 'b',
    marker = 'o', markersize = 5, label="Residual = T(fid)- T(fid-1_σ)")
    plt.scatter(nu_arr, noise, marker = 'v', color='k', label="Input Covariance")
    plt.scatter(nu_arr, -noise, marker = '^', color='k')

    plt.fill_between(nu_arr, noise, 100, color='grey', alpha = 0.3)
    plt.fill_between(nu_arr, -noise, -100, color='grey', alpha = 0.3)

    txt = name + ": fid=" + str(fid) + ', 1_σ='+ str(np.round(std,3))
    chi2_value = np.sum(((data_dict_res['Fiducial']-data_dict_res[namep])/noise)**2)
    txt_t = txt + ' ,χ^2=' + str(np.round(chi2_value, 3))
    plt.title(txt_t, fontsize = 15);plt.ylim(-ylim,ylim)
    plt.ylabel('T(mK)',fontsize=15);plt.xlabel("$\mathcal{V}$ (MHz)",fontsize=15)
    plt.legend(fontsize=15)
    print('# of datapoints = ', len(nu_arr))
    print('χ^2 = Σ(ΔTi/δi)^2 = ', np.round(chi2_value, 5))

#     axes2 = plt.subplot(212)
#     plt.plot(nu_arr, np.abs((data_dict_res['Fiducial']-data_dict_res[namep])/noise),
#     label="residual = T(fid)- T(fid-step)")
#     plt.plot(nu_arr, (data_dict_res['Fiducial']-data_dict_res[namep])/noise, "--", color='blue')
#     print(np.sum(((data_dict_res['Fiducial']-data_dict_res[namep])/noise)**2))
#     plt.plot(nu_arr, nu_arr*0, ":", color='red')
#     plt.title('delta T / radio_noise', fontsize = 15)
#     plt.ylim(-1,1)
#     plt.xlabel("$\mathcal{V}$ (MHz)",fontsize=15)

    txt = txt + ".pdf"
    plt.grid();plt.savefig(txt)
    return None

def linear_step(dict0, dict_m):
    #########################################################
    # Combines two dictionaries. Output will give the step  #
    # size in linear space.                                 #
    #########################################################
    dict3 = {}
    for key in dict_m:
    	if key in dict0:
        	dict3[key] = 10**(dict0[key])-10**(dict0[key]-dict_m[key])
    	else:
        	pass
    return dict3

def datadict_oneparam(nu, fid_dictt, p_name, par_ar):
    #################################################################
    # returns a dictionary with these rows:                         #
    # 1,2: frequency and the fiducial temp                          #
    # 2i+1,2i+2: temp(par+) and temp(par-), e.g. T(fX+-step)        #
    #################################################################
    time0 = time.time()
    DD = {}
    DD["nu"] = nu                                   #row1
    DD["Fiducial"] = data_matrix(nu, fid_dictt)    #row2

    for i in range(len(par_ar)):
#         print("i =", i)
        parss = fid_dictt.copy()
#         print("fid =", parss)
        label = p_name+str(par_ar[i])
        parss.update({p_name: par_ar[i]})
        DD[label] = data_matrix(nu, parss)
#         print("dict = ",parss, "\n")

    print("Data Dictionary Runtime = %1.2f seconds." % (time.time() - time0))
#     print(DD.keys())
    return DD

def chi2(datadict, p_name, par_ar, er):
    chi_2 = np.zeros(len(par_ar))
    for i in range(len(par_ar)):
        label = p_name+str(par_ar[i])
        chi_2[i] = np.sum(((datadict[label]-datadict['Fiducial'])/ (1000*er))**2)
    return chi_2

def plot_likelihood(datadict, p_name, par_ar, er):
    fig, ax = plt.subplots(figsize=(15,5))
    chi_2 = chi2(datadict, p_name, par_ar, er)
    plt.plot(par_ar, np.exp(-chi_2), ".", markersize =10)
    plt.xlabel(p_name, fontsize = 15); plt.ylabel("Likelihood", fontsize = 15);plt.grid();
    # ax.xaxis.set_ticks(np.arange(0.98, 1.02, 0.002));
    plt.savefig(p_name+'_Likelihood.png'); plt.close(fig)
    return None

def plotsingleparam(mean, stdev, paramname, xsigma, color, label):
    x = np.linspace(mean - xsigma*stdev, mean + xsigma*stdev, 1000)
    plt.plot(x, np.sqrt(2*np.pi*stdev**2) *stats.norm.pdf(x, mean, stdev), color=color, label = label)
    plt.axvline(x = mean, ymin=0.05, ymax=0.95, color = color, ls='--')
    plt.xlabel(paramname, fontsize=15)
    plt.xscale("log", nonposx='clip')
    return None

def interpolate(nu, x, y):
    f = scipy.interpolate.interp1d(x, y)
    return f(nu)

def interpTAresPkl(filename, nu_array):
    #################################
    # returns dTb                   #
    #################################

    openfile = open(filename,'rb')
    pkl_dict = pickle.load(openfile)
    openfile.close()

    dTb = pkl_dict['dTb'];z = pkl_dict['z'];nu = 1420/(z+1)
    dTb_interp = interpolate(nu_array, nu, dTb)
    return dTb_interp

######################################################################
### Functions for upperbound plotting
######################################################################

def getSigmaListSorted(file_path, idm_mass, lower_sigma, upper_sigma):
    files = os.listdir(file_path)
    sigma_list = []
    counter = 0
    for file in files:
        splts = file.split("_")
        for i, splt in enumerate(splts):
            ## filemass =  ##only for pkl files
            if splt=="mass" and float((splts[i+1]).split('.history', 1)[0]) == idm_mass and lower_sigma <= float(splts[i-1])<= upper_sigma:
                counter += 1
                sigma_list.append(float(splts[i-1]))
    print(counter, "unique cross sections for", idm_mass,"Gev between",
         lower_sigma, ",", upper_sigma, "[cm^2]")
    sigma_list.sort()
    return sigma_list


def plot_67CL_steps(file_path, idm_mass, lower_sigma, upper_sigma, nu_arr, lcdm_ed,
                    inputcov_mat, experiment, print_result=True):
    files = os.listdir(file_path)
######### setting up the plot
    fig,ax = plt.subplots(figsize=(9,5))
######### reading the files for idm_mass and calculating the fisher value for each cross section.
    sigma_list = getSigmaListSorted(file_path, idm_mass, lower_sigma, upper_sigma)
######### initializing lists and counter
    conf_lvls = np.zeros(len(sigma_list))
    step_sizes = np.zeros(len(sigma_list))
############
    for file in files:
        file = file_path + file
        splts = file.split("_")
        for i, splt in enumerate(splts):
            if splt=="mass" and float((splts[i+1]).split('.history', 1)[0]) == idm_mass: # and 1.0e-45 <= float(splts[i-1])<= 1.0e-42:
                # data = np.load(file);nus=data["nu"];dTbs=data["dTb"]
                # dTbs_i = interpolate(nu_arr, nus, dTbs)
                
                dTbs_i = interpTAresPkl(file, nu_arr)
                nus = nu_arr

                label = file[:26]
                derivv = (dTbs_i-lcdm_ed)/float(splts[i-1])
                fish_matt = fisher_matrix(derivv, inputcov_mat)
                I_sigg = np.sqrt(1/fish_matt)
                try:
                    index = sigma_list.index(float(splts[i-1]))
                    conf_lvls[index] = I_sigg
                    step_sizes[index] = float(splts[i-1])
                    # print(derivv)
                except:
                    continue
    plt.plot(step_sizes, conf_lvls, color="cornflowerblue");
    plt.scatter(step_sizes, conf_lvls, color="cornflowerblue")
    ax.set_xscale("log");ax.set_yscale("log")
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel('67%CL $σ_{0}$',fontsize=14);plt.xlabel("stepsize_σ ($cm^{2}$)",fontsize=14)
    name_to_save = "mass_" + str(idm_mass) + "Gev_67%CLs_"+ experiment+ ".png"
    plt.title(name_to_save, fontsize=15);plt.savefig(name_to_save, dpi=500)
    if print_result == True:
        print("step_sizes:", step_sizes, "\nconf_lvls:", conf_lvls)
    return conf_lvls

def plotSignals(file_path, idm_mass, lower_sigma, upper_sigma, freq_arr, lcdm, nu_arr,
                lcdm_ed, noise, experiment):
    files = os.listdir(file_path)
    ### setting up the plot size
    fig,ax = plt.subplots(figsize=(12,8))
    ### Plotting all the signals within the limit of sigma
    sigma_list = getSigmaListSorted(file_path, idm_mass, lower_sigma, upper_sigma)
    N = len(sigma_list)
#     print(sigma_list)
    ### setting up colors
    cmap = plt.get_cmap("viridis")
    colors = []
    order = np.zeros(N)
    counter = -1
    for i in range(N):
        colors.append(cmap(i/N))
####### write the lines here to plot the curves in a sorted manner.
    for file in files:
        label_file = file
        file = file_path + file
        splts = file.split("_")
        for i, splt in enumerate(splts):
            # Split the string at the first dot and select the first part (for pkl files only)
            ### filemass = 

            if splt=="mass" and float((splts[i+1]).split('.history', 1)[0]) == idm_mass: # and 1.0e-45 <= float(splts[i-1])<= 1.0e-42:
                ## pkl
                dTbs = interpTAresPkl(file, freq_arr)
                nus = freq_arr
                # npz
                # data = np.load(file);nus=data["nu"];dTbs=data["dTb"]
                # siggg = float(splts[i-1])/upper_sigma*N
                try:
                    index = sigma_list.index(float(splts[i-1]))
                    counter += 1
                    # print(counter, index, splts[i-1])
                    order[index] = counter
                    # label = "$σ_0 $= "+ splts[i-1] + "[$cm^2$]"
                    label = "$σ_0 $= "+ splts[i-1]
                    label = label + ',fstar='+ splts[i-3] + ',fX='+ splts[i-5] + ',Tmin='+ splts[i-7]
                    # pkl
                    # plt.plot(nus, dTbs, color = colors[index], label = label)
                    if float(splts[i-5])==0.2:
                        plt.plot(nus, dTbs, color = 'r', label = label_file)
                    elif float(splts[i-5])==4:
                        plt.plot(nus, dTbs, color = 'orange', label = label_file)
                    else:
                        plt.plot(nus, dTbs, color = colors[index], label = label_file)
                    # npz
                    # plt.plot(data["nu"], data["dTb"], color = colors[index], label = label)
                except:
                    continue
    order = order.astype(int)
    order_list = list(order)
    order_list.append(N)
### plotting lcdm
    plt.plot(freq_arr, lcdm, label = 'ΛCDM', color= 'black', linestyle="--")
    plt.errorbar(nu_arr, lcdm_ed, noise, ls='none', color= 'royalblue')
    plt.scatter(nu_arr, lcdm_ed, label = 'ΛCDM', color= 'royalblue', linestyle="--")
##############
    handles, labels = plt.gca().get_legend_handles_labels()
##############
    plt.ylabel('$T_{21}[mK]$',fontsize=18);plt.xlabel("$\mathcal{V}$ [MHz]",fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend([handles[j] for j in order_list], [labels[j] for j in order_list], fontsize=12,
               loc='lower left')
    plt.xlim([40,140])
    name_to_save = "mass_" + str(idm_mass) + "Gev_signals_"+ experiment+ ".png"
    plt.title(name_to_save, fontsize=22);plt.savefig(name_to_save, dpi=500)
    return None

def allMasses(file_path):
    files = os.listdir(file_path)
    all_masses = []
    for file in files:
        splts = file.split("_")
        for i, splt in enumerate(splts):
            if splt=="mass":
                all_masses.append(float((splts[i+1]).split('.history', 1)[0]))
    return np.unique(all_masses)

def plotAtlas(file_path):
    files = os.listdir(file_path)

    ### setting up colors
    cmap = plt.get_cmap("rainbow")
    colors = []
    for i in range(len(files)):
        colors.append(cmap(1*i/len(files)))

    fig,ax = plt.subplots(figsize=(15,10))
    for i, file in enumerate(files):
        if file[-3:] == "npz":
            data = np.load(file_path + file);nus=data["nu"];dTbs=data["dTb"]
            if i%50==0:
                ax.plot(data["nu"], data["dTb"]/1000.0, color = colors[i], label = file)
            else:
                ax.plot(data["nu"], data["dTb"]/1000.0, color = colors[i])

    plt.grid();plt.legend()
    return None

def plotRawData(file_path, color, marker = ".", s=10):
    files = os.listdir(file_path)
    # fig, ax = plt.subplots(figsize=(15,10))

    for file in files:
        file = file_path + file
        splts = file.split("_")
        for i, splt in enumerate(splts):
            if splt=="mass":
                plt.scatter(float((splts[i+1]).split('.history', 1)[0]), float(splts[i-1]),
                color=color, marker=marker, s=s)
    # ax.set_xscale("log");ax.set_yscale("log");
    plt.grid();#plt.legend(fontsize=15)
    plt.ylabel('$σ_{0}$ $[cm^{2}]$',fontsize=15);plt.xlabel("$m_{χ}$ [GeV]",fontsize=15)
    return None


def plotResiduals(file_path, idm_mass, lower_sigma, upper_sigma, freq_arr, lcdm, nu_arr,
                lcdm_ed, noise, experiment):
    files = os.listdir(file_path)
    ### setting up the plot size
    fig,ax = plt.subplots(figsize=(12,8))
    ### Plotting all the signals within the limit of sigma
    sigma_list = getSigmaListSorted(file_path, idm_mass, lower_sigma, upper_sigma)
    N = len(sigma_list)
#     print(sigma_list)
    ### setting up colors
    cmap = plt.get_cmap("viridis")
    colors = []
    order = np.zeros(N)
    counter = -1
    for i in range(N):
        colors.append(cmap(i/N))
####### write the lines here to plot the curves in a sorted manner.
    for file in files:
        label_file = file
        file = file_path + file
        splts = file.split("_")
        for i, splt in enumerate(splts):
            # Split the string at the first dot and select the first part (for pkl files only)
            ### filemass = 

            if splt=="mass" and float((splts[i+1]).split('.history', 1)[0]) == idm_mass: # and 1.0e-45 <= float(splts[i-1])<= 1.0e-42:
                ## pkl
                dTbs = interpTAresPkl(file, freq_arr)
                nus = freq_arr
                # npz
                # data = np.load(file);nus=data["nu"];dTbs=data["dTb"]
                # siggg = float(splts[i-1])/upper_sigma*N
                try:
                    index = sigma_list.index(float(splts[i-1]))
                    counter += 1
                    # print(counter, index, splts[i-1])
                    order[index] = counter
                    # label = "$σ_0 $= "+ splts[i-1] + "[$cm^2$]"
                    label = "$σ_0 $= "+ splts[i-1]
                    # label = label + ',fstar='+ splts[i-3] + ',fX='+ splts[i-5] + ',Tmin='+ splts[i-7]
                    # pkl
                    plt.plot(nus, dTbs-lcdm, color = colors[index], label = label)
                    # npz
                    # plt.plot(data["nu"], data["dTb"], color = colors[index], label = label)
                except:
                    continue
    order = order.astype(int)
    order_list = list(order)
    order_list.append(N)
### plotting lcdm
    plt.plot(freq_arr, lcdm-lcdm, label = 'ΛCDM', color= 'black', linestyle="--")
    # plt.errorbar(nu_arr, lcdm_ed-lcdm_ed, noise, ls='none', color= 'royalblue')
    plt.fill_between(nu_arr, noise, 200*lcdm_ed/lcdm_ed, color='grey', alpha=0.3)
    plt.fill_between(nu_arr, -200*lcdm_ed/lcdm_ed, -noise, color='grey', alpha=0.3)
    plt.scatter(nu_arr, noise,  color= 'grey', linestyle="--")
    plt.scatter(nu_arr, -noise, color= 'grey', linestyle="--")
##############
    handles, labels = plt.gca().get_legend_handles_labels()
##############
    plt.ylabel('$T_{21}-T_{ΛCDM}[mK]$',fontsize=18);plt.xlabel("$\mathcal{V}$ [MHz]",fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend([handles[j] for j in order_list], [labels[j] for j in order_list], fontsize=12,
               loc='upper right')
    plt.xlim([40,150]);plt.ylim([-100,100])
    name_to_save = "mass_" + str(idm_mass) + "Gev_residuals_"+ experiment+ ".png"
    plt.title(name_to_save, fontsize=22);plt.savefig(name_to_save, dpi=500)
    return None