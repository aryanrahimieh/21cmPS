import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import time


h = 0.7
omega_m = 0.11711/(h**2)
#alpha=0.049 for m_wdm<=3; alpha=0.045 for 3<=m_wdm<=6; alpha=0.043 for 6<=m_wdm
def transfer(k,mwdm,alpha):
    nu = 1.12
    lambda_fs = (alpha*(mwdm**(-1.11))*((omega_m/0.25)**(0.11))*((h/0.7)**1.22))
    alpha1 = lambda_fs
    transfer = (1+(alpha1*k)**(2*nu))**(-5./nu)
    return transfer


fig, ax = plt.subplots(figsize=(8,6))

kvec = np.logspace(0, np.log10(500), 1000) # array of kvec in h/Mpc
ax.plot(kvec, transfer(kvec, 6.1, 0.043)**2, color='k',linestyle='--', label = '6.1 kev wdm')  

ax.set_xscale("log")
ax.legend(loc="lower left", ncol = 1, fontsize=14)
ax.set_ylabel(r"$T^2(k)$",fontsize=20)
ax.set_xlabel(r"$\mathrm{Wavenumber}\ k\ [h/\mathrm{Mpc}]$",fontsize=20)
ax.tick_params('both',length=12,width=0.5,labelsize=20,which='major')
ax.set_xlim(1,400)
ax.set_ylim(0,1.1)


# In[4]:


from dmeff_classy import Class
import dmeff_classy
print(dmeff_classy.__version__)


lcdm_settings = {'output':'mPk',
                  'P_k_max_1/Mpc':500.0,
                  #standard parameters
                  'omega_b':0.023030,
                  'omega_cdm':0.11711,
                  'h':0.7,
                  'A_s':2.15132e-9,
                  'n_s':0.96,
                  'tau_reio':0.0543,
                }

lcdm=Class()
lcdm.set(lcdm_settings)
lcdm.compute()


dmeff_settings = {'output':'mPk',
                  'P_k_max_1/Mpc':500.0,
                  'k_scalar_k_per_decade_for_pk': 40,
#                   'k_scalar_k_per_decade_for_bao':70,
                  #standard parameters
                  'omega_b':0.023030,
                  'omega_cdm':1e-10,
                  'h':0.7,
                  'A_s':2.15132e-9,
                  'n_s':0.96,
                  'tau_reio':0.0543,
                  #dmeff parameters
                  'omega_dmeff': 0.11711, 
#                   'N_dmeff': 1,
                  'npow_dmeff': 0, 
                  'Vrel_dmeff': 0, 
                  'dmeff_target': 'hydrogen',
                  'm_dmeff': 1e-03
                }

N = 1
sigma0=np.logspace(np.log10(3e-29),np.log10(3e-29), N)
sigma0[0]
# dmeff=Class()
# dmeff.set(dmeff_settings)
# dmeff.compute()

dmeff={}
for i in range(len(sigma0)):
    dmeff[i]=Class()
    print(dmeff[i])
    dmeff[i].set(dmeff_settings)
    dmeff[i].set({'sigma_dmeff': sigma0[i]})
    dmeff[i].compute()


# In[8]:


cmap = plt.get_cmap("viridis")
colors = []
for i in range(N):
    colors.append(cmap(i/N))
    

fig, ax = plt.subplots(figsize=(10,8))

kvec = np.logspace(0, np.log10(500), 1000) # array of kvec in h/Mpc
khvec = kvec*h # khvec in 1/Mpc
pk_L = []
for kh in khvec:
    pk_L.append(lcdm.pk(kh,0.)*h**3)

pk_d = {}
for i, sigma in enumerate(sigma0):
    pk_d[i]=[]
    for kh in khvec:
        pk_d[i].append(dmeff[i].pk(kh,0.)*h**3)
    ax.plot(kvec, np.array(pk_d[i])/np.array(pk_L), color=colors[i], linestyle='-',
            label=r'$\sigma_0 = %.1e \rm{cm}^2$'%sigma)
    
ax.plot(kvec, transfer(kvec, 6., 0.049)**2, color='k',  linestyle='--', label = r'$6.0 kev wdm, \alpha=0.049$')  
ax.plot(kvec, transfer(kvec, 6., 0.043)**2, color='red',linestyle='--', label = r'$6.0 kev wdm, \alpha=0.043$')  

ax.set_xscale("log")
ax.legend(loc="lower left", ncol = 1, fontsize=10)
ax.set_ylabel(r"$T^2(k)$",fontsize=20)
ax.set_xlabel(r"$\mathrm{Wavenumber}\ k\ [h/\mathrm{Mpc}]$",fontsize=20)
ax.tick_params('both',length=12,width=0.5,labelsize=20,which='major')
ax.set_xlim(2,500);ax.set_ylim(0,1.1)
n_save = str(dmeff_settings['m_dmeff']) + '_.png'
fig.savefig(n_save,dpi=500)
