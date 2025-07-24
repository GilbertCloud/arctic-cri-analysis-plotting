## Packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.rm'] = 'sans'
# mpl.rcParams['mathtext.it'] = 'sans:italic'
# mpl.rcParams['mathtext.default'] = 'it'

temp = '263K'

## Filepaths
path_to_old_optics = '/Users/ash/Documents/Liquid_optics/CESM_CRI.nc'
path_to_new_optics = '/Users/ash/Documents/Liquid_optics/CESM_CRI_RFN_'+temp+'.nc'
path_to_graphs = '/Users/ash/Documents/Liquid_optics/graphs/'

## Load optics properties
old_optics_data = xr.open_dataset(path_to_old_optics)
new_optics_data = xr.open_dataset(path_to_new_optics)

'''
Graphing format: 2x2 subplots per figure
upper left: old optics as func of mu at midpoint of lambda
upper right: old optics as func of lambda at midpoint of mu
lower left: new optics as func of mu at midpoint of lambda
lower right: new optics as func of lambda at midpoint of mu
'''



## Create arrays for plotting
lw_wvl_edges = old_optics_data.wvl_min_lw.values
lw_wvl_edges = 1e6*np.insert(lw_wvl_edges,0,old_optics_data.wvl_max_lw.values[0])
lw_wvn_edges = 10000/lw_wvl_edges

mu_idxs = np.array([0,4,9,14,19])
mu_fxidx = 9
mus = old_optics_data.mu.values

lam_idxs = np.array([0,12,24,36,49])
lam_fxidx = 24
lambdas = old_optics_data['lambda'].values

clrs = ['k','b','c','m','r']

## Functions for plotting
def wvn2wvl(x):
    return 10000/x

def wvl2wvn(x):
    return 10000/x


## Plot absorption coefficient - LW
# Set up
fig6 = plt.figure(figsize=(6.9,5))
gs = fig6.add_gridspec(2,2,hspace=0.05)
axlist6 = np.array([[fig6.add_subplot(gs[0,0]),fig6.add_subplot(gs[0,1])],[fig6.add_subplot(gs[1,0]),fig6.add_subplot(gs[1,1])]])

# Plot
for i in np.arange(0,len(mu_idxs)):
    mi = mu_idxs[i]
    plot_vals = old_optics_data.k_abs_lw.values[:,lam_fxidx,mi]/1000
    axlist6[0,0].stairs(plot_vals,lw_wvn_edges,baseline=None,linestyle='-',color=clrs[i],label="$\mu = $"+'{:.2f}'.format(mus[mi]),zorder=4)

for i in np.arange(0,len(lam_idxs)):
    li = lam_idxs[i]
    plot_vals = old_optics_data.k_abs_lw.values[:,li,mu_fxidx]/1000
    axlist6[0,1].stairs(plot_vals,lw_wvn_edges,baseline=None,linestyle='--',color=clrs[i],label='$\lambda$ = {:.2e}'.format(lambdas[li,mu_fxidx])+' m$^{-1}$',zorder=4)

for i in np.arange(0,len(mu_idxs)):
    mi = mu_idxs[i]
    plot_vals = (new_optics_data.k_abs_lw.values[:,lam_fxidx,mi]-old_optics_data.k_abs_lw.values[:,lam_fxidx,mi])/1000
    axlist6[1,0].stairs(plot_vals,lw_wvn_edges,baseline=None,linestyle='-',color=clrs[i],label='$\mu$ = {:.2f}'.format(mus[mi]),zorder=4)

for i in np.arange(0,len(lam_idxs)):
    li = lam_idxs[i]
    plot_vals = (new_optics_data.k_abs_lw.values[:,li,mu_fxidx]-old_optics_data.k_abs_lw.values[:,li,mu_fxidx])/1000
    axlist6[1,1].stairs(plot_vals,lw_wvn_edges,baseline=None,linestyle='--',color=clrs[i],label='$\lambda$ = {:.2e}'.format(lambdas[li,mu_fxidx])+' m$^{-1}$',zorder=4)

# Formatting
#axlist6[0,0].set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=13)
axlist6[0,0].set_ylabel('$k_{\mathrm{abs}}$ (RRTMG optics)',fontsize=14)
axlist6[0,0].set_ylim(top=195)
axlist6[0,0].set_xscale('log')
axlist6[0,0].tick_params(labelsize=12, labelbottom=False)
axlist6[0,0].set_title("f($\mu$); {0:.2e} < $\lambda$ < {1:.2e}".format(lambdas[lam_fxidx,mu_idxs[0]],lambdas[lam_fxidx,mu_idxs[-1]]),fontsize=14)
axlist6[0,0].legend(fontsize=10,bbox_to_anchor=(0.2,-1.35),loc='upper left',framealpha=0)
axlist6[0,0].grid(alpha=0.2)
axlist6[0,0].text(0.05,0.88,'(a)',fontsize=14,transform=axlist6[0,0].transAxes)

secax00 = axlist6[0,0].secondary_xaxis('top',functions=(wvn2wvl,wvl2wvn))
secax00.set_xlabel('Wavelength ($\mathrm{\mu}$m)',fontsize=14)
secax00.set_xlim([1000,3])
secax00.tick_params(labelsize=12)

#axlist6[0,1].set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=13)
axlist6[0,1].set_ylim(top=350)
axlist6[0,1].set_xscale('log')
axlist6[0,1].tick_params(labelsize=12, labelbottom=False)
axlist6[0,1].set_title("f($\lambda$); $\mu$ = {:.2f}".format(mus[mu_fxidx]),fontsize=14)
axlist6[0,1].legend(fontsize=10,bbox_to_anchor=(0.2,-1.35),loc='upper left',framealpha=0)
axlist6[0,1].grid(alpha=0.2)
axlist6[0,1].text(0.05,0.88,'(b)',fontsize=14,transform=axlist6[0,1].transAxes)

secax01 = axlist6[0,1].secondary_xaxis('top',functions=(wvn2wvl,wvl2wvn))
secax01.set_xlabel('Wavelength ($\mathrm{\mu}$m)',fontsize=14)
secax01.set_xlim([1000,3])
secax01.tick_params(labelsize=12)

axlist6[1,0].set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=14)
axlist6[1,0].set_ylabel('$k_{\mathrm{abs}}$ (263 K optics$-$\nRRTMG optics)',fontsize=14)
axlist6[1,0].set_ylim([-40,120])
axlist6[1,0].set_xscale('log')
axlist6[1,0].tick_params(labelsize=12)
#axlist6[1,0].legend(fontsize=12)
axlist6[1,0].grid(alpha=0.2)
axlist6[1,0].axhline(0,color='k',alpha=0.5,lw=0.5,zorder=3,ls='--')
axlist6[1,0].text(0.05,0.88,'(c)',fontsize=14,transform=axlist6[1,0].transAxes)

secax10 = axlist6[1,0].secondary_xaxis('top',functions=(wvn2wvl,wvl2wvn))
#secax10.set_xlabel('Wavelength ($\mu$m)',fontsize=13)
secax10.set_xlim([1000,3])
secax10.tick_params(labelsize=12, labeltop=False)

axlist6[1,1].set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=14)
axlist6[1,1].set_ylim([-40,120])
axlist6[1,1].set_xscale('log')
axlist6[1,1].tick_params(labelsize=12)
#axlist6[1,1].legend(fontsize=12)
axlist6[1,1].grid(alpha=0.2)
axlist6[1,1].axhline(0,color='k',alpha=0.5,lw=0.5,zorder=3,ls='--')
axlist6[1,1].text(0.05,0.88,'(d)',fontsize=14,transform=axlist6[1,1].transAxes)

secax11 = axlist6[1,1].secondary_xaxis('top',functions=(wvn2wvl,wvl2wvn))
#secax11.set_xlabel('Wavelength ($\mu$m)',fontsize=13)
secax11.set_xlim([1000,3])
secax11.tick_params(labelsize=12, labeltop=False)

#fig6.suptitle('k$_{abs}$ (m$^{2}$/kg)',fontsize=16,y=0.995)

fig6.savefig(path_to_graphs+'LW_kabs_'+temp+'.pdf',bbox_inches='tight',dpi=400)


