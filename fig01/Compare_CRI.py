## Packages
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.rm'] = 'sans'
# mpl.rcParams['mathtext.it'] = 'sans:italic'
# mpl.rcParams['mathtext.default'] = 'it'

## Filepaths
path_to_240KCRI = '/Users/ash/Documents/Rowe_CRI/CRI/water_RFN_240K.txt'
path_to_263KCRI = '/Users/ash/Documents/Rowe_CRI/CRI/water_RFN_263K.txt'
path_to_273KCRI = '/Users/ash/Documents/Rowe_CRI/CRI/water_RFN_273K.txt'
path_to_DWCRI = '/Users/ash/Documents/Rowe_CRI/CRI/Water_DW_300.txt'
path_to_graphs = '/Users/ash/Documents/Liquid_optics/graphs/'

## Load CRI data
# Read in 240K data
wvn240 = [] # cm^-1
rfr240 = [] # real part of refractive index
rfi240 = [] # imaginary part of refractive index

# Read in CRI line-by-line
lines240 = open(path_to_240KCRI).readlines()
for line in lines240[1:]: # skip first few header line - 1: for RFN, 9: for DW
    line = bytes(textwrap.dedent(line),'utf-8')
    entries = line.decode("utf-8").split()

    # Add values to arrays for wavenumber, real & imaginary parts
    if entries[0][0] != '\n':
        wvn240.append(float(entries[0]))
        rfr240.append(float(entries[1]))      
        rfi240.append(float(entries[2]))

# Convert to numpy arrays
wvn240 = np.array(wvn240)
lam240 = (10000/wvn240)
rfr240 = np.array(rfr240)
rfi240 = np.array(rfi240)

# Read in 263K data
wvn263 = [] # cm^-1
rfr263 = [] # real part of refractive index
rfi263 = [] # imaginary part of refractive index

# Read in CRI line-by-line
lines263 = open(path_to_263KCRI).readlines()
for line in lines263[1:]: # skip first few header line - 1: for RFN, 9: for DW
    line = bytes(textwrap.dedent(line),'utf-8')
    entries = line.decode("utf-8").split()

    # Add values to arrays for wavenumber, real & imaginary parts
    if entries[0][0] != '\n':
        wvn263.append(float(entries[0]))
        rfr263.append(float(entries[1]))      
        rfi263.append(float(entries[2]))

# Convert to numpy arrays
wvn263= np.array(wvn263)
lam263 = (10000/wvn263)
rfr263 = np.array(rfr263)
rfi263 = np.array(rfi263)


# Read in 273K data
wvn273 = [] # cm^-1
rfr273 = [] # real part of refractive index
rfi273 = [] # imaginary part of refractive index

# Read in CRI line-by-line
lines273 = open(path_to_273KCRI).readlines()
for line in lines273[1:]: # skip first few header line - 1: for RFN, 9: for DW
    line = bytes(textwrap.dedent(line),'utf-8')
    entries = line.decode("utf-8").split()

    # Add values to arrays for wavenumber, real & imaginary parts
    if entries[0][0] != '\n':
        wvn273.append(float(entries[0]))
        rfr273.append(float(entries[1]))      
        rfi273.append(float(entries[2]))

# Convert to numpy arrays
wvn273= np.array(wvn273)
lam273 = (10000/wvn273)
rfr273 = np.array(rfr273)
rfi273 = np.array(rfi273)

# Read in DW 300K data
wvnDW = [] # cm^-1
rfrDW = [] # real part of refractive index
rfiDW = [] # imaginary part of refractive index

# Read in CRI line-by-line
linesDW = open(path_to_DWCRI).readlines()
for line in linesDW[9:]: # skip first few header line - 1: for RFN, 9: for DW
    line = bytes(textwrap.dedent(line),'utf-8')
    entries = line.decode("utf-8").split()

    # Add values to arrays for wavenumber, real & imaginary parts
    if entries[0][0] != '\n':
        wvnDW.append(float(entries[0]))
        rfrDW.append(float(entries[1]))      
        rfiDW.append(float(entries[2]))

# Convert to numpy arrays
wvnDW = np.array(wvnDW)
lamDW = (10000/wvnDW)
rfrDW = np.array(rfrDW)
rfiDW = np.array(rfiDW)

## Functions for plotting
def wvn2wvl(x):
    return 10000/x

def wvl2wvn(x):
    return 10000/x

slDW = np.logical_and(wvnDW <=3300,wvnDW >=10)
sl240 = np.logical_and(wvn240 <=3300,wvn240 >=10)
sl263 = np.logical_and(wvn263 <=3300,wvn263 >=10)
sl273 = np.logical_and(wvn273 <=3300,wvn273 >=10)

## Plot CRI
# Set up
fig = plt.figure(figsize=(10,8))
gs = fig.add_gridspec(2,1,hspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Plotting CRI
ax1.plot(wvnDW[slDW],rfrDW[slDW],'k',lw=2,zorder=3,label='Downing & Williams 300 K')
ax1.plot(wvn240[sl240],rfr240[sl240],'b',zorder=4,label='Rowe 240 K')
ax1.plot(wvn263[sl263],rfr263[sl263],'tab:gray',zorder=4,label='Rowe 263 K')
ax1.plot(wvn273[sl273],rfr273[sl273],'r',zorder=4,label='Rowe 273 K')

ax2.plot(wvnDW[slDW],rfiDW[slDW],'k',lw=2,zorder=3,label='Downing & Williams 300 K')
ax2.plot(wvn240[sl240],rfi240[sl240],'b',zorder=4,label='Rowe 240 K')
ax2.plot(wvn263[sl263],rfi263[sl263],'tab:gray',zorder=4,label='Rowe 263 K')
ax2.plot(wvn273[sl273],rfi273[sl273],'r',zorder=4,label='Rowe 273 K')

# Plotting windows
# ax1.axvspan(460,600,color='c',zorder=2,alpha=0.4)
# ax1.axvspan(800,1250,color='m',zorder=2,alpha=0.4)

# ax2.axvspan(460,600,color='c',zorder=2,alpha=0.4)
# ax2.axvspan(800,1250,color='m',zorder=2,alpha=0.4)

# Formatting
ax1.set_ylabel('Real part - Scattering (n)',fontsize=14)
ax1.tick_params(axis='y',labelsize=12)
ax1.tick_params(axis='x',labelbottom=False)
ax1.set_xscale('log')
ax1.legend(fontsize=12,loc='lower left')
ax1.set_xlim([10,3300])
ax1.set_xlabel('')
ax1.grid(alpha=0.2)
#ax1.set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=12)
#ax1.set_title('Real part - Scattering (n)',fontsize=14)
# ax1.annotate('Atmospheric\nwindow',xy=(1250,1.85),xytext=(2150,2.1),
#                    arrowprops=dict(facecolor='black',width=2,headwidth=10),fontsize=12,ha='center')
# ax1.annotate('Dirty\nwindow',xy=(460,1.85),xytext=(300,2.1),
#                    arrowprops=dict(facecolor='black',width=2,headwidth=10),fontsize=12,ha='center')

ax2.set_ylabel('Imaginary part - Absorption (k)',fontsize=14)
ax2.tick_params(axis='y',labelsize=12)
ax2.set_xscale('log')
# ax2.legend()
ax2.set_xlim([10,3300]) 
ax2.set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=14)
ax2.tick_params(axis='x',labelsize=12)
ax2.grid(alpha=0.2)
#ax2.set_title('Imaginary part - Absorption (k)',fontsize=14)

# Secondary axis
secax0 = ax1.secondary_xaxis('top',functions=(wvn2wvl,wvl2wvn))
secax0.set_xlabel('Wavelength ($\mathrm{\mu}$m)',fontsize=14)
secax0.set_xlim([1000,3])
secax0.tick_params(axis='x',labelsize=12)

secax1 = ax2.secondary_xaxis('top',functions=(wvn2wvl,wvl2wvn))
#secax1.set_xlabel('Wavelength ($\mu$m)',fontsize=12)
secax1.set_xlim([1000,3])
secax1.tick_params(axis='x',labeltop=False)

ax1.text(0.95,0.91,'(a)',fontsize=14,transform=ax1.transAxes)
ax2.text(0.95,0.91,'(b)',fontsize=14,transform=ax2.transAxes)

fig.savefig(path_to_graphs+'CRI_LW_nowindow.pdf',bbox_inches='tight',dpi=400)

