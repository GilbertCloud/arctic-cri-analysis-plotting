# arctic-cri-analysis-plotting

This repository contains the Python and Jupyter Notebooks for all of the experiments in the paper "A Novel Model Hierarchy Isolates the Limited Effect of Supercooled Liquid Cloud Optics on Infrared Radiation" by Ash Gilbert, Jennifer Kay, and Penny Rowe.

Created by Ash Gilbert (ash.gilbert@colorado.edu) and contact them for any questions.

## Structure

- functions: Contains custom Python functions for Figures 3-6
    - Processing_functions.py
    - Plotting_functions.py

- fig01: Contains Python script to create Figure 1
    - Compare_CRI.py

- fig02: Contains Jupyter Notebook to create Figure 2
    - Paper_figures_mpace_criRFN.ipynb

- fig03: Contains data processing and plotting Jupyter Notebooks to create Figure 3
    - Proc_wnd_nd_ens.ipynb: processing file
    - Ensemble_control_cri_nudge.ipynb: plotting file

- fig04: Contains data processing and plotting Jupyter Notebooks to create Figure 4
    - Proc_wnd_nd_ens_40yr.ipynb: processing file
    - Ensemble_control_cri_nudge_40yr.ipynb: plotting file

- fig05: Contains data processing and plotting Jupyter Notebooks to create Figure 5
    - Proc_wnd_nd_ens_cp.ipynb: processing file
    - Ensemble_control_cri_nudge_coupled.ipynb: plotting file

- fig06: Contains data processing and plotting Jupyter Notebooks to create Figure 6
    - Proc_F1850_extended.ipynb: processing file
    - Free_evolving_control_cri.ipynb: plotting file

- figa01: Contains Python script to create Figure A1
    - Compare_RFN_CESM_optics_paper.py

## Data for Figure creation

The data needed by the Python and Jupyter Notebooks for Figure generation are available at https://doi.org/10.5281/zenodo.15741756. The data required for each figure are broken down below.

Note: For Figures 3-6, the datasets listed are the outputs of the processing notebooks and should only be used as inputs to the plotting notebooks.

- Figure 1: 
    - Water_DW_300.txt
    - water_RFN_240K.txt
    - water_RFN_263K.txt
    - water_RFN_273K.txt

- Figure 2: 
    - tutorial.FSCAM.mpace.cam.h0.2004-10-05-07171.nc
    - cri240K_test.FSCAM.mpace.cam.h0.2004-10-05-07171.nc
    - cri263K_test.FSCAM.mpace.cam.h0.2004-10-05-07171.nc
    - cri273K_test.FSCAM.mpace.cam.h0.2004-10-05-07171.nc

- Figure 3:
    - f.e22.F1850.f09_f09_mg17.control_test_nudge.FLDS.avg.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.control_test_nudge.FLDS.n.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.control_test_nudge.FLDS.std.Mean.All_data.non_filtered.nc

    - f.e22.F1850.f09_f09_mg17.cri240K_test_nudge.FLDS.avg.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri240K_test_nudge.FLDS.n.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri240K_test_nudge.FLDS.std.Mean.All_data.non_filtered.nc

    - f.e22.F1850.f09_f09_mg17.cri263K_test_nudge.FLDS.avg.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri263K_test_nudge.FLDS.n.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri263K_test_nudge.FLDS.std.Mean.All_data.non_filtered.nc

    - f.e22.F1850.f09_f09_mg17.cri273K_test_nudge.FLDS.avg.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri273K_test_nudge.FLDS.n.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri273K_test_nudge.FLDS.std.Mean.All_data.non_filtered.nc

- Figure 4:
    - f.e22.F1850.f09_f09_mg17.control_test_nudge_long.FLDS.avg.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.control_test_nudge_long.FLDS.n.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.control_test_nudge_long.FLDS.std.Mean.All_data.non_filtered.nc

    - f.e22.F1850.f09_f09_mg17.cri263K_test_nudge_long.FLDS.avg.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri263K_test_nudge_long.FLDS.n.Mean.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri263K_test_nudge_long.FLDS.std.Mean.All_data.non_filtered.nc

- Figure 5:
    - b.e22.B1850.f09_g17.control_test_nudge.FLDS.avg.Mean.All_data.non_filtered.nc
    - b.e22.B1850.f09_g17.control_test_nudge.FLDS.n.Mean.All_data.non_filtered.nc
    - b.e22.B1850.f09_g17.control_test_nudge.FLDS.std.Mean.All_data.non_filtered.nc

    - b.e22.B1850.f09_g17.cri263K_test_nudge.FLDS.avg.Mean.All_data.non_filtered.nc
    - b.e22.B1850.f09_g17.cri263K_test_nudge.FLDS.n.Mean.All_data.non_filtered.nc
    - b.e22.B1850.f09_g17.cri263K_test_nudge.FLDS.std.Mean.All_data.non_filtered.nc

- Figure 6:
    - f.e22.F1850.f09_f09_mg17.control_test.FLDS.avg.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.control_test.FLDS.n.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.control_test.FLDS.std.All_data.non_filtered.nc

    - f.e22.F1850.f09_f09_mg17.cri240K_test.FLDS.avg.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri240K_test.FLDS.n.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri240K_test.FLDS.std.All_data.non_filtered.nc

    - f.e22.F1850.f09_f09_mg17.cri263K_test.FLDS.avg.All_data.non_filtered.nc 
    - f.e22.F1850.f09_f09_mg17.cri263K_test.FLDS.n.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri263K_test.FLDS.std.All_data.non_filtered.nc

    - f.e22.F1850.f09_f09_mg17.cri273K_test.FLDS.avg.All_data.non_filtered.nc 
    - f.e22.F1850.f09_f09_mg17.cri273K_test.FLDS.n.All_data.non_filtered.nc
    - f.e22.F1850.f09_f09_mg17.cri273K_test.FLDS.std.All_data.non_filtered.nc









