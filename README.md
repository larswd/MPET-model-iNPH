# MPET model of infusion test in the human brain
This directory contains the code and algorithm used to conduct simulations in the article _Modeling CSF circulation and the glymphatic system during infusion using subject specific intracranial pressures and brain geometries_ (DOI tbd). 
All simulations performed, with the exception of the subject specific data, can be performed by running the script ´´´MPET_brain.py´´´. This script reads the mesh in ´´´meshes/C0´´´ and writes all data to a ´´´results´´´folder which is generation in runtime. The mesh in the repo is of a lower resolution than used in the article, but a higher resolution mesh can be found at: https://drive.proton.me/urls/ADJQJRPBMM#Jt2WoC4F6CzD

## Command line arguments for the MPET_brain script. 
- ´´´--order´´´, gives an integer specifying the order of the CG polynomials to be used in the Finite Element Analysis
- ´´´--variation´´´, specifies which simulation experiment from the article to be run. Options are "base" for the base model, or ´´´var1´´´, ´´´var2´´´, ´´´var3´´´, ´´´var4´´´ and ´´´var5´´´ which are the model variations described in table 4 in the article.
- ´´´-res´´´ or ´´´--resolution´´´, sets the spatial resolution to be used. Default is 16, the high resolution 32-mesh can be found in the drive link above. 
- ´´´--xdmf´´´ is an integer, either 0 or 1, specifying if you want XDMF files which will allow the user to inspect the pressure fields in 3D using for example paraview. 


## Legacy folder
This folder contains all code used for simulation and postprocessing in my master's thesis "Normal Pressure With Abnormal Geometry: A Biomechanical Model of Normal Pressure Hydrocephalus During Infusion Tests"
The script used to run simulations is in the Main file folder. All scripts for making graphs and posprocessing of data in the postprocessing folder, and analysis scripts in the sensitivity folder.
