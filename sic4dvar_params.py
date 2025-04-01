#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:   Callum TYLER, callum.tyler@inrae.fr
            Dylan QUITTARD, dylan.quittard@inrae.fr
                All functions not mentioned above.
            Cécile Cazals, cecile.cazals@cs-soprasteria.com
                logger

Description:
    Parameters file for sic4dvar (algo315).
"""

#NEW: most parameters are now read from the config file
#For Confluence bindings, see default config file.

from pathlib import Path
import numpy as np
import multiprocessing
from datetime import datetime

## Algo31 internal parameters 
## NOTE: do not change these parameters below without contacting Inrae SWOT team directly!
algo_bounds = [
    [5.0],                     ## Discharge (Q) upper/lower bound coefficient 
    [-20.05, -0.05, -0.25],    #-20.05  ## Elevation (Zb) Lower, Upper, Step parameters. ## Igor recommended dZb step to -0.25 7 Oct. '21 # Orig:-20, -0.05, -0.5
    [10.0, 60.0, 5.0]          # 5.0 ## Friction (Km) Lower, Upper, Step parameters
]

algo_bounds[2][2] = (algo_bounds[2][1] - algo_bounds[2][0]) / 8.0 #Change by Igor, friction step parameter computed from bounds

slope_smoothing_num_passes = 0 ## Number of smoothing passes, LSM ## Don't smooth
approx_section_params = [
    8,                         ## Number of additional points to add to section approximation, default 4.
    0.15,                      ## Threshold for max distance, default 0.1.
    2                          ## Sorting algorithm to use, default 2.
]

kDim = 101              ## Dimensions of PDF table
shape03 = 10.0 #5.0           ## Discharge coef. for shape
val1 = (algo_bounds[1][0] + algo_bounds[1][1] ) / 2.0  #-1.0 ## Elevation coef. for delay calculation
shape13= 2.1 #1.1      # 1.2 ## Elevation coef. for shape
val2 = (algo_bounds[2][0] + algo_bounds[2][1]) / 2.0 #Change by Igor, mean value from bounds #old=35.0 ## Friction coef. for delay calculation
shape23 = 2.1 #1.2      #2.1 ## Friction coef. for shape
local_QM1 = 1.0        ## Local QM1 bound #original value from SIC: 20.0

#Other parameters:
######
cython_version = False  ## Use cython version (fast) or python version (slow) #TODO: remove
min_num_nodes = 1       ## Mininum num. of nodes allowed to esti. discharge

## Hind Oubanas :
reachNB = 0             ## reachNB to process

## Parameters of extrapolation method :
extrapolation = True    #Extrapolation of WSE and width of irregular SWOT data
Option = 2              ## Options of old interpolation methods. Option 1 is Hind's version. Option 2 is old Isadora's version. 
LSMX = 10               ## Number of iterations for smoothing in space
LSMT = 1                ## Number of iterations for smoothing in time
corx = 0.2*1000         ## Spatial correlation coefficient in meters #original value: 0.2*1000
cort = 1*6*3600.       ## Temporal correlation coefficient in seconds

# Filtering parameters :
DX_Length = 20 * 1000. #20. *1000.
DX_max_in = np.inf      #original valued 10. * 1000.
DX_max_out = 5. * 1000.
valid_min_z = -1000.
valid_min_dA = -10000000.

figures = False

# Running mode:
node_run = False
DT_obs = 1200.

#Other params
node_length = True          ## Option to use node_length (True) or dist_out (False), node_length preferred.
create_tables = False        ## Option to create tables for SWORD data for faster reading
opt_sword_boost = False      ## POM option to read SWORD tables (faster)
large_deviations = True     ## Run Igor's routine to remove large deviations from raw data 
old_extrapolation = False    ## Option to run old Extrapolation routine
a31_early_stop = False       ##Igor modif to stop if probability big enough

force_specific_dates = False #Option to force start and end dates (if available in SWOT data)
start_date = datetime(2023, 7, 11)
end_date = datetime(2023, 12, 31)

replace_config = True #Option to force a path to your desired config file
#config_file_path = "/mnt/DATA/worksync/Nouveau dossier/sic/stations_sos_v16/input/sic4dvar_param.ini"
#config_file_path = "/mnt/DATA/worksync/Nouveau dossier/sic/stations_sos_v16/input/sic4dvar_param_permissive_filter.ini"
config_file_path = "sic4dvar_param_confluence.ini"

densification = False

num_cores = multiprocessing.cpu_count()
max_cores = int(num_cores/2) #max number of cores to use

no_print = False             ## disable all prints call
#Set this to false for Confluence runs !



##################################################

#Tests
eps1=1e-2	# re-ordering threshold    
eps2=1e-4	# stopping threshold for relaxation sweeps
force_create_reach_t = True

#Isadora params
run_algo31_v3 = False         ## Run Isadora's version of algo31
def_float_atol = 1e-2       ## Isadora routine relaxation sweep

#Pankaj testing/Bathymetry reading
pankaj_test = False           ## Option to read bathymetry from .txt file and use generated .nc file from excel files 

# H.O. : parameters of the likelihood and steady state model
V32 = False
sigmaZ = 1.0
thres = 1.0
DH_Iter=10 # Max number of iterations for calculation of headloss
DH_Esp=0.0001 # Threshold to stop iterations for calculation of headloss
useEXT=True
