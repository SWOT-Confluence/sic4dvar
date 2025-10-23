#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

@authors:   Callum TYLER, callum.tyler@inrae.fr
            Dylan QUITTARD, dylan.quittard@inrae.fr
                All functions not mentioned above.
            Cécile Cazals, cecile.cazals@cs-soprasteria.com
                logger

Description:
    Parameters file for sic4dvar (algo315).
"""
#SIC4DVAR LOW-COST VERSION 2.1
#NEW: most parameters are now read from the config file
#For Confluence bindings, see default config file.

from pathlib import Path
import numpy as np
import multiprocessing
from datetime import datetime
algo_bounds = [[5.0], [-10.01, -0.01, -0.25], [10.0, 60.0, 5.0]]
algo_bounds[2][2] = (algo_bounds[2][1] - algo_bounds[2][0]) / 20.0
slope_smoothing_num_passes = 0
approx_section_params = [8, 0.15, 2]
kDim = 101
shape03 = 10.0
val1 = -1.0
shape13 = 2.1
val2 = 28.6
shape23 = 2.1
local_QM1 = 1.0
cython_version = False
min_num_nodes = 1
reachNB = 0
extrapolation = True
Option = 2
LSMX = 5
LSMT = 1
corx = 0.2 * 1000
cort = 1 * 6 * 3600.0
DX_Length = 20 * 1000.0
DX_max_in = np.inf
DX_max_out = 5.0 * 1000.0
valid_min_z = -1000.0
valid_min_dA = -10000000.0
figures = False
node_run = False
DT_obs = 1200.0
node_length = True
create_tables = False
opt_sword_boost = False
large_deviations = True
old_extrapolation = False
a31_early_stop = True
force_specific_dates = False
start_date = datetime(2023, 7, 12)
end_date = datetime(2025, 12, 31)
replace_config = True
config_file_path = 'sic4dvar_param_confluence.ini'
densification = False
num_cores = multiprocessing.cpu_count()
max_cores = int(num_cores / 2)
no_print = False
q_mean_computed = False
akgd = False
override_cort = False
constant_slope = True
kgokrgo = True
uniform_friction = False
qsdev_activate = False
qsdev_option = 0
indicators_experimental_computation = True
use_fallback_sos = False
use_dynamic_spread = False
use_mean_for_bounds = True
relative_variance = 1.0
auto_spread_for_runs = True
ogpdrogpgkl = False
corx_option = 0
drop_threshold = 0.3 / 200
ML_time_series_path = ''
q_prior_from_ML = False
ML_priors_force_quit = True
run_dkfkf = True
run_fzekje = True
day_length = 1000000.0
test = 'normal'
if test == 'stations':
    shape03 = 100.0
    station_period_only = True
    activate_facc = False
    use_fallback_sos = True
    fallback_sos_dir = '/mnt/DATA/worksync/Nouveau dossier/sic/Igor_run_compa/input/sos/unconstrained/v16d/'
    use_dynamic_spread = False
    use_mean_for_bounds = True
elif test == 'ML':
    shape03 = 30.0
    station_period_only = True
    activate_facc = False
    use_fallback_sos = False
    fallback_sos_dir = ''
    use_dynamic_spread = False
    use_mean_for_bounds = True
    q_prior_from_ML = True
else:
    shape03 = 30.0
    station_period_only = False
    activate_facc = True
    use_fallback_sos = False
    fallback_sos_dir = ''
    use_dynamic_spread = False
    use_mean_for_bounds = True
    q_prior_from_ML = False
eps1 = 0.01
eps2 = 0.0001
force_create_reach_t = True
run_algo31_v3 = False
def_float_atol = 0.01
pankaj_test = False
V32 = False
sigmaZ = 1.0
thres = 1.0
DH_Iter = 10
DH_Esp = 0.0001
useEXT = True