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

"""
from pathlib import Path
import numpy as np
import multiprocessing
from datetime import datetime
algo_bounds = [[5.0], [-20.05, -0.05, -0.25], [10.0, 60.0, 5.0]]
algo_bounds[2][2] = (algo_bounds[2][1] - algo_bounds[2][0]) / 8.0
slope_smoothing_num_passes = 0
approx_section_params = [8, 0.15, 2]
kDim = 101
shape03 = 10.0
val1 = (algo_bounds[1][0] + algo_bounds[1][1]) / 2.0
shape13 = 2.1
val2 = (algo_bounds[2][0] + algo_bounds[2][1]) / 2.0
shape23 = 2.1
local_QM1 = 1.0
cython_version = False
min_num_nodes = 1
reachNB = 0
extrapolation = True
Option = 2
LSMX = 10
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
a31_early_stop = False
force_specific_dates = False
start_date = datetime(2023, 7, 11)
end_date = datetime(2023, 12, 31)
replace_config = True
config_file_path = '/app/sic4dvar_param_confluence.ini'
densification = False
num_cores = multiprocessing.cpu_count()
max_cores = int(num_cores / 2)
no_print = False
qsdev_activate = False
qsdev_option = 1
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
