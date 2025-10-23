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
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import scipy
import logging
import numpy as np
from copy import deepcopy
from pathlib import Path
import sic4dvar_params as params
from sic4dvar_functions.sic4dvar_helper_functions import gnuplot_save_c1c2, global_large_deviations_removal, compute_mean_var_from_2D_array_sum, gnuplot_save, compute_mean_var_from_2D_array, grad_variance
from sic4dvar_functions.sic4dvar_calculations import verify_name_length, check_na
from sic4dvar_functions.S841 import K as v
from sic4dvar_functions.R186 import W, X

def Extrapolation(sic4dvar_dict, params):
    sic4dvar_dict['input_data']['node_t_ini'] = deepcopy(sic4dvar_dict['input_data']['node_t'])
    sic4dvar_dict['input_data']['node_z_ini'] = deepcopy(sic4dvar_dict['input_data']['node_z'])
    sic4dvar_dict['input_data']['node_w_ini'] = deepcopy(sic4dvar_dict['input_data']['node_w'])
    node_x = sic4dvar_dict['input_data']['node_x']
    if params.corx_option == 0:
        corx = 200.0
    elif params.corx_option == 1:
        corx = np.mean(np.diff(node_x))
    elif params.corx_option == 2:
        corx = 200.0
    corx_array = np.ones(len(node_x)) * corx
    test_swot_z_obs = sic4dvar_dict['input_data']['node_z']
    test_swot_w_obs = sic4dvar_dict['input_data']['node_w']
    test_x_array = node_x
    if sic4dvar_dict['param_dict']['run_type'] == 'seq':
        test_t_array = sic4dvar_dict['input_data']['reach_t']
        if params.force_create_reach_t:
            test_t_array = np.ones(len(sic4dvar_dict['input_data']['node_t'][0]))
            for t in range(0, len(sic4dvar_dict['input_data']['node_t'][0])):
                test_t_array[t] = np.mean(sic4dvar_dict['input_data']['node_t'][:, t])
            sic4dvar_dict['input_data']['reach_t'] = test_t_array
    if sic4dvar_dict['param_dict']['run_type'] == 'set':
        test_t_array = np.ones(len(sic4dvar_dict['input_data']['node_t'][0]))
        for t in range(0, len(sic4dvar_dict['input_data']['node_t'][0])):
            test_t_array[t] = np.mean(sic4dvar_dict['input_data']['node_t'][:, t])
        sic4dvar_dict['input_data']['separate_reach_t'] = deepcopy(sic4dvar_dict['input_data']['reach_t'])
        sic4dvar_dict['input_data']['reach_t'] = test_t_array
    cort = params.cort
    cort_width = cort
    if params.override_cort:
        cort_tmp = []
        for t in range(0, len(test_t_array)):
            if not check_na(test_t_array[t]):
                cort_tmp.append(test_t_array[t])
        cort = np.mean(np.diff(cort_tmp))
    cort_wse = cort
    sic4dvar_dict['cort_wse'] = cort_wse
    sections_plot = False
    sic4dvar_dict['tmp_interp_values'] = []
    sic4dvar_dict['tmp_interp_values_w'] = []
    arr = sic4dvar_dict['input_data']['node_z_ini']
    arr[arr < -100000] = np.nan
    sic4dvar_dict['intermediates_values']['z0'] = test_swot_z_obs.filled(np.nan)
    sic4dvar_dict['intermediates_values']['w0'] = test_swot_w_obs.filled(np.nan)
    if sic4dvar_dict['param_dict']['gnuplot_saving']:
        reach_id = str(sic4dvar_dict['input_data']['reach_id'])
        reach_id = verify_name_length(reach_id)
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id), 'before_devia')
        gnuplot_save(nodes2, times2, test_swot_z_obs, sic4dvar_dict['input_data']['node_w_ini'], output_path, np.min(test_swot_z_obs), 2)
    if params.large_deviations:
        sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
        test_swot_z_obs, c1, c2 = global_large_deviations_removal(sic4dvar_dict['input_data']['node_x'], sic4dvar_dict['input_data']['node_z_ini'])
        nodes2 = (node_x - node_x[0]) / 1000
        times2 = test_t_array / 3600 / 24
        if sic4dvar_dict['param_dict']['gnuplot_saving']:
            reach_id = str(sic4dvar_dict['input_data']['reach_id'])
            reach_id = verify_name_length(reach_id)
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
            if not Path(output_path).is_dir():
                Path(output_path).mkdir(parents=True, exist_ok=True)
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id), 'c1c2')
            gnuplot_save_c1c2(nodes2, c1, c2, times2, output_path)
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id), 'after_devia')
            gnuplot_save(nodes2, times2, test_swot_z_obs, sic4dvar_dict['input_data']['node_w_ini'], output_path, np.min(test_swot_z_obs), 2)
    sic4dvar_dict['intermediates_values']['z1'] = test_swot_z_obs.filled(np.nan)
    sic4dvar_dict['intermediates_values']['w1'] = sic4dvar_dict['input_data']['node_w'].filled(np.nan)
    if params.run_dkfkf:
        test_swot_z_obs = v(dim=1, value0_array=test_swot_z_obs, base0_array=test_x_array, max_iter=1, cor=corx_array, always_run_first_iter=True, behaviour='', inter_behaviour=False, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['intermediates_values']['z2'] = test_swot_z_obs.filled(np.nan)
    times = np.arange(0, len(sic4dvar_dict['input_data']['node_z'][0]))
    nodes = np.arange(0, len(sic4dvar_dict['input_data']['node_z']))
    times2 = np.around(test_t_array / 3600 / 24)
    times2 = times2 - min(times2)
    nodes2 = (node_x - node_x[0]) / 1000
    reach_id = str(sic4dvar_dict['input_data']['reach_id'])
    reach_id = verify_name_length(reach_id)
    if sic4dvar_dict['param_dict']['gnuplot_saving']:
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
    sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
    if params.run_fzekje:
        test_swot_z_obs = W(values0_array=test_swot_z_obs, space0_array=test_x_array, dx_max_in=params.DX_max_in, dx_max_out=params.DX_max_out, dw_min=0.01, clean_run=False, debug_mode=False)
    sic4dvar_dict['intermediates_values']['z3'] = test_swot_z_obs.filled(np.nan)
    sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
    if params.run_dkfkf:
        for i in range(0, 1):
            test_swot_z_obs = v(dim=1, value0_array=test_swot_z_obs, base0_array=test_x_array, max_iter=10, cor=corx_array, always_run_first_iter=True, behaviour='decrease', inter_behaviour=True, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='force', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
    if params.run_dkfkf:
        test_swot_z_obs = v(dim=0, value0_array=test_swot_z_obs, base0_array=test_t_array, max_iter=1, cor=cort_wse, always_run_first_iter=False, behaviour='', inter_behaviour=False, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['intermediates_values']['z4'] = test_swot_z_obs.filled(np.nan)
    sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs)
    reach_t = sic4dvar_dict['input_data']['reach_t']
    if params.run_dkfkf:
        test_swot_w_obs = v(dim=1, value0_array=test_swot_w_obs, base0_array=test_x_array, max_iter=1, cor=corx, always_run_first_iter=True, behaviour='', inter_behaviour=False, check_behaviour='', min_change_v_thr=0.01, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['intermediates_values']['w2'] = test_swot_w_obs.filled(np.nan)
    sic4dvar_dict['tmp_interp_values_w'].append(test_swot_w_obs)
    if params.run_fzekje:
        test_swot_w_obs = X(values0_array=test_swot_w_obs, space0_array=test_x_array, weight0_array=test_swot_z_obs, dx_max_in=params.DX_max_in, dx_max_out=params.DX_max_out, dw_min=0.1, clean_run=False, debug_mode=False)
    sic4dvar_dict['intermediates_values']['w3'] = test_swot_w_obs.filled(np.nan)
    sic4dvar_dict['tmp_interp_values_w'].append(test_swot_w_obs)
    if params.run_dkfkf:
        test_swot_w_obs = v(dim=1, value0_array=test_swot_w_obs, base0_array=test_x_array, max_iter=10, cor=corx, always_run_first_iter=True, behaviour='', inter_behaviour=False, check_behaviour='', min_change_v_thr=0.01, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['tmp_interp_values_w'].append(test_swot_w_obs)
    if params.run_dkfkf:
        test_swot_w_obs = v(dim=0, value0_array=test_swot_w_obs, base0_array=test_t_array, max_iter=1, cor=cort_wse, always_run_first_iter=True, behaviour='', inter_behaviour=False, check_behaviour='', min_change_v_thr=0.01, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=False)
    sic4dvar_dict['intermediates_values']['w4'] = test_swot_w_obs.filled(np.nan)
    sic4dvar_dict['tmp_interp_values_w'].append(test_swot_w_obs)
    sic4dvar_dict['input_data']['node_z'] = test_swot_z_obs.filled(np.nan)
    sic4dvar_dict['input_data']['node_w'] = test_swot_w_obs.filled(np.nan)
    return sic4dvar_dict