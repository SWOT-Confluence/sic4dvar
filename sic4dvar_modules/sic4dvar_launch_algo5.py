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
import sic4dvar_params as params
from sic4dvar_algos.algo5 import *
from sic4dvar_algos.algo3 import *
from sic4dvar_functions.sic4dvar_helper_functions import get_weighted_q_data, correlation_nodes, build_output_q_masked_array
from sic4dvar_functions.sic4dvar_calculations import compute_bb, verify_name_length
from sic4dvar_modules.sic4dvar_compute_slope_and_bathymetry import bathymetry_computation, call_func_APR, slope_computation
from lib.lib_swot_obs import get_flag_dict_from_config, filter_swot_obs_on_quality
from lib.lib_dates import get_swot_dates, seconds_to_date_old

def check_reach_data_SET(sic4dvar_dict, iR=0):
    sic4dvar_dict['output']['q_algo31_masked'] = deepcopy(sic4dvar_dict['output']['q_algo31'][sic4dvar_dict['list_to_keep']])
    reach = dict()
    if sic4dvar_dict['param_dict']['run_type'] == 'seq':
        reach['dA'] = deepcopy([sic4dvar_dict['input_data']['reach_dA']])
        reach['w'] = deepcopy([sic4dvar_dict['input_data']['reach_w']])
        reach['s'] = deepcopy([sic4dvar_dict['input_data']['reach_s']])
        reach['z'] = deepcopy([sic4dvar_dict['input_data']['reach_z']])
    if sic4dvar_dict['param_dict']['run_type'] == 'set':
        reach['dA'] = deepcopy(sic4dvar_dict['filtered_data']['reach_dA'])
        reach['w'] = deepcopy(sic4dvar_dict['filtered_data']['reach_w'])
        reach['s'] = deepcopy(sic4dvar_dict['filtered_data']['reach_s'])
        reach['z'] = deepcopy(sic4dvar_dict['filtered_data']['reach_z'])

    def return_seq_filtered_data(dA, w, s):
        return (dA, w, s)

    def output(dA, w, s):
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            sic4dvar_dict['filtered_data']['reach_dA'], sic4dvar_dict['filtered_data']['reach_w'], sic4dvar_dict['filtered_data']['reach_s'] = return_seq_filtered_data(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
            sic4dvar_dict['filtered_data']['reach_dA'], sic4dvar_dict['filtered_data']['reach_w'], sic4dvar_dict['filtered_data']['reach_s'] = return_seq_filtered_data(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            sic4dvar_dict['filtered_data']['reach_dA'][iR], sic4dvar_dict['filtered_data']['reach_w'][iR], sic4dvar_dict['filtered_data']['reach_s'][iR] = return_seq_filtered_data(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
    if sic4dvar_dict['output']['q_algo31'].mask.all() or reach['dA'][iR].mask.all() or reach['w'][iR].mask.all() or reach['s'][iR].mask.all():
        return False
    elif sic4dvar_dict['output']['q_algo31'].mask.any() or reach['dA'][iR].mask.any() or reach['w'][iR].mask.any() or reach['s'][iR].mask.any():
        indices_to_mask = []
        keys = ['q_algo31', 'dA', 'w', 's']
        for key in keys:
            print('KEY:', key)
            if key == 'q_algo31':
                data_mask = []
                for i in range(0, len(sic4dvar_dict['output'][key])):
                    data_mask.append(calc.check_na(sic4dvar_dict['output'][key][i]))
            else:
                data_mask = []
                for i in range(0, len(reach[key][iR])):
                    data_mask.append(calc.check_na(reach[key][iR][i]))
            index_to_mask = np.where(np.array(data_mask) == True)
            indices_to_mask.append(index_to_mask)
        indices_to_mask = np.concatenate([arr[0] for arr in indices_to_mask])
        indices_to_mask = np.unique(indices_to_mask)
        sic4dvar_dict['removed_indices'] = indices_to_mask
        if indices_to_mask.size > 0:
            mask = np.ones(len(sic4dvar_dict['output']['q_algo31']), dtype=bool)
            print('mask:', mask)
            mask[indices_to_mask] = False
            sic4dvar_dict['output']['q_algo31_masked'] = sic4dvar_dict['output']['q_algo31'][mask]
            reach['dA'][iR] = reach['dA'][iR][mask]
            reach['w'][iR] = reach['w'][iR][mask]
            reach['s'][iR] = reach['s'][iR][mask]
            keys = ['dA', 'w', 's', 'q_algo31_masked']
            for key in keys:
                full_nan = 0
                for t in range(0, len(sic4dvar_dict['output']['q_algo31_masked'])):
                    if not key == 'q_algo31_masked':
                        if calc.check_na(reach[key][iR][t]):
                            full_nan += 1
                    elif calc.check_na(sic4dvar_dict['output'][key][t]):
                        full_nan += 1
                if full_nan == len(sic4dvar_dict['output']['q_algo31_masked']):
                    output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
                    return False
        print('q_algo31_masked:', sic4dvar_dict['output']['q_algo31_masked'], len(sic4dvar_dict['output']['q_algo31_masked']))
        if len(sic4dvar_dict['output']['q_algo31_masked']) < sic4dvar_dict['param_dict']['min_obs']:
            logging.info('Not enough timestamps to process with algo5!')
            output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
            return False
    OUTLOG = []
    for ind in range(len(reach['w'][iR])):
        if not calc.check_suitability(0.1, abs(reach['dA'][iR].min()) * 1.5, reach['dA'][iR][ind], reach['w'][iR][ind], reach['s'][iR][ind]):
            OUTLOG += [True]
        else:
            OUTLOG += [False]
    if np.array(OUTLOG).all():
        logging.info('Input data into algo5 is not suitable for algorithm!')
        output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
        return False
    else:
        output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
        return True
    output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
    return True

def optional_dA_fill(sic4dvar_dict, input_reach_dA, filtered_reach_w, filtered_reach_z, params):
    dA_from_mike_method = False
    force_compute = False
    if input_reach_dA.mask.all() or force_compute:
        if not dA_from_mike_method:
            a5_xr, a5_yr, _ = bathymetry_computation([filtered_reach_w], [filtered_reach_z], sic4dvar_dict['param_dict'], params, input_data=sic4dvar_dict['input_data'], filtered_data=sic4dvar_dict['filtered_data'], slope=sic4dvar_dict['input_data']['reach_s'], algo='algo5')
            node_a, _, _, _, _ = call_func_APR([filtered_reach_w], [filtered_reach_z], a5_xr, a5_yr, params, sic4dvar_dict['param_dict'])
            masked_data = np.ma.masked_values(np.array(node_a[0]), value=-9999.0)
            filtered_reach_dA = masked_data
        elif dA_from_mike_method:
            a5_xr, a5_yr, node_a = bathymetry_computation([filtered_reach_w], [filtered_reach_z], sic4dvar_dict['param_dict'], params, input_data=sic4dvar_dict['input_data'], filtered_data=sic4dvar_dict['filtered_data'], slope=sic4dvar_dict['input_data']['reach_s'], force_method='Mike', algo='algo5')
            masked_data = np.ma.masked_values(np.array(node_a), value=-9999.0)
            filtered_reach_dA = masked_data
        return filtered_reach_dA
    elif not input_reach_dA.mask.all() and (not force_compute):
        return input_reach_dA

def launch_algo5(sic4dvar_dict, NBR_REACHES):
    for i in range(0, NBR_REACHES):
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            sic4dvar_dict['filtered_data']['reach_dA'] = optional_dA_fill(sic4dvar_dict, sic4dvar_dict['input_data']['reach_dA'], sic4dvar_dict['input_data']['reach_w'], sic4dvar_dict['input_data']['reach_z'], params)
            sic4dvar_dict['reach_check'] = check_reach_data_SET(sic4dvar_dict)
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            sic4dvar_dict['filtered_data']['reach_dA'][i] = optional_dA_fill(sic4dvar_dict, sic4dvar_dict['input_data']['reach_dA'][i], sic4dvar_dict['filtered_data']['reach_w'][i], sic4dvar_dict['filtered_data']['reach_z'][i], params)
            sic4dvar_dict['reach_check'] = check_reach_data_SET(sic4dvar_dict, i)
        if sic4dvar_dict['reach_check']:
            logging.info('Running algo5 at REACH level to estimate discharge, A0 & n.')
            if sic4dvar_dict['param_dict']['run_type'] == 'seq':
                t0_a5 = datetime.utcnow()
                sic4dvar_dict['output']['q_algo5'], sic4dvar_dict['algo5_results'] = algo5(sic4dvar_dict['output']['q_algo31_masked'], sic4dvar_dict['filtered_data']['reach_dA'], sic4dvar_dict['filtered_data']['reach_w'], sic4dvar_dict['filtered_data']['reach_s'], sic4dvar_dict['input_data']['reach_id'], equation='ManningLW')
                t1_a5 = datetime.utcnow()
                sic4dvar_dict['output']['q_algo5'] = algo5_fill_removed_data(sic4dvar_dict['output']['q_algo5'], sic4dvar_dict['removed_indices'], len(sic4dvar_dict['filtered_data']['node_z'][0]))
                sic4dvar_dict['output']['q_algo5'] = build_output_q_masked_array(sic4dvar_dict, 'q_algo5')
            if sic4dvar_dict['param_dict']['run_type'] == 'set':
                t0_a5 = datetime.utcnow()
                sic4dvar_dict['output']['q_algo5'], sic4dvar_dict['algo5_results'] = algo5(sic4dvar_dict['output']['q_algo31_masked'], sic4dvar_dict['filtered_data']['reach_dA'][i], sic4dvar_dict['filtered_data']['reach_w'][i], sic4dvar_dict['filtered_data']['reach_s'][i], sic4dvar_dict['input_data']['reach_id'][i], equation='ManningLW')
                t1_a5 = datetime.utcnow()
                sic4dvar_dict['output']['q_algo5'] = algo5_fill_removed_data(sic4dvar_dict['output']['q_algo5'], sic4dvar_dict['removed_indices'], len(sic4dvar_dict['filtered_data']['node_z'][0]))
                sic4dvar_dict['output']['q_algo5'] = build_output_q_masked_array(sic4dvar_dict, 'q_algo5')
                sic4dvar_dict['output']['q_algo5_all'] += [sic4dvar_dict['output']['q_algo5']]
            sic4dvar_dict['intermediates_values']['q_mm'] = sic4dvar_dict['output']['q_algo5']
        else:
            sic4dvar_dict['output']['q_algo31'] = build_output_q_masked_array(sic4dvar_dict, 'q_algo31')
            logging.warning('Not enough REACH data found to process reach.')
            if sic4dvar_dict['param_dict']['run_type'] == 'seq':
                sic4dvar_dict['output']['valid_a5'] = 0
            if sic4dvar_dict['param_dict']['run_type'] == 'set':
                sic4dvar_dict['output']['valid_a5_sets'][i] = 0
    return sic4dvar_dict