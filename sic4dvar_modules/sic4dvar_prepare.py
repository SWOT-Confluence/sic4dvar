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
import numpy as np
import logging
import datetime
import sic4dvar_params as params
import sic4dvar_functions.sic4dvar_calculations as calc
from lib.lib_dates import seconds_to_date_old

def prepare_params(input_data, flag_dict, param_dict, params):
    sic4dvar_dict = {'input_data': input_data, 'flag_dict': flag_dict, 'param_dict': param_dict, 'filtered_data': {}, 'output': {}, 'intermediates_values': {}, 'algo5_results': {}, 'bb': 9999.0, 'reliability': 'valid'}
    sic4dvar_dict['output']['valid'] = 1
    sic4dvar_dict['output']['valid_a5'] = 1
    sic4dvar_dict['output']['valid_a5_sets'] = []
    sic4dvar_dict['output']['q_std'] = input_data['reach_qsdev']
    sic4dvar_dict["output"]['node_id'] = input_data['node_id']
    if params.force_specific_dates:
        print(sic4dvar_dict['input_data']['reach_t'])
        dates = []
        for t in range(0, len(sic4dvar_dict['input_data']['reach_t'])):
            if calc.check_na(sic4dvar_dict['input_data']['reach_t'][t]):
                dates.append(datetime(1, 1, 1))
            else:
                dates.append(seconds_to_date_old(sic4dvar_dict['input_data']['reach_t'][t]))
        indexes = [i for i, date in enumerate(dates) if params.start_date <= date <= params.end_date]
        if np.array(indexes).size == 0:
            if sic4dvar_dict['param_dict']['run_type'] == 'seq':
                logging.info('All data removed by date filtering.')
                logging.info("Sequential run, can't process without SWOT data.")
                sic4dvar_dict['output']['valid'] = 0
                return None
        else:
            reach_keys = ['reach_dA', 'reach_s', 'reach_w', 'reach_z', 'reach_t']
            node_keys = ['node_w', 'node_z', 'node_dA', 'node_s', 'node_t']
            for key in reach_keys:
                sic4dvar_dict['input_data'][key] = sic4dvar_dict['input_data'][key][indexes]
            for key in node_keys:
                sic4dvar_dict['input_data'][key] = sic4dvar_dict['input_data'][key][:, indexes]
            for key in list(sic4dvar_dict['flag_dict']['node'].keys()):
                if not sic4dvar_dict['flag_dict']['node'][key].shape == (0,):
                    sic4dvar_dict['flag_dict']['node'][key] = sic4dvar_dict['flag_dict']['node'][key][:, indexes]
            for key in list(sic4dvar_dict['flag_dict']['reach'].keys()):
                if key != 'reach_width_min' and key != 'reach_slope_min' and (key != 'reach_length_min'):
                    sic4dvar_dict['flag_dict']['reach'][key] = sic4dvar_dict['flag_dict']['reach'][key][indexes]
            sic4dvar_dict['flag_dict']['nt'] = len(indexes)
            print(sic4dvar_dict['input_data']['node_z'].shape, sic4dvar_dict['input_data']['reach_z'].shape)
    if sic4dvar_dict['input_data'] == 0:
        raise IOError('Error reading inputs ... input_data is empty')
    if 'node_t' in sic4dvar_dict['input_data'] and 'reach_w' in sic4dvar_dict['input_data']:
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            mask = np.ones(sic4dvar_dict['input_data']['node_t'].shape[1], dtype=bool)
            arr = np.full(sic4dvar_dict['input_data']['node_t'].shape[1], np.nan)
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            mask = np.ones(len(sic4dvar_dict['input_data']['reach_w']), dtype=bool)
            arr = np.full(len(sic4dvar_dict['input_data']['reach_w']), np.nan)
        sic4dvar_dict['output']['q_algo5'] = np.ma.array(arr, mask=mask)
        sic4dvar_dict['output']['q_algo5_all'] = []
        sic4dvar_dict['output']['q_algo31'] = np.ma.array(arr, mask=mask)
        sic4dvar_dict['output']['a0'] = np.ma.array([np.nan], mask=[True])
        sic4dvar_dict['output']['n'] = np.ma.array([np.nan], mask=[True])
        sic4dvar_dict['output']['node_a0'] = []
        sic4dvar_dict['output']['node_n'] = []
        sic4dvar_dict['lim_node_xr'] = []
        sic4dvar_dict['output']['half_width'] = np.empty(len(sic4dvar_dict['input_data']['node_w']), dtype=object)
        sic4dvar_dict['output']['elevation'] = np.empty(len(sic4dvar_dict['input_data']['node_w']), dtype=object)
        sic4dvar_dict['output']['half_width'].fill(np.ma.array(arr, mask=mask))
        sic4dvar_dict['output']['elevation'].fill(np.ma.array(arr, mask=mask))
        sic4dvar_dict['output']['time'] = []
        sic4dvar_dict['removed_nodes_ind'] = np.zeros(0)
        sic4dvar_dict['indices'] = []
    elif sic4dvar_dict['param_dict']['run_type'] == 'seq':
        logging.info("ERROR: Sequential run, can't process without SWOT data.")
        return 0
    elif sic4dvar_dict['param_dict']['run_type'] == 'set':
        mask = []
        arr = []
        logging.info('INFO: Missing SWOT data. Set_run, processing.')
        sic4dvar_dict['output']['q_algo5'] = []
        sic4dvar_dict['output']['q_algo5_all'] = []
        sic4dvar_dict['output']['q_algo31'] = []
        sic4dvar_dict['output']['a0'] = []
        sic4dvar_dict['output']['n'] = []
        sic4dvar_dict['output']['node_a0'] = []
        sic4dvar_dict['output']['node_n'] = []
        sic4dvar_dict['lim_node_xr'] = []
        sic4dvar_dict['output']['half_width'] = []
        sic4dvar_dict['output']['elevation'] = []
        sic4dvar_dict['output']['time'] = []
        sic4dvar_dict['removed_nodes_ind'] = np.zeros(0)
        sic4dvar_dict['indices'] = []
    sic4dvar_dict['intermediates_values']['node_t'] = sic4dvar_dict['input_data']['node_t']
    sic4dvar_dict['intermediates_values']['reach_t'] = sic4dvar_dict['input_data']['reach_t']
    if not params.node_length:
        sic4dvar_dict['input_data']['node_x'] = sic4dvar_dict['input_data']['dist_out']
    if params.node_length:
        acc_node_len = np.cumsum(sic4dvar_dict['input_data']['node_length'])
        dist_out_0 = scipy.stats.mode(np.abs(acc_node_len - sic4dvar_dict['input_data']['dist_out']), axis=None).mode
        dist_out_1 = []
        dist_out_1.append(sic4dvar_dict['input_data']['dist_out'][0])
        for i in range(1, len(sic4dvar_dict['input_data']['node_length'])):
            dist_out_1.append(dist_out_1[i - 1] - sic4dvar_dict['input_data']['node_length'][i - 1])
        sic4dvar_dict['input_data']['node_x'] = np.array(dist_out_1)[::-1]
    return sic4dvar_dict