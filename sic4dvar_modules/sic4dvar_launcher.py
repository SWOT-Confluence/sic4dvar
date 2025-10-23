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
from pathlib import Path
import sic4dvar_params as params
from sic4dvar_modules.sic4dvar_filtering import filter_based_on_config, remove_unuseable_nodes
from sic4dvar_modules.sic4dvar_extrapolation import Extrapolation
from sic4dvar_modules.sic4dvar_create_filtered_arrays import create_filtered_arrays
from sic4dvar_modules.sic4dvar_qwbm_replace import replace_prior
from sic4dvar_modules.sic4dvar_compute_slope_and_bathymetry import compute_slope, compute_bathymetry
from sic4dvar_modules.sic4dvar_launch_algo5 import launch_algo5
from sic4dvar_algos.algo3 import algo3
from sic4dvar_algos.algo5 import algo5
from sic4dvar_functions.sic4dvar_helper_functions import build_output_q_masked_array, compute_mean_var_from_2D_array, grad_variance, gnuplot_save_var, gnuplot_save_cs
from lib.lib_verif import verify_name_length

def sic4dvar_preprocessing(sic4dvar_dict, params, reach_number=0):
    logging.info('Runs launch_sic4dvar over %d reaches' % reach_number)
    sic4dvar_dict['data_is_useable'] = True
    if sic4dvar_dict['param_dict']['run_type'] == 'set':
        for i in range(0, reach_number):
            sic4dvar_dict['output']['valid_a5_sets'].append(1)
    sic4dvar_dict = filter_based_on_config(sic4dvar_dict)
    logging.info('Finished filtering the data.')
    if len(sic4dvar_dict['input_data']['node_x']) == 0:
        sic4dvar_dict['data_is_useable'] = False
        logging.warning('Empty SWORD node data.')
    if len(sic4dvar_dict['input_data']['node_x']) != len(sic4dvar_dict['input_data']['node_z']):
        sic4dvar_dict['data_is_useable'] = False
        logging.warning('Number of nodes is different between SWORD and SWOT file !')
    logging.info('Finished checking nodes not empty + same size as in SWORD.')
    if params.extrapolation and sic4dvar_dict['data_is_useable']:
        logging.info('Extrapolating wse data')
        sic4dvar_dict = Extrapolation(sic4dvar_dict, params)
        valid_z = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_z']))
        valid_w = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_w']))
        logging.info(f'After extrapolation: {valid_z} valid wse and {valid_w} valid width')
    logging.info('Finished extrapolating the data.')
    if sic4dvar_dict['data_is_useable']:
        sic4dvar_dict['data_is_useable'], sic4dvar_dict['observed_nodes'], sic4dvar_dict['list_to_keep'], sic4dvar_dict['removed_indices'] = remove_unuseable_nodes(sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['node_w'])
        sic4dvar_dict['output']['observed_nodes'] = sic4dvar_dict['observed_nodes']
    if sic4dvar_dict['data_is_useable'] and np.array(sic4dvar_dict['list_to_keep']).size > 0:
        sic4dvar_dict = create_filtered_arrays(sic4dvar_dict)
    logging.info('Finished creating filtered data with useable nodes.')
    return sic4dvar_dict

def sic4dvar_set_prior(sic4dvar_dict):
    logging.info(f'Qwbm value: {sic4dvar_dict['input_data']['reach_qwbm']}')
    sic4dvar_dict, flag_qwbm = replace_prior(sic4dvar_dict)
    return (sic4dvar_dict, flag_qwbm)

def sic4dvar_compute_discharge(sic4dvar_dict, params, flag_qwbm):
    sic4dvar_dict, flag_qwbm = sic4dvar_set_prior(sic4dvar_dict)
    if sic4dvar_dict['data_is_useable'] and flag_qwbm:
        logging.info('INFO: preparing to estimate discharge.')
        if params.densification:
            pass
        if not params.densification:
            SLOPEM1, sic4dvar_dict = compute_slope(sic4dvar_dict, params)
            sic4dvar_dict, bathymetry_array, apr_array, last_time_instant, time_indexes_to_keep = compute_bathymetry(sic4dvar_dict, params, SLOPEM1)
            if sic4dvar_dict['param_dict']['gnuplot_saving']:
                nodes2 = (sic4dvar_dict['input_data']['node_x'] - sic4dvar_dict['input_data']['node_x'][0]) / 1000
                times2 = sic4dvar_dict['input_data']['reach_t'] / 3600 / 24
                reach_id = str(sic4dvar_dict['input_data']['reach_id'])
                reach_id = verify_name_length(reach_id)
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
            if sic4dvar_dict['output']['valid']:
                logging.info(f'Computed bathymetry.')
            else:
                return sic4dvar_dict
            logging.info(f'Lauching discharge estimation.')
            sic4dvar_dict['list_to_keep'] = sic4dvar_dict['list_to_keep'][time_indexes_to_keep]
            apr_array['node_w_simp'] = apr_array['node_w_simp'][:, time_indexes_to_keep]
            apr_array['node_a'] = apr_array['node_a'][:, time_indexes_to_keep]
            apr_array['node_p'] = apr_array['node_p'][:, time_indexes_to_keep]
            apr_array['node_r'] = apr_array['node_r'][:, time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['node_z'] = sic4dvar_dict['filtered_data']['node_z'][:, time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['reach_t'] = sic4dvar_dict['filtered_data']['reach_t'][time_indexes_to_keep]
            sic4dvar_dict['output']['q_algo31'], sic4dvar_dict['output']['valid'], sic4dvar_dict['reliability'] = algo3(apr_array, sic4dvar_dict['input_data']['reach_qwbm'], params, SLOPEM1, sic4dvar_dict['output']['valid'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['node_z_ini'], sic4dvar_dict['filtered_data']['node_x'], sic4dvar_dict['last_node_for_integral'], bathymetry_array, sic4dvar_dict['param_dict'], sic4dvar_dict['input_data']['reach_id'], sic4dvar_dict['filtered_data']['reach_t'], bb=sic4dvar_dict['bb'], reliability=sic4dvar_dict['reliability'], Qsdev=sic4dvar_dict['input_data']['reach_qsdev'], last_time_instant=last_time_instant, input_data=sic4dvar_dict['input_data'], time_indexes_to_keep=time_indexes_to_keep)
            logging.info(f'Finished discharge estimation.')
            sic4dvar_dict['intermediates_values']['q_da'] = sic4dvar_dict['output']['q_algo31']
        Q31 = sic4dvar_dict['output']['q_algo31']
        if np.array(Q31).size == 0:
            sic4dvar_dict['output']['valid'] = 0
            return sic4dvar_dict
        sic4dvar_dict['output']['q_algo31'] = build_output_q_masked_array(sic4dvar_dict, 'q_algo31')
        sic4dvar_dict['output']['time'] = build_output_q_masked_array(sic4dvar_dict, 'time')
        sic4dvar_dict['output']['width'] = sic4dvar_dict['input_data']['node_xr']
        sic4dvar_dict['output']['elevation'] = sic4dvar_dict['input_data']['node_yr']
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            NBR_REACHES = len(sic4dvar_dict['filtered_data']['reach_w'])
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            NBR_REACHES = 1
        if sic4dvar_dict['param_dict']['run_algo5']:
            sic4dvar_dict = launch_algo5(sic4dvar_dict, NBR_REACHES)
            logging.info('Finished running Algo5.')
    else:
        logging.warning('Not enough consecutive data available to process reach.')
        sic4dvar_dict['output']['valid'] = 0
    if sic4dvar_dict['param_dict']['write_intermediate_products'] == True:
        if isinstance(sic4dvar_dict['input_data']['reach_id'], list):
            reach_id_pattern = str(sic4dvar_dict['input_data']['reach_id'][0]) + '_' + str(sic4dvar_dict['input_data']['reach_id'][-1])
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath(f'{reach_id_pattern}_sic4dvar_intermediates_values')
        else:
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath(f'{sic4dvar_dict['input_data']['reach_id']}_sic4dvar_intermediates_values')
        np.savez(output_path, **sic4dvar_dict['intermediates_values'])
    return sic4dvar_dict