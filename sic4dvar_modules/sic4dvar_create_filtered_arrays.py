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
import sic4dvar_params as params
from sic4dvar_modules.sic4dvar_filtering import remove_unuseable_nodes

def find_array_with_most_valid_values(arrays):
    max_valid_count = -1
    max_valid_index = -1
    for i, arr in enumerate(arrays):
        valid_count = np.sum(~np.isnan(arr))
        if valid_count > max_valid_count:
            max_valid_count = valid_count
            max_valid_index = i
    return max_valid_index

def create_filtered_arrays(sic4dvar_dict):
    if sic4dvar_dict['param_dict']['use_reach_slope']:
        data_is_useable_sl, observed_nodes_sl, list_to_keep_sl, removed_indices_sl = remove_unuseable_nodes(np.array([sic4dvar_dict['input_data']['reach_s']]), np.array([sic4dvar_dict['input_data']['reach_s']]))
        if data_is_useable_sl:
            list_to_keep_intersection = np.intersect1d(sic4dvar_dict['list_to_keep'], list_to_keep_sl)
            total_array = np.arange(len(sic4dvar_dict['input_data']['node_z'][0]))
            removed_indices_intersection = np.setdiff1d(total_array, list_to_keep_intersection)
            if np.array(list_to_keep_intersection).size > 0:
                sic4dvar_dict['list_to_keep'] = list_to_keep_intersection
                sic4dvar_dict['removed_indices'] = removed_indices_intersection
            else:
                sic4dvar_dict['data_is_useable'] = False
    sic4dvar_dict['filtered_data']['node_w'] = sic4dvar_dict['input_data']['node_w'][:, sic4dvar_dict['list_to_keep']][sic4dvar_dict['observed_nodes']]
    sic4dvar_dict['filtered_data']['node_z'] = sic4dvar_dict['input_data']['node_z'][:, sic4dvar_dict['list_to_keep']][sic4dvar_dict['observed_nodes']]
    sic4dvar_dict['filtered_data']['node_t'] = sic4dvar_dict['input_data']['node_t'][:, sic4dvar_dict['list_to_keep']][sic4dvar_dict['observed_nodes']]
    keys = ['reach_w', 'reach_z', 'reach_s', 'reach_dA']
    for key in keys:
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            sic4dvar_dict['filtered_data'][key] = []
            for i in range(0, len(sic4dvar_dict['input_data'][key])):
                sic4dvar_dict['filtered_data'][key].append(sic4dvar_dict['input_data'][key][i][sic4dvar_dict['list_to_keep']])
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            sic4dvar_dict['filtered_data'][key] = []
            sic4dvar_dict['filtered_data'][key] = sic4dvar_dict['input_data'][key][sic4dvar_dict['list_to_keep']]
    sic4dvar_dict['filtered_data']['reach_t'] = sic4dvar_dict['input_data']['reach_t'][sic4dvar_dict['list_to_keep']]
    sic4dvar_dict['filtered_data']['n_obs'] = np.ones(sic4dvar_dict['filtered_data']['node_w'].shape[0], dtype=int) * sic4dvar_dict['filtered_data']['node_w'].shape[1]
    if not params.node_length:
        sic4dvar_dict['filtered_data']['node_x'] = sic4dvar_dict['input_data']['node_x'][sic4dvar_dict['observed_nodes']]
    elif params.node_length:
        sic4dvar_dict['filtered_data']['node_x'] = sic4dvar_dict['input_data']['node_x'][sic4dvar_dict['observed_nodes']]
    index = find_array_with_most_valid_values(sic4dvar_dict['filtered_data']['node_t'][:])
    sic4dvar_dict['filtered_data']['orig_time'] = sic4dvar_dict['filtered_data']['node_t'][0]
    sic4dvar_dict['output']['time'] = sic4dvar_dict['input_data']['reach_t'] / 86400
    return sic4dvar_dict