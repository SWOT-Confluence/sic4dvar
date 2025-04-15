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
import matplotlib.pyplot as plt
from sic4dvar_functions import sic4dvar_calculations as calc
import sic4dvar_params as params
from copy import deepcopy
import datetime
import mplcursors
import sys
import os
from pathlib import Path
import json
from netCDF4 import Dataset
import logging
import numpy as np
import copy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array, nan_array_to_masked_array, find_nearest, find_n_nearest, get_index_valid_data, get_mask_nan_across_arrays
from lib.lib_sos import get_station_q_and_qt
from lib.lib_netcdf import get_nc_variable_data
from lib.lib_dates import seconds_to_date_old, convert_to_YMD, count_days_in_months, date_to_seconds
from lib.lib_verif import reorder_ids_with_indices
from lib.lib_log import call_error_message

def read_discharge_from_npy(reach_id, path):
    try:
        q = np.load(path + str(reach_id) + '.npy')
    except:
        q = []
    return q

def algo5_fill_removed_data(computed_q, removed_indices, full_size):
    j = 0
    q_est = []
    for i in range(0, full_size):
        if i in removed_indices:
            q_est.append(np.nan)
        else:
            q_est.append(computed_q[j])
            j += 1
    mask_q_est = np.ones(len(q_est), dtype=bool)
    for i in range(len(q_est)):
        if not calc.check_na(q_est[i]):
            mask_q_est[i] = False
    q_est_masked = np.ma.array(q_est, mask=mask_q_est)
    return q_est_masked

def gnuplot_save(distance, times, elevation, width, location, zmin, spaces):
    all_data = []
    zmin2 = np.nanmin(elevation)
    file = open(location, 'w')
    for i in range(elevation.shape[1]):
        x = distance
        z = elevation[:, i]
        t = np.repeat(times[i], elevation[:, i].shape[0])
        w = width[:, i]
        for k in range(elevation.shape[0]):
            z[k] = z[k]
        j = 0
        for x1, z1, t1, w1 in zip(x, z, t, w):
            file.write('{} {} {} {}\n'.format(t1, x1, z1, w1))
            j += 1
            if j == elevation.shape[0]:
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_list(distance, times, elevation, width, location, zmin, spaces):
    all_data = []
    file = open(location, 'w')
    max_length = max((len(arr) for arr in elevation))
    for i in range(0, len(elevation)):
        elevation[i] = np.pad(elevation[i], (0, max_length - len(elevation[i])))
        width[i] = np.pad(width[i], (0, max_length - len(width[i])))
    elevation = np.array(elevation)
    width = np.array(width)
    times = np.arange(max_length)
    zmin2 = np.nanmin(elevation)
    for i in range(0, elevation.shape[1]):
        x = distance
        z = elevation[:, i]
        t = np.repeat(times[i], elevation[:, i].shape[0])
        w = width[:, i]
        j = 0
        for x1, z1, t1, w1 in zip(x, z, t, w):
            file.write('{} {} {} {}\n'.format(t1, x1, z1, w1))
            j += 1
            if j == elevation.shape[0]:
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_q(discharge, times, location):
    all_data = []
    file = open(location, 'w')
    x = discharge
    t = np.around(times)
    t = t - min(t)
    j = 0
    for x1, t1 in zip(x, t):
        file.write('{} {}\n'.format(t1, x1))
        j += 1
    file.close()

def gnuplot_save_cs(width_cs, wse_cs, nodes, times, location, spaces=2):
    all_data = []
    file = open(location, 'w')
    for i in range(0, len(times)):
        w = width_cs[i]
        z = wse_cs[i]
        t = np.repeat(times[i], len(width_cs[i]))
        x = nodes
        j = 0
        for x1, w1, z1 in zip(x, w, z):
            file.write('{} {} {}\n'.format(x1, w1, z1))
            j += 1
            if j == len(t):
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_q2(discharge, nodes, times, location, spaces=2):
    all_data = []
    file = open(location, 'w')
    for i in range(0, len(times)):
        q = discharge[i]
        t = np.repeat(times[i], len(discharge[i]))
        x = nodes
        j = 0
        for x1, q1, t1 in zip(x, q, t):
            file.write('{} {} {}\n'.format(t1, x1, q1))
            j += 1
            if j == len(t):
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def gnuplot_save_slope(slope, times, location):
    all_data = []
    file = open(location, 'w')
    x = slope
    t = np.around(times)
    t = t - min(t)
    j = 0
    for x1, t1 in zip(x, t):
        file.write('{} {}\n'.format(x1, t1))
        j += 1
    file.close()

def gnuplot_save_c1c2(node_x, c1, c2, times, location, spaces=2):
    all_data = []
    file = open(location, 'w')
    for i in range(len(times)):
        x = node_x
        z = c1 * node_x + c2
        t = np.around(times)
        t = t - min(t)
        t = np.repeat(t[i], node_x.shape[0])
        j = 0
        for x1, z1, t1 in zip(x, z, t):
            file.write('{} {} {} \n'.format(t1, x1, z1))
            j += 1
            if j == node_x.shape[0]:
                if spaces <= 1:
                    file.write('\n')
                if spaces >= 2:
                    file.write('\n\n')
    file.close()

def large_deviations_removal(node_x, z):
    new_z = deepcopy(z)
    for i in range(0, 1):
        for t in range(0, len(z[0])):
            index_valid = []
            for n in range(0, len(z)):
                if not calc.check_na(z[n, t]):
                    index_valid.append(n)
            num_valid_pts = len(index_valid)
            num_overpass = np.arange(0, len(z[0]))
            if num_valid_pts > 2:
                a11 = 0.0
                a12 = 0.0
                a21 = 0.0
                a22 = 0.0
                b1 = 0.0
                b2 = 0.0
                for n in range(0, num_valid_pts):
                    a11 = a11 + node_x[index_valid[n]] * node_x[index_valid[n]]
                    a12 = a12 + node_x[index_valid[n]]
                    a21 = a21 + node_x[index_valid[n]]
                    a22 = a22 + 1.0
                    b1 = b1 + node_x[index_valid[n]] * z[index_valid[n], t]
                    b2 = b2 + z[index_valid[n], t]
                c2 = (b1 * a21 - b2 * a11) / (a12 * a21 - a22 * a11)
                c1 = (b1 - a12 * c2) / a11
                c3 = 0.0
                for n in range(0, num_valid_pts):
                    dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
                    c3 = c3 + dist ** 2
                c3 = np.sqrt(c3 / num_valid_pts)
                for n in range(0, num_valid_pts):
                    dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
                    if abs(dist) > 2.0 * c3:
                        new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2
    return new_z

def global_large_deviations_removal(node_x, z):
    node_x = (node_x - node_x[0]) / 1000
    new_z = deepcopy(z)
    a11 = 0.0
    a12 = 0.0
    a21 = 0.0
    a22 = 0.0
    b1 = 0.0
    b2 = 0.0
    c1 = 0.0
    c2 = 0.0
    total_valid_points = 0.0
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        for n in range(0, num_valid_pts):
            a11 = a11 + node_x[index_valid[n]] * node_x[index_valid[n]]
            a12 = a12 + node_x[index_valid[n]]
            a21 = a21 + node_x[index_valid[n]]
            a22 = a22 + 1.0
            b1 = b1 + node_x[index_valid[n]] * z[index_valid[n], t]
            b2 = b2 + z[index_valid[n], t]
    if total_valid_points >= 1:
        c2 = (b1 * a21 - b2 * a11) / (a12 * a21 - a22 * a11)
        c1 = (b1 - a12 * c2) / a11
    i_pslope = 0
    total_valid_points = 0.0
    if c1 > 0:
        i_pslope = 1
        ss0 = 0.0
        ss1 = 0.0
        for t in range(0, len(z[0])):
            index_valid = []
            for n in range(0, len(z)):
                if not calc.check_na(z[n, t]):
                    index_valid.append(n)
            num_valid_pts = len(index_valid)
            total_valid_points += num_valid_pts
            num_overpass = np.arange(0, len(z[0]))
            for n in range(0, num_valid_pts):
                ss0 = ss0 + c1 * node_x[index_valid[n]] + c2
                ss1 = ss1 + 1.0
        ss0 = ss0 / ss1
        c1 = 0.0
        c2 = ss0
    c3 = 0.0
    ss1 = 0.0
    total_valid_points = 0.0
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        for n in range(0, num_valid_pts):
            dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
            c3 = c3 + dist ** 2
            ss1 = ss1 + 1.0
    if total_valid_points >= 1:
        c3 = np.sqrt(c3 / ss1)
    sigmp = 2.0
    sigmn = 2.0
    if i_pslope == 1:
        sigmp = 1.0
        sigmn = 1.0
    total_valid_points = 0.0
    for t in range(0, len(z[0])):
        index_valid = []
        for n in range(0, len(z)):
            if not calc.check_na(z[n, t]):
                index_valid.append(n)
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0, len(z[0]))
        for n in range(0, num_valid_pts):
            dist = c1 * node_x[index_valid[n]] + c2 - z[index_valid[n], t]
            if dist > 0:
                if abs(dist) > sigmp * c3:
                    new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2 - dist * (1.0 * c3) / abs(dist)
            elif abs(dist) > sigmn * c3:
                new_z[index_valid[n], t] = c1 * node_x[index_valid[n]] + c2 - dist * (1.0 * c3) / abs(dist)
    return (new_z, c1, c2)

def gnuplot_save_tables(x, y, z, location, spaces):
    all_data = []
    file = open(location, 'w')
    j = 0
    for x1, y1, z1 in zip(x, y, z):
        file.write('{} {} {}\n'.format(x1, y1, z1))
        j += 1
        if j == len(x):
            if spaces <= 1:
                file.write('\n')
            if spaces >= 2:
                file.write('\n\n')
    file.close()

def get_weighted_q_data(dates_swot, q_mean):
    tot_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    indices = np.where(np.array(dates_swot) != '')
    indices = indices[0]
    first_date = dates_swot[indices[0]]
    last_date = dates_swot[indices[-1]]
    if first_date == '' and last_date == '':
        return -1
    i = int(first_date[5:7]) - 1
    j = int(last_date[5:7]) - 1
    new_dates_swot = np.array(dates_swot)[indices]
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in new_dates_swot]
    days_in_months = {}
    for l in range(len(dates) - 1):
        days_in_months = count_days_in_months(dates[l], dates[l + 1], days_in_months)
    days_tot = np.full((1, 12), 0)[0]
    for (year, month), days in sorted(days_in_months.items()):
        days_tot[month - 1] = days
        logging.debug(f'{year}-{month:02d}: {days} days')
    if i == j and first_date[0:4] == last_date[0:4]:
        Qwbm = q_mean[i]
    else:
        Qwbm = 0
        for k in range(0, 12):
            mod = k % 11
            Qwbm += days_tot[mod] * q_mean[mod]
        time_start = date_to_seconds(first_date)
        time_end = date_to_seconds(last_date)
        delta_t = (time_end - time_start) / 86400
        Qwbm = Qwbm / delta_t
    return Qwbm

def interp_pdf_tables(n, u, x, y):
    istop = 0
    i = 0
    while i < n and istop == 0:
        i = i + 1
        if x[i] > u:
            x1 = x[i - 1]
            x2 = x[i]
            v = y[i - 1] + (y[i] - y[i - 1]) * (u - x1) / (x2 - x1)
            istop = 1
    if u <= x[0]:
        v = y[0]
    if u >= x[n]:
        v = y[n]
    return v

def disable_prints():
    sys.stdout = open(os.devnull, 'w')

def enable_prints():
    sys.stdout = sys.__stdout__

def get_reach_dataset(param_dict, index=None):
    reachjson = param_dict['json_path']
    if param_dict['aws']:
        if os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'):
            index = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'))
    with open(reachjson) as jsonfile:
        data = json.load(jsonfile)
    if index != None and param_dict['run_type'] == 'seq':
        out_data = data[index]
    elif index != None and param_dict['run_type'] == 'set':
        if isinstance(data[index], dict):
            out_data = [data[index]]
        else:
            out_data = data[index]
    else:
        out_data = data
    return out_data

def write_to_log(file_path, string):
    with open(file_path, 'a') as file:
        file.write(string + '\n')

def get_input_data(param_dict, _reach_dict):
    swot_file = param_dict['swot_dir'].joinpath(f'{_reach_dict['reach_id']}_SWOT.nc')
    swot_dict = {}
    swot_dataset = []
    sos_dataset = []
    sword_dataset = []
    swot_dict['reach_id'] = _reach_dict['reach_id']
    flag_dict = {}
    flag_dict['node'] = {}
    flag_dict['reach'] = {}
    sword_file = param_dict['sword_dir'].joinpath(_reach_dict['sword'])
    if sword_file.exists():
        sword_dataset = Dataset(sword_file)
        index_node_ids = np.array(np.where(get_nc_variable_data(sword_dataset, 'nodes/reach_id') == int(_reach_dict['reach_id'])))[0]
        sword_nodes_id = sword_dataset['nodes']['node_id'][index_node_ids]
        sword_nodes_id_reordered_index, sword_nodes_id_reordered = reorder_ids_with_indices(sword_nodes_id)
        index_node_ids = index_node_ids[sword_nodes_id_reordered_index]
        if not params.opt_sword_boost:
            swot_dict['node_length'] = get_nc_variable_data(sword_dataset, 'nodes/node_length')[index_node_ids]
            swot_dict['dist_out'] = get_nc_variable_data(sword_dataset, 'nodes/dist_out')[index_node_ids]
            if params.pankaj_test:
                middle = round(len(index_node_ids) / 2)
                node = index_node_ids[middle]
                swot_dict['dist_out'] = get_nc_variable_data(sword_dataset, 'nodes/dist_out')[node]
                swot_dict['node_length'] = get_nc_variable_data(sword_dataset, 'nodes/node_length')[node]
        elif params.opt_sword_boost:
            ref_index_rch = np.array(np.where(get_nc_variable_data(sword_dataset, 'reaches/reach_id') == int(_reach_dict['reach_id'])))[0]
            index_rch_node = np.load('index_rch_node.npy')
            i1 = int(index_rch_node[ref_index_rch, 0])
            i2 = int(index_rch_node[ref_index_rch, 1])
            list_nodes = np.arange(i1, i2 + 1)
            swot_dict['node_length'] = get_nc_variable_data(sword_dataset, 'nodes/node_length')[list_nodes]
            swot_dict['dist_out'] = get_nc_variable_data(sword_dataset, 'nodes/dist_out')[list_nodes]
        swot_dict['node_w_mean'] = get_nc_variable_data(sword_dataset, 'nodes/width')[index_node_ids]
        index_reach_sword = np.array(np.where(get_nc_variable_data(sword_dataset, 'reaches/reach_id') == int(swot_dict['reach_id'])))[0]
        swot_dict['reach_w_mean'] = get_nc_variable_data(sword_dataset, 'reaches/width')[index_reach_sword]
        swot_dict['reach_slope'] = get_nc_variable_data(sword_dataset, 'reaches/slope')[index_reach_sword]
        swot_dict['reach_length'] = get_nc_variable_data(sword_dataset, 'reaches/reach_length')[index_reach_sword]
        flag_dict['reach']['reach_width_min'] = swot_dict['reach_w_mean']
        flag_dict['reach']['reach_slope_min'] = swot_dict['reach_slope']
        flag_dict['reach']['reach_length_min'] = swot_dict['reach_length']
        swot_dict['facc'] = get_nc_variable_data(sword_dataset, 'reaches/facc')[index_reach_sword]
        sword_dataset.close()
    else:
        logging.error(call_error_message(504).format(reach_id=_reach_dict['reach_id'], sword_file=sword_file))
        return (0, {})
    if swot_file.exists():
        swot_dataset = Dataset(swot_file)
        swot_node_ids = swot_dataset['node']['node_id'][:]
        swot_nodes_id_reordered_index, swot_nodes_id_reordered = reorder_ids_with_indices(swot_node_ids)
        swot_node_ids = swot_node_ids[swot_nodes_id_reordered_index]
        check_na_vector = np.vectorize(calc.check_na)
        flag_dict['reach']['obs_frac_n'] = get_nc_variable_data(swot_dataset, 'reach/obs_frac_n')
        flag_dict['reach']['reach_q_b'] = get_nc_variable_data(swot_dataset, 'reach/reach_q_b')
        flag_dict['reach']['slope_r_u'] = get_nc_variable_data(swot_dataset, 'reach/slope_r_u')
        flag_dict['reach']['partial_f'] = get_nc_variable_data(swot_dataset, 'reach/partial_f')
        node_param_flags = ['ice_clim_f', 'xtrk_dist', 'dark_frac', 'node_dist', 'layovr_val', 'wse_u', 'width_u', 'flow_angle', 'node_q', 'node_q_b', 'xovr_cal_q', 'wse_r_u']
        for attribute in node_param_flags:
            if get_nc_variable_data(swot_dataset, f'node/{attribute}').shape != (0,):
                flag_dict['node'][attribute] = get_nc_variable_data(swot_dataset, f'node/{attribute}')[swot_nodes_id_reordered_index]
        flag_dict['nx'] = get_nc_variable_data(swot_dataset, 'node/wse').shape[0]
        flag_dict['nt'] = get_nc_variable_data(swot_dataset, 'node/wse').shape[1]
        logging.info(f'Loading file {swot_file} - nx : {flag_dict['nx']} nt {flag_dict['nt']}')
        if hasattr(swot_dataset['reach/wse'], 'valid_min'):
            swot_dict['valid_min_z'] = swot_dataset['reach/wse'].valid_min
        else:
            swot_dict['valid_min_z'] = params.valid_min_z
        if hasattr(swot_dataset['reach/d_x_area'], 'valid_min'):
            swot_dict['valid_min_dA'] = swot_dataset['reach/d_x_area'].valid_min
        else:
            swot_dict['valid_min_dA'] = params.valid_min_dA
        swot_dict['reach_dA'] = get_nc_variable_data(swot_dataset, 'reach/d_x_area')
        swot_dict['reach_s'] = get_nc_variable_data(swot_dataset, 'reach/slope2')
        swot_dict['reach_w'] = get_nc_variable_data(swot_dataset, 'reach/width')
        swot_dict['reach_z'] = get_nc_variable_data(swot_dataset, 'reach/wse')
        swot_dict['reach_t'] = get_nc_variable_data(swot_dataset, 'reach/time')
        swot_dict['node_w'] = get_nc_variable_data(swot_dataset, 'node/width')[swot_nodes_id_reordered_index]
        swot_dict['node_z'] = get_nc_variable_data(swot_dataset, 'node/wse')[swot_nodes_id_reordered_index]
        swot_dict['node_dA'] = get_nc_variable_data(swot_dataset, 'node/d_x_area')[swot_nodes_id_reordered_index]
        swot_dict['node_s'] = get_nc_variable_data(swot_dataset, 'node/slope2')[swot_nodes_id_reordered_index]
        swot_dict['node_t'] = get_nc_variable_data(swot_dataset, 'node/time')[swot_nodes_id_reordered_index]
    else:
        logging.warning(call_error_message(505).format(reach_id=_reach_dict['reach_id'], swot_file=swot_file))
        if param_dict['run_type'] == 'seq':
            logging.error(call_error_message(103))
            return (0, {})
        elif param_dict['run_type'] == 'set':
            pass
    sos_file = param_dict['sos_dir'].joinpath(_reach_dict['sos'])
    if sos_file.exists():
        sos_dataset = Dataset(sos_file)
        sos_rids = get_nc_variable_data(sos_dataset, 'reaches/reach_id')
        index = np.where(sos_rids == _reach_dict['reach_id'])
        swot_dict['q_monthly_mean'] = get_nc_variable_data(sos_dataset, 'model/monthly_q')[index, :][0][0]
        swot_dict['reach_qwbm'] = get_nc_variable_data(sos_dataset, 'model/mean_q')[index]
        quant_mean, quant_var = calc.compute_mean_discharge_from_SoS_quantiles(sos_dataset['model']['flow_duration_q'][index[0], :][0])
        if params.qsdev_activate:
            masked_data = np.ma.masked_values(np.array([float(quant_mean)]), value=-9999.0)
            swot_dict['reach_qwbm'] = masked_data
        if params.qsdev_activate:
            if params.qsdev_option == 0:
                masked_data2 = np.ma.masked_values(np.array([float(quant_var)]), value=-9999.0)
            elif params.qsdev_option == 1:
                masked_data2 = np.ma.masked_values(np.array([float(quant_var / quant_mean)]), value=-9999.0)
            swot_dict['reach_qsdev'] = masked_data2
        elif not params.qsdev_activate:
            swot_dict['reach_qsdev'] = 0.0
        if calc.check_na(swot_dict['reach_qwbm']):
            pass
        if param_dict['override_q_prior']:
            masked_data = np.ma.masked_values(np.array([float(param_dict['q_prior_value'])]), value=-9999.0)
            swot_dict['reach_qwbm'] = masked_data
        sos_reach_id = get_nc_variable_data(sos_dataset, 'reaches/reach_id')
        index_reach_sos = np.array(np.where(sos_reach_id == int(swot_dict['reach_id'])))[0]
        swot_dict['station_q'], swot_dict['station_date'] = get_station_q_and_qt(sos_dataset, _reach_dict['reach_id'])
        if param_dict['q_prior_from_stations']:
            if np.array(swot_dict['station_q']).size == 0:
                logging.warning(call_error_message(501).format(reach_id=_reach_dict['reach_id']))
                return (0, {})
            q_mean, q_std = use_stations_for_q_prior(swot_dict['reach_t'], swot_dict['station_q'], swot_dict['station_date'])
            masked_data = np.ma.masked_values(np.array([q_mean]), value=-9999.0)
            swot_dict['reach_qwbm'] = masked_data
            if params.qsdev_activate:
                if params.qsdev_option == 0:
                    masked_data2 = np.ma.masked_values(np.array([float(q_std)]), value=-9999.0)
                elif params.qsdev_option == 1:
                    masked_data2 = np.ma.masked_values(np.array([float(q_std / q_mean)]), value=-9999.0)
                swot_dict['reach_qsdev'] = masked_data2
            if q_mean <= 1.0:
                logging.warning(call_error_message(506))
                return (0, {})
        sos_dataset.close()
    else:
        logging.error(call_error_message(104).format(reach_id=_reach_dict['reach_id'], sos_file=sos_file))
        return (0, {})
    if swot_file.exists():
        swot_dataset.close()
    return (swot_dict, flag_dict)

def write_output(output_dir, param_dict, _reach_id, _output, reach_number=0, folder_name='', dim_t=0, algo5_results={}, bb='9999.0', reliability=''):
    run_flag = True
    if len(_output) <= 0:
        run_flag = False
    if run_flag:
        if param_dict['run_type'] == 'seq':
            _output['time_steps'] = np.arange(0, _output['q_algo5'].shape[0])
        if param_dict['run_type'] == 'set':
            if reach_number < 0:
                null_a5 = np.full((1, _output['q_algo31'].shape[0]), np.nan)[0]
                _output['time_steps'] = np.arange(0, null_a5.shape[0])
            elif np.array(_output['q_algo5_all']).size > 0:
                _output['time_steps'] = np.arange(0, _output['q_algo5_all'][reach_number].shape[0])
            else:
                null_a5 = np.full((1, _output['q_algo31'].shape[0]), np.nan)[0]
                _output['time_steps'] = np.arange(0, null_a5.shape[0])
    if not run_flag:
        _output = {}
        _output['time_steps'] = np.arange(0, dim_t)
        _output['q_algo31'] = np.full((1, dim_t), np.nan)[0]
        null_a5 = np.full((1, dim_t), np.nan)[0]
        _output['time'] = np.full((1, dim_t), np.nan)[0]
        _output['valid'] = 0
    out_file = output_dir.joinpath(f'{_reach_id}_sic4dvar.nc')
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_nc = Dataset(out_file, 'w', format='NETCDF4')
    fill_value = -999999999999.0
    out_nc.createDimension('nt', _output['time_steps'].shape[0])
    if param_dict['run_type'] == 'seq':
        if np.array(_output['time']).size > 0:
            out_nc.createDimension('times', _output['time'].shape[0])
        else:
            out_nc.createDimension('times', _output['time_steps'].shape[0])
    if param_dict['run_type'] == 'set':
        out_nc.createDimension('times', _output['time_steps'].shape[0])
    nt = out_nc.createVariable('nt', 'i4', ('nt',))
    nt.units = 'time'
    nt[:] = range(_output['time_steps'].shape[0])
    times = out_nc.createVariable('times', 'f8', ('times',), fill_value=fill_value)
    times.units = 'days since 1st of January 2000'
    if np.array(_output['time']).size > 0:
        times[:] = _output['time']
    out_nc.reach_id = _reach_id
    out_nc.valid = _output['valid']
    out_nc.reliability = reliability
    nc_dict = {}
    for key in list(algo5_results.keys()):
        key_string = ''
        if key.lower() == 'a0':
            key_string = 'A0'
        elif key.lower() == 'n':
            key_string = 'n'
        else:
            key_string = key
        nc_dict[key] = out_nc.createVariable(key_string, 'f8', fill_value=fill_value)
    if np.array(list(algo5_results.keys())).size == 0:
        nc_dict['A0'] = out_nc.createVariable('A0', 'f8', fill_value=fill_value)
        nc_dict['n'] = out_nc.createVariable('n', 'f8', fill_value=fill_value)
    bb_out = out_nc.createVariable('bb', 'f8', fill_value=fill_value)
    bb_out[:] = bb
    if param_dict['run_type'] == 'seq':
        for key in list(nc_dict.keys()):
            if np.array(list(algo5_results.keys())).size > 0:
                nc_dict[key].assignValue(algo5_results[key])
            else:
                nc_dict['A0'].assignValue(np.nan)
                nc_dict['n'].assignValue(np.nan)
    if param_dict['run_type'] == 'set':
        if reach_number < 0:
            a0.assignValue(np.nan)
            n.assignValue(np.nan)
        else:
            if np.array(_output['node_a0']).size > 0:
                a0.assignValue(_output['node_a0'][reach_number])
            else:
                a0.assignValue(np.nan)
            if np.array(_output['node_n']).size > 0:
                n.assignValue(_output['node_n'][reach_number])
            else:
                n.assignValue(np.nan)
    q_algo5 = out_nc.createVariable('Q_mm', 'f8', ('nt',), fill_value=fill_value)
    if param_dict['run_type'] == 'seq':
        q_algo5[:] = _output['q_algo5']
    if param_dict['run_type'] == 'set':
        if reach_number < 0:
            q_algo5[:] = null_a5
        elif np.array(_output['q_algo5_all']).size > 0:
            q_algo5[:] = _output['q_algo5_all'][reach_number]
        else:
            q_algo5[:] = null_a5
    q_algo31 = out_nc.createVariable('Q_da', 'f8', ('nt',), fill_value=fill_value)
    q_algo31[:] = _output['q_algo31']
    if 'width' in _output:
        max_length = max((len(arr) for arr in _output['width']))
        nodes = out_nc.createDimension('nodes', len(_output['width']))
    else:
        max_length = max(_output['time_steps']) + 1
        nodes = out_nc.createDimension('nodes', len(_output['time_steps']))
    nb_pts = out_nc.createDimension('nb_pts', max_length)
    width = out_nc.createVariable('width', 'f8', ('nodes', 'nb_pts'), fill_value=fill_value)
    elevation = out_nc.createVariable('elevation', 'f8', ('nodes', 'nb_pts'), fill_value=fill_value)
    q_u = out_nc.createVariable('q_u', 'f8', ('nt',), fill_value=fill_value)
    if 'width' in _output:
        for i in range(0, len(_output['width'])):
            width[i] = np.pad(_output['width'][i], (0, max_length - len(_output['width'][i])))
            elevation[i] = np.pad(_output['elevation'][i], (0, max_length - len(_output['elevation'][i])))
    else:
        for i in range(0, len(_output['q_algo31'])):
            width[i] = np.pad(_output['q_algo31'][i], (0, max_length - 0))
            elevation[i] = np.pad(_output['q_algo31'][i], (0, max_length - 0))
    q_u[:] = np.ones(len(nt[:])) * fill_value
    logging.info('q_u:%s' % str(q_u[:]))
    logging.info('q_algo31:%s' % str(q_algo31[:]))
    logging.info('len q_algo31:%s' % str(len(q_algo31[:])))
    logging.info('q_algo5:%s' % str(q_algo5[:]))
    logging.info('len q_algo5:%s' % str(len(q_algo5[:])))
    for key in list(nc_dict.keys()):
        logging.info(f'{key}=%s' % str(nc_dict[key][:]))
    logging.info('times=%s' % str(times[:]))
    logging.info('bb:%s' % str(bb_out[:]))
    if param_dict['gnuplot_saving']:
        reach_id = str(_reach_id)
        output_dir = output_dir.joinpath('gnuplot_data', reach_id)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)
        gnuplot_save_q(q_algo31[:], times[:], output_dir.joinpath('qalgo31'))
    out_nc.close()

def read_bathymetry(bathymetry_path, dist):
    width = []
    wse = []
    temp1 = []
    temp2 = []
    indexes = []
    with open(bathymetry_path, 'r') as file:
        j = 0
        for line in file:
            words = line.split()
            if len(words) == 2:
                element1 = float(words[0])
                element2 = float(words[1])
                temp1.append(element1)
                temp2.append(element2)
            if len(words) != 2:
                if len(words) >= 11:
                    index = []
                    index = np.where(np.isclose(float(words[2]), dist))
                    if np.array(temp1).size > 0 and np.array(temp2).size > 0:
                        width.append(temp1)
                        wse.append(temp2)
                        temp1 = []
                        temp2 = []
                        j = len(wse)
                        if np.array(index[0]).size > 0:
                            indexes.append(j)
                    if np.array(index[0]).size > 0:
                        pass
    return (width, wse, indexes)

def tmp_filter_function(node_z0_array, node_w0_array, q_min_n_nodes_param: int=1, q_min_n_times_param: int=1, q_min_per_nodes_param: float=0.0, q_min_per_times_param: float=0.0) -> tuple:
    Proceed = True
    if node_z0_array.shape != node_w0_array.shape:
        raise AssertionError('arrays should have the same shape')
    total_n_nodes, total_n_times = node_z0_array.shape
    node_z_array = masked_array_to_nan_array(copy.deepcopy(node_z0_array))
    node_w_array = masked_array_to_nan_array(copy.deepcopy(node_w0_array))
    node_w_array = np.where(node_w_array <= 0, np.nan, node_w_array)
    if np.all(np.isnan(node_z_array)):
        Proceed = False
        missing_indexes = [i for i in range(total_n_times)]
        return (Proceed, [], [], missing_indexes)
    valid_cs_node_ids = (np.count_nonzero(np.isfinite(node_w_array), axis=1) > 2).nonzero()[0]
    if valid_cs_node_ids.size == 0:
        Proceed = False
        missing_indexes = [i for i in range(total_n_times)]
        return (Proceed, [], [], missing_indexes)
    invalid_cs_node_ids = [i_ for i_ in range(total_n_nodes) if i_ not in valid_cs_node_ids]
    node_w_array[invalid_cs_node_ids, :] = np.nan
    nan_mask_array = get_mask_nan_across_arrays(node_z_array, node_w_array)
    if np.all(nan_mask_array):
        Proceed = False
        missing_indexes = [i for i in range(total_n_times)]
        return (Proceed, [], [], missing_indexes)
    node_z_array[nan_mask_array] = np.nan
    node_z_array.shape = (total_n_nodes, total_n_times)
    df = pd.DataFrame(index=range(total_n_nodes), columns=range(total_n_times), data=node_z_array)
    original_shape = df.shape
    df = df.dropna(axis='columns', how='all')
    df = df.dropna(axis='rows', how='all')
    df = df.dropna(axis='columns', thresh=q_min_n_times_param)
    if total_n_times > 10:
        pd_thr = int(q_min_per_times_param / 100 * total_n_times)
        df = df.dropna(axis='columns', thresh=pd_thr)
    df = df.dropna(axis='rows', thresh=q_min_n_nodes_param)
    if total_n_nodes > 10:
        pd_thr = int(q_min_per_nodes_param / 100 * total_n_nodes)
        df = df.dropna(axis='rows', thresh=pd_thr)
    df = df.dropna(axis='columns', how='any')
    df = df.dropna(axis='rows', how='any')
    if df.empty:
        Proceed = False
        missing_indexes = [i for i in range(total_n_times)]
        return (Proceed, [], [], missing_indexes)
    val_node_index_array, val_time_index_array = (df.index.to_numpy(), df.columns.to_numpy())
    missing_indexes = []
    for i in range(original_shape[1]):
        if i not in val_time_index_array:
            missing_indexes.append(i)
    return (Proceed, val_node_index_array, val_time_index_array, missing_indexes)

def sword_table(_input_dir, _reach_dict):
    sword_file = _input_dir.joinpath('sword', f'{_reach_dict['sword']}')
    sword_file_exists = os.path.isfile(sword_file)
    if sword_file_exists:
        sword_dataset = Dataset(sword_file)
        id_ctl = sword_dataset['centerlines/cl_id'][:]
        nb_nod = sword_dataset['nodes/node_id'][:].size
        nb_rch = sword_dataset['reaches/reach_id'][:].size
        id_nod_ctl = sword_dataset['nodes/cl_ids'][:]
        id_rch_ctl = sword_dataset['reaches/cl_ids'][:]
    min_ctr_id = np.min(id_ctl)
    max_ctr_id = np.max(id_ctl)
    nbr_ctr_id = max_ctr_id - min_ctr_id + 1
    index_ctl_ctl = np.ones(nbr_ctr_id) * -1
    nb_ctl = id_ctl.size
    for i in range(0, nb_ctl):
        index_ctl_ctl[id_ctl[i] - min_ctr_id] = i
    index_nod_ctl = np.ones(nbr_ctr_id) * -1
    for i in range(0, nb_nod):
        for j in range(id_nod_ctl[0][i], id_nod_ctl[1][i] + 1):
            index_nod_ctl[j - min_ctr_id] = i
    index_rch_ctl = np.ones(nbr_ctr_id) * -1
    for i in range(0, nb_rch):
        for j in range(id_rch_ctl[0][i], id_rch_ctl[1][i] + 1):
            index_rch_ctl[j - min_ctr_id] = i
    index_rch_node = np.zeros((nb_rch, 2))
    for i in range(0, nb_rch):
        for j in range(0, 2):
            if i == 15348:
                pass
            index_rch_node[i, j] = index_nod_ctl[id_rch_ctl[j, i] - min_ctr_id]
    np.save('index_rch_node', index_rch_node)

def get_run_type(json_path):
    if not json_path.exists() or not json_path.suffix == '.json':
        raise IOError('Unable to load json file %s' % json_path)
    with open(json_path) as jsonfile:
        json_data = json.load(jsonfile)
    run_type = 'seq'
    for elem in json_data:
        if isinstance(elem, list):
            run_type = 'set'
    return run_type

def usgs_q(id_test, ds, ds2, timestamp1, timestamp2):

    def usgs_nonNaN(id_test, ds, timestamp1, timestamp2):

        def usgs_id_decode(data, index):
            var1 = data['model/usgs/usgs_id'][index].astype('U13')[0]
            var1 = int(''.join(var1))
            return var1

        def usgs_id_encode(integer):
            arr = np.pad([*str(testint)], (0, 8), 'constant', constant_values='')
            arr = np.char.encode(arr, encoding='UTF-8')
            return arr
        timetmp1 = timestamp1
        timetmp2 = timestamp2
        timestamp1 = int(datetime2matlabdn(timestamp1))
        timestamp2 = int(datetime2matlabdn(timestamp2))
        jj_usgs = np.where(ds['model']['usgs']['usgs_reach_id'][:] == id_test)
        len_jj_usgs = np.array(jj_usgs).size
        jj_usgs = np.array(jj_usgs)[0]
        q_threshold_ohio = 5
        mean_q_usgs_sos = ds['model']['usgs']['mean_q'][:]
        nb_station_usgs_case = 0
        jj_usgs_qNaN = []
        jj_usgs_qtNaN = []
        if len_jj_usgs >= 1:
            times = ds['model']['usgs']['usgs_qt'][jj_usgs]
            good_usgs_station = []
            nb_station_usgs_case = 0
            for jj in range(0, len_jj_usgs):
                good_usgs_station.append(0)
                index_time = np.where((ds['model']['usgs']['usgs_qt'][jj_usgs[jj]] >= timestamp1) & (ds['model']['usgs']['usgs_qt'][jj_usgs[jj]] <= timestamp2))
                jj_usgs_qNaN = ds['model']['usgs']['usgs_q'][jj_usgs[jj]]
                index_nonNaN = np.where(jj_usgs_qNaN.data > 0)
                all_nonNaN = np.intersect1d(index_time, index_nonNaN)
                jj_usgs_qNaN = jj_usgs_qNaN[all_nonNaN]
                jj_usgs_qtNaN = ds['model']['usgs']['usgs_qt'][jj_usgs[jj]]
                jj_usgs_qtNaN = jj_usgs_qtNaN[all_nonNaN]
                if len(jj_usgs_qNaN) > 0:
                    if np.any(np.array(np.where(jj_usgs_qNaN > 0))[0]):
                        good_usgs_station[jj] += 1
                    if mean_q_usgs_sos[jj_usgs[jj]] > q_threshold_ohio:
                        good_usgs_station[jj] += 1
                else:
                    pass
                val_test = max(good_usgs_station)
                jj_usgs_best = jj_usgs[jj]
                if val_test == 2:
                    nb_station_usgs_case += 1
                    ind_usgs = jj_usgs_best
                else:
                    ind_usgs = 0
        else:
            ind_usgs = 0
        if nb_station_usgs_case > 0:
            pass
        else:
            pass
        return (jj_usgs_qNaN, jj_usgs_qtNaN)

    def q_array(ds2, out):
        timestamp2000 = 946684800
        q_array = []
        for i in range(0, np.array(ds2['reach']['time'][:]).size):
            test_date = int(ds2['reach']['time'][i]) + timestamp2000
            str_test = datetime.datetime.fromtimestamp(test_date).strftime('%A %B %d %Y %I:%M:%S')
            testo = datetime.datetime.strptime(str_test, '%A %B %d %Y %I:%M:%S')
            timestamp3 = int(datetime2matlabdn(testo))
            index_times2 = np.where(out[1][:] == int(timestamp3))
            q_array.append(out[0][index_times2])
        q_array = np.array(q_array)
        q_array = q_array.reshape(1, q_array.size)[0]
        return q_array
    out = usgs_nonNaN(id_test, ds, timestamp1, timestamp2)
    if np.array(out).size != 0:
        q_array = q_array(ds2, out)
        return q_array
    return []

def width_replacement(node_w, reach_w, node_w_mean, reach_w_mean):
    option = 1
    condition_reach = []
    if len(node_w[0]) != len(reach_w) or len(node_w[0]) != len(node_w_mean):
        return (True, node_w)
    if option == 1:
        for t in range(0, len(reach_w)):
            condition_reach.append(calc.check_na(reach_w[t]))
        reach_index_nobs = np.where((reach_w == 0) | condition_reach)
        if np.array(reach_index_nobs).size == len(reach_w):
            return (False, [])
        for t in range(0, len(reach_w)):
            condition2 = []
            for n in range(0, len(node_w)):
                condition2.append(calc.check_na(node_w[n, t]))
            condition = (node_w[:, t] == 0) | condition2
            index_to_replace = np.array(np.where(condition))
            index_to_replace1 = np.array(np.where(node_w[:, t] == 0))
            index_to_replace2 = np.array(np.where(condition2))
            if np.array(index_to_replace).size > 0:
                if not calc.check_na(reach_w[t]) or reach_w[t] > 0.0:
                    node_w[index_to_replace, t] = reach_w[t] * (node_w_mean[t] / reach_w_mean)
                else:
                    pass
    return (True, node_w)

def correlation_nodes(node_z, ref_node, reach_id):
    indicators_nodes = {}
    ref_array = node_z[ref_node, :]
    ref_idx = []
    for t in range(0, len(node_z[0])):
        if calc.check_na(node_z[0, t]):
            pass
        else:
            ref_idx.append(t)
    print('LEN REF ARRAY T:', len(ref_array))
    from lib.lib_indicators import compute_all_indicators_from_predict_true
    for n in range(0, len(node_z)):
        idx_non_nan = []
        for t in range(0, len(node_z[n])):
            if calc.check_na(node_z[n, t]):
                pass
            else:
                idx_non_nan.append(t)
        common_indexes = np.intersect1d(ref_idx, idx_non_nan)
        print(f'common_indexes in t for node {n}:', len(common_indexes))
        indicators_nodes[n] = {'reach_id': str(reach_id), 'nb_dates': len(common_indexes)}
        indicators_nodes[n].update(compute_all_indicators_from_predict_true(ref_array[common_indexes], node_z[n, common_indexes]))
    indicators_node_df = pd.DataFrame(indicators_nodes).transpose()
    indicators_node_df.to_csv(Path('/home/kabey/Bureau/worksync/Pankaj_Ganga/output/nodes_corr_z' + '.csv'), index=True, sep=';')
    print(bug2)

def use_stations_for_q_prior(reach_t, station_q, station_qt):
    option = 1
    station_df = pd.DataFrame({'station_date': station_qt, 'station_q': station_q})
    if reach_t.mask.any():
        sic4dvar_valid_idx = np.where(reach_t.mask == False)
        sic4dvar_t = reach_t[sic4dvar_valid_idx]
    else:
        sic4dvar_t = reach_t[:]
    if np.array(sic4dvar_t).size < 1:
        logging.warning(call_error_message(503))
        return (-9999.0, -9999.0)
    dates_sic = []
    for date in sic4dvar_t:
        dates_sic.append(seconds_to_date_old(date))
    sic_df = pd.DataFrame({'sic4dvar_date': dates_sic})
    sic_df['date_only'] = pd.to_datetime(sic_df['sic4dvar_date'], format='%Y-%m-%d').dt.date
    station_df['date_only'] = pd.to_datetime(station_df['station_date'], format='%Y-%m-%d').dt.date
    if option == 0:
        data_df = pd.merge(station_df, sic_df, on='date_only', how='inner')
        if len(data_df['station_q']) < 1:
            return (-9999.0, -9999.0)
        q_mean = np.nanmean(data_df['station_q'])
    elif option == 1:
        if not params.force_specific_dates:
            start_date = sic_df['date_only'].min()
            end_date = sic_df['date_only'].max()
        else:
            start_date = params.start_date.date()
            end_date = params.end_date.date()
        filtered_station_df = station_df[(station_df['date_only'] >= start_date) & (station_df['date_only'] <= end_date)]
        if len(filtered_station_df['station_q']) < 1:
            logging.warning(call_error_message(502))
            return (-9999.0, -9999.0)
        q_mean = np.nanmean(filtered_station_df['station_q'])
        q_std = np.nanstd(filtered_station_df['station_q'])
    return (q_mean, q_std)
