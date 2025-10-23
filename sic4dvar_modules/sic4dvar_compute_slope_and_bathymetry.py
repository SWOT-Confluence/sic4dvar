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
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import sic4dvar_params as params
from sic4dvar_functions.sic4dvar_helper_functions import gnuplot_save, gnuplot_save_list, gnuplot_save_slope
from sic4dvar_functions.sic4dvar_calculations import check_na, verify_name_length, compute_bb, fnc_APR, f_approx_sections_v6, M
from sic4dvar_functions.S841 import K

try:
    from Confluence.input.input.extract.CalculateHWS import CalculateHWS
    from Confluence.input.input.extract.DomainHWS import DomainHWS
    from Confluence.input.input.extract.HWS_IO import HWS_IO
    Confluence_HWS_method = True
except ImportError:
    Confluence_HWS_method = False

def create_confluence_dict(reach_time, reach_z, reach_w, reach_s):
    ObsData = {}
    ObsData['nR'] = 1
    ObsData['xkm'] = np.nan
    ObsData['L'] = np.nan
    ObsData['nt'] = len(reach_time)
    ObsData['t'] = reach_time
    ObsData['h'] = np.empty((ObsData['nR'], ObsData['nt']))
    ObsData['h0'] = np.empty((ObsData['nR'], 1))
    ObsData['S'] = np.empty((ObsData['nR'], ObsData['nt']))
    ObsData['w'] = np.empty((ObsData['nR'], ObsData['nt']))
    for i in range(0, ObsData['nR']):
        ObsData['h'][i, :] = reach_z
        ObsData['w'][i, :] = reach_w
        ObsData['S'][i, :] = reach_s
    ObsData['sigh'] = 0.1
    ObsData['sigw'] = 10.0
    ObsData['sigS'] = 1.7e-05
    ObsData['iDelete'] = np.where(np.isnan(ObsData['w'][0, :]) | np.isnan(ObsData['h'][0, :]))
    return ObsData

def call_func_APR(node_w, node_z, node_xr, node_yr, params, param_dict):
    node_a = node_w.copy()
    node_p = node_w.copy()
    node_r = node_w.copy()
    node_w_simp = node_w.copy()
    depth_mean = []
    A0_mean = []
    P0_mean = []
    W0_mean = []
    for i in range(len(node_w)):
        node_a[i], node_p[i], node_r[i], node_w_simp[i] = fnc_APR(node_z[i], node_xr[i], node_yr[i])
        depth_section = 0.0
        if param_dict['bb_computation']:
            depth_section = np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])
            A0 = 0
            P0 = node_xr[i][0]
            for j in range(0, len(node_xr[i]) - 1):
                A0 = A0 + (node_xr[i][j] + node_xr[i][j + 1]) / 2 * (node_yr[i][j + 1] - node_yr[i][j])
                P0 = P0 + 2 * np.sqrt((node_xr[i][j + 1] / 2 - node_xr[i][j] / 2) ** 2 + (node_yr[i][j + 1] - node_yr[i][j]) ** 2)
                if node_yr[i][j] - np.nanmin(node_yr[i]) > (np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])) / 2:
                    depth_section = (np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])) / 2
                    break
            if False:
                results2 = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
                plt.plot(node_xr[i], node_yr[i])
                plt.plot(results2[0], results2[1])
                plt.plot(node_w[i], node_z[i], marker='.', linestyle='None')
                print('A0, P0, R, depth, depth/R:', A0, P0, A0 / P0, depth_section, depth_section / (A0 / P0))
                plt.show()
                plt.clf()
            A0_mean.append(A0)
            P0_mean.append(P0)
            W0_mean.append(node_xr[i][0])
        index_temp = np.where(node_w_simp[i] == 0.0)
        node_w_simp[i][index_temp] = np.nanmin(node_xr[i][np.where(node_xr[i] > 0.0)])
        depth_mean.append(depth_section)
    if not param_dict['bb_computation']:
        bb = 9999.0
    else:
        depth_mean = np.nanmean(depth_mean)
        A0_mean = np.nanmean(A0_mean)
        P0_mean = np.nanmean(P0_mean)
        W0_mean = np.nanmean(W0_mean)
        print('depth_mean, A0_mean, P0_mean, W0_mean:', depth_mean, A0_mean, P0_mean, W0_mean)
        print('mean Radius:', A0_mean / P0_mean, depth_mean / (A0_mean / P0_mean))
        bb = compute_bb(depth_mean, W0_mean, A0_mean, P0_mean)
        print('Final computed bb:', bb)
        if bb < 1:
            bb = 1
        print('Final bb:', bb)
    return (node_a, node_p, node_r, node_w_simp, bb)

def slope_computation(sic4dvar_dict):
    if sic4dvar_dict['param_dict']['use_reach_slope']:
        if sic4dvar_dict['param_dict']['run_type'] == 'set':
            for t in range(sic4dvar_dict['filtered_data']['node_z'][0].shape[0]):
                SLOPEM1 = np.zeros(sic4dvar_dict['filtered_data']['node_z'][0].shape[0])
                if sic4dvar_dict['filtered_data']['reach_s'][0][t] > 0.0:
                    if not params.node_length:
                        SLOPEM1[t] = sic4dvar_dict['filtered_data']['reach_s'][0][t] * (sic4dvar_dict['filtered_data']['node_x'][-1] - sic4dvar_dict['filtered_data']['node_x'][0])
                        for i in range(1, sic4dvar_dict['filtered_data']['node_x'].size):
                            pass
                    elif params.node_length:
                        SLOPEM1[t] = sic4dvar_dict['filtered_data']['reach_s'][0][t] * np.sum(sic4dvar_dict['filtered_data']['node_x'])
                else:
                    if len(sic4dvar_dict['filtered_data']['node_z']) < 2:
                        raise 'Need more than 2 sections/reaches to use Algo3.1 to calculate Qm!'
                    if len(sic4dvar_dict['filtered_data']['node_z']) >= 2 and len(sic4dvar_dict['filtered_data']['node_z']) <= 3:
                        SS1 = sic4dvar_dict['filtered_data']['node_z'][0][t]
                        SS2 = sic4dvar_dict['filtered_data']['node_z'][-1][t]
                    if len(sic4dvar_dict['filtered_data']['node_z']) > 3:
                        SS1 = (sic4dvar_dict['filtered_data']['node_z'][0][t] + 2.0 * sic4dvar_dict['filtered_data']['node_z'][1][t] + sic4dvar_dict['filtered_data']['node_z'][2][t]) / 4.0
                        index_x = np.where(abs(sic4dvar_dict['filtered_data']['node_x'][0] - sic4dvar_dict['filtered_data']['node_x'][0:]) < params.DX_Length)
                        index_x = index_x[0]
                        SS2 = (sic4dvar_dict['filtered_data']['node_z'][index_x[-3]][t] + 2.0 * sic4dvar_dict['filtered_data']['node_z'][index_x[-2]][t] + sic4dvar_dict['filtered_data']['node_z'][index_x[-1]][t]) / 4.0
                    SLOPEM1[t] = SS1 - SS2
        if sic4dvar_dict['param_dict']['run_type'] == 'seq':
            if not params.node_length:
                SLOPEM1 = sic4dvar_dict['filtered_data']['reach_s'] * (sic4dvar_dict['filtered_data']['node_x'][-1] - sic4dvar_dict['filtered_data']['node_x'][0])
            elif params.node_length:
                SLOPEM1 = sic4dvar_dict['filtered_data']['reach_s'] * (sic4dvar_dict['filtered_data']['node_x'][-1] - sic4dvar_dict['filtered_data']['node_x'][0])
                sic4dvar_dict['last_node_for_integral'] = len(sic4dvar_dict['filtered_data']['node_x'])
    else:
        SLOPEM1 = np.zeros(sic4dvar_dict['filtered_data']['node_z'][0].shape[0])
        for t in range(sic4dvar_dict['filtered_data']['node_z'][0].shape[0]):
            if len(sic4dvar_dict['filtered_data']['node_z']) < 2:
                logging.warning('WARNING: Need more than 2 sections/reaches to use Algo3.1 to calculate Qm!')
                sic4dvar_dict['output']['valid'] = 0
                sic4dvar_dict['last_node_for_integral'] = 0
                return np.ones(sic4dvar_dict['filtered_data']['node_z'][0].shape[0]) * np.nan
            if len(sic4dvar_dict['filtered_data']['node_z']) == 2:
                SS1 = sic4dvar_dict['filtered_data']['node_z'][0][t]
                SS2 = sic4dvar_dict['filtered_data']['node_z'][-1][t]
                sic4dvar_dict['last_node_for_integral'] = 2
            if len(sic4dvar_dict['filtered_data']['node_z']) >= 3:
                SS1 = sic4dvar_dict['filtered_data']['node_z'][0][t]
                SS2 = sic4dvar_dict['filtered_data']['node_z'][-1][t]
                sic4dvar_dict['last_node_for_integral'] = len(sic4dvar_dict['filtered_data']['node_z'])
            SLOPEM1[t] = SS1 - SS2
        if sic4dvar_dict['param_dict']['gnuplot_saving']:
            reach_id = verify_name_length(str(sic4dvar_dict['input_data']['reach_id']))
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
            if not Path(output_path).is_dir():
                Path(output_path).mkdir(parents=True, exist_ok=True)
            times2 = np.around(sic4dvar_dict['filtered_data']['reach_t'] / 3600 / 24)
            times2 = times2 - min(times2)
    sic4dvar_dict['intermediates_values']['slope'] = SLOPEM1
    return (SLOPEM1, sic4dvar_dict)

def compute_widths_from_breakpoints(height_breakpoints, poly_fits):
    height_breakpoints = np.array(height_breakpoints)
    widths = []
    for i in range(len(height_breakpoints)):
        h = height_breakpoints[i]
        if np.isnan(h):
            widths.append(np.nan)
            continue
        if i == len(height_breakpoints) - 1:
            p = poly_fits[-1]
        else:
            p = poly_fits[i]
        w = np.polyval(p, h)
        widths.append(w)
    return (widths, height_breakpoints)

def mike_method(param_dict, filtered_data, node_z, node_w, i, slope=[], algo='', input_data=[]):
    if param_dict['use_reach_slope']:
        ObsData = create_confluence_dict(filtered_data['reach_t'], node_z[i], node_w[i], filtered_data['reach_s'])
    elif algo != 'algo5':
        print('TEST0')
        print('len:', len(filtered_data['reach_t']), len(node_z[i]), len(node_w[i]), len(slope))
        ObsData = create_confluence_dict(filtered_data['reach_t'], node_z[i], node_w[i], slope)
    else:
        print('TEST1')
        print('len:', len(input_data['reach_t']), len(input_data['node_z'][i]), len(input_data['node_w'][i]), len(slope))
        ObsData = create_confluence_dict(input_data['reach_t'], input_data['node_z'][i], input_data['node_w'][i], slope)
    D = DomainHWS(ObsData)
    hws_obj = CalculateHWS(D, ObsData)
    if hasattr(hws_obj, 'area_fit'):
        results = [hws_obj.area_fit['w_break'], hws_obj.area_fit['h_break']]
        results2 = [hws_obj.wobs, hws_obj.hobs]
    if len(hws_obj.dAall) == 1:
        hws_obj.dAall = hws_obj.dAall[0]
    reach_dA = hws_obj.dAall
    h_breakpoints = hws_obj.h_breakpoints
    polyfits = hws_obj.poly_fits
    check_na_vec = np.vectorize(check_na)
    mask = ~check_na_vec(node_z[i]) & ~check_na_vec(node_w[i])
    new_node_z = node_z[i][mask]
    new_node_w = node_w[i][mask]
    if check_na(h_breakpoints[0]) or h_breakpoints[0] < 0.0:
        pass
        h_breakpoints[0] = np.nanmin(new_node_z)
    if check_na(h_breakpoints[3]) or h_breakpoints[3] < 0.0:
        pass
        h_breakpoints[3] = np.nanmax(new_node_z)
    results = compute_widths_from_breakpoints(np.array(h_breakpoints), np.array(polyfits))
    return (results, reach_dA)

def bathymetry_computation(node_w, node_z, param_dict, params, input_data=[], filtered_data=[], slope=[], force_method='', algo=''):
    node_xr = []
    node_yr = []
    dA = []
    cs_method = param_dict['cs_method']
    if force_method != '':
        cs_method = force_method
    logging.info(f'CS_METHOD: {cs_method}')
    for i in range(len(node_w)):
        if not params.pankaj_test:
            if cs_method == 'POM':
                results = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
            elif cs_method == 'Igor':
                results = M(node_w[i], node_z[i], max_iter=params.LSMX, cor_z=None, inter_behavior=True, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, min_change_v_thr=0.0001, first_sweep='forward', cs_float_atol=params.def_float_atol, number_of_nodes=len(node_z), plot=False)
                results = f_approx_sections_v6(results[0], results[1], NbIter=4, SeuilDistance=0.1, FSort=0)
            elif cs_method == 'Mike' and Confluence_HWS_method:
                results, _ = mike_method(param_dict, filtered_data, node_z, node_w, i, slope, algo, input_data)
            node_xr += [results[0]]
            node_yr += [results[1]]
            if param_dict['cs_plot_debug']:
                results_pom = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
                results_igor = M(node_w[i], node_z[i], max_iter=params.LSMX, cor_z=None, inter_behavior=True, inter_behavior_min_thr=params.def_float_atol, inter_behavior_max_thr=params.DX_max_in, min_change_v_thr=0.0001, first_sweep='forward', cs_float_atol=params.def_float_atol, number_of_nodes=len(node_z), plot=False)
                if Confluence_HWS_method:
                    results_mike, dA_mike_2 = mike_method(param_dict, filtered_data, node_z, node_w, i, slope)
                    plt.plot(results_mike[0], results_mike[1], label='Mike')
                plt.plot(results_pom[0], results_pom[1], label='POM')
                print(results_pom[0].shape, results_pom[1].shape)
                plt.plot(results_igor[0], results_igor[1], label='Igor')
                print(results_igor[0].shape, results_igor[1].shape)
                plt.plot(node_w[i], node_z[i], marker='.', linestyle='None', label='orig pts')
                plt.legend(loc='upper right')
                plt.show()
                plt.clf()
    if param_dict['gnuplot_saving']:
        node_x = input_data['node_x']
        times2 = np.around(input_data['reach_t'] / 3600 / 24)
        times2 = times2 - np.nanmin(times2)
        nodes2 = (node_x - node_x[0]) / 1000
        reach_id = str(input_data['reach_id'])
        reach_id = verify_name_length(reach_id)
        output_path = param_dict['output_dir'].joinpath('gnuplot_data', str(reach_id))
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = param_dict['output_dir'].joinpath('gnuplot_data', str(reach_id), 'cs')
    return (node_xr, node_yr, dA)

def compute_slope(sic4dvar_dict, params):
    SLOPEM1, sic4dvar_dict = slope_computation(sic4dvar_dict)
    if params.akgd:
        SLOPEM1_2D = []
        for n in range(0, len(sic4dvar_dict['filtered_data']['node_z'])):
            SLOPEM1_2D.append(SLOPEM1)
        SLOPEM1_2D = np.array(SLOPEM1_2D)
        SLOPEM1_2D = v(dim=0, value0_array=SLOPEM1_2D, base0_array=sic4dvar_dict['filtered_data']['reach_t'], max_iter=100, cor=sic4dvar_dict['cort_wse'], always_run_first_iter=False, behaviour='', inter_behaviour=False, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False)
        if sic4dvar_dict['param_dict']['gnuplot_saving']:
            reach_id = verify_name_length(str(sic4dvar_dict['input_data']['reach_id']))
            output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
            if not Path(output_path).is_dir():
                Path(output_path).mkdir(parents=True, exist_ok=True)
            times2 = np.around(sic4dvar_dict['filtered_data']['reach_t'] / 3600 / 24)
            times2 = times2 - min(times2)
            gnuplot_save_slope(SLOPEM1_2D[0], times2, output_path.joinpath('slope_smooth'))
    if params.constant_slope:
        SLOPEM1_mean = 0.0
        time_scaling = 0.0
        for t in range(1, len(SLOPEM1)):
            SLOPEM1_mean += (SLOPEM1[t] + SLOPEM1[t - 1]) / 2 * (sic4dvar_dict['filtered_data']['reach_t'][t] - sic4dvar_dict['filtered_data']['reach_t'][t - 1])
            time_scaling += sic4dvar_dict['filtered_data']['reach_t'][t] - sic4dvar_dict['filtered_data']['reach_t'][t - 1]
        for t in range(0, len(SLOPEM1)):
            SLOPEM1[t] = SLOPEM1_mean / time_scaling
    if not sic4dvar_dict['output']['valid']:
        sic4dvar_dict['output']['valid'] = 0
        logging.warning('Slope not valid.')
        return (sic4dvar_dict, [], [], [], [])
    return (SLOPEM1, sic4dvar_dict)

def compute_bathymetry(sic4dvar_dict, params, SLOPEM1):
    sic4dvar_dict['input_data']['node_xr'], sic4dvar_dict['input_data']['node_yr'], _ = bathymetry_computation(node_w=sic4dvar_dict['filtered_data']['node_w'], node_z=sic4dvar_dict['filtered_data']['node_z'], param_dict=sic4dvar_dict['param_dict'], params=params, input_data=sic4dvar_dict['input_data'], filtered_data=sic4dvar_dict['filtered_data'], slope=SLOPEM1)
    node_x = sic4dvar_dict['input_data']['node_x']
    reach_id = str(sic4dvar_dict['input_data']['reach_id'])
    reach_id = verify_name_length(reach_id)
    times2 = np.around(sic4dvar_dict['input_data']['reach_t'] / 3600 / 24)
    times2 = times2 - np.nanmin(times2)
    nodes2 = (node_x - node_x[0]) / 1000
    tmp_min = []
    for i in range(0, len(sic4dvar_dict['input_data']['node_yr'])):
        tmp_min.append(np.nanmin(sic4dvar_dict['input_data']['node_yr'][i]))
    if sic4dvar_dict['param_dict']['gnuplot_saving']:
        output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', reach_id)
        if not Path(output_path).is_dir():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        gnuplot_save(nodes2, times2, sic4dvar_dict['input_data']['node_z_ini'], sic4dvar_dict['input_data']['node_w_ini'], output_path.joinpath('out_wse_w' + '_ini'), tmp_min, 2)
        gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][0], sic4dvar_dict['input_data']['node_w_ini'], output_path.joinpath('out_wse' + '_deviation'), tmp_min, 2)
        gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][1], sic4dvar_dict['tmp_interp_values_w'][0], output_path.joinpath('out_wse_w' + '_relax_space1'), tmp_min, 2)
        gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][2], sic4dvar_dict['tmp_interp_values_w'][1], output_path.joinpath('out_wse_w' + '_interpolated'), tmp_min, 2)
        gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][3], sic4dvar_dict['tmp_interp_values_w'][2], output_path.joinpath('out_wse_w' + '_relax_space2'), tmp_min, 2)
        gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][4], sic4dvar_dict['tmp_interp_values_w'][3], output_path.joinpath('out_wse_w' + '_final'), tmp_min, 1)
        if len(sic4dvar_dict['tmp_interp_values']) > 5:
            gnuplot_save(nodes2, times2, sic4dvar_dict['tmp_interp_values'][5], sic4dvar_dict['input_data']['node_w_ini'], output_path.joinpath('out_wse' + '_deviation_new'), tmp_min, 2)
        gnuplot_save_list(nodes2, times2, sic4dvar_dict['input_data']['node_yr'], sic4dvar_dict['input_data']['node_xr'], output_path.joinpath('out_z_w' + '_bathy'), tmp_min, 2)
    sic4dvar_dict['input_data']['node_a'], sic4dvar_dict['input_data']['node_p'], sic4dvar_dict['input_data']['node_r'], sic4dvar_dict['input_data']['node_w_simp'], sic4dvar_dict['bb'] = call_func_APR(sic4dvar_dict['filtered_data']['node_w'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_xr'], sic4dvar_dict['input_data']['node_yr'], params, sic4dvar_dict['param_dict'])
    tmp40 = []
    tmp41 = []
    tmp42 = []
    tmp43 = []
    tmp44 = []
    for n in range(0, sic4dvar_dict['filtered_data']['node_z'].shape[0]):
        tmp41.append(sic4dvar_dict['filtered_data']['node_z'][n, sic4dvar_dict['filtered_data']['node_z'][n, :].argmax()])
        tmp43.append(sic4dvar_dict['filtered_data']['node_w'][n, sic4dvar_dict['filtered_data']['node_w'][n, :].argmax()])
        tmp40.append(sic4dvar_dict['input_data']['node_xr'][n][sic4dvar_dict['input_data']['node_xr'][n].argmin()])
        tmp44.append(sic4dvar_dict['input_data']['node_xr'][n].argmin())
    apr_array = {'node_w_simp': sic4dvar_dict['input_data']['node_w_simp'], 'node_a': sic4dvar_dict['input_data']['node_a'], 'node_p': sic4dvar_dict['input_data']['node_p'], 'node_r': sic4dvar_dict['input_data']['node_r']}
    bathymetry_array = {'node_xr': sic4dvar_dict['input_data']['node_xr'], 'node_yr': sic4dvar_dict['input_data']['node_yr']}
    if (sic4dvar_dict['param_dict']['q_prior_from_stations'] or params.q_prior_from_ML) and params.station_period_only:
        if sic4dvar_dict['param_dict']['q_prior_from_stations']:
            string = 'station_qt_2000'
        elif params.q_prior_from_ML:
            string = 'ML_qt_2000'
        reach_times = []
        for t in range(0, len(sic4dvar_dict['filtered_data']['reach_t'])):
            reach_times.append(sic4dvar_dict['filtered_data']['reach_t'][t] / 86400)
        reach_times_to_keep = []
        for t in range(0, len(reach_times)):
            if reach_times[t] >= sic4dvar_dict['input_data'][string].min() and reach_times[t] <= sic4dvar_dict['input_data'][string].max():
                reach_times_to_keep.append(reach_times[t])
        last_time_instant = np.max(reach_times_to_keep) * 86400
        time_indexes_to_keep = []
        seen = set()
        for val in reach_times_to_keep:
            close_matches = np.where(np.abs(np.array(reach_times) - val) <= 0.01)[0]
            for idx in close_matches:
                if idx not in seen:
                    seen.add(idx)
                    time_indexes_to_keep.append(idx)
    else:
        last_time_instant = np.max(sic4dvar_dict['filtered_data']['reach_t'])
        time_indexes_to_keep = np.arange(0, len(sic4dvar_dict['filtered_data']['reach_t']))
    return (sic4dvar_dict, bathymetry_array, apr_array, last_time_instant, time_indexes_to_keep)