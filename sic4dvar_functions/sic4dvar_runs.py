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

import traceback
import logging
import datetime
import os
import multiprocessing as mp
import traceback
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import json
import pandas as pd
import sic4dvar_params as params
from sic4dvar_functions import sic4dvar_calculations as calc
from sic4dvar_functions.sic4dvar_helper_functions import get_input_data, get_reach_dataset, enable_prints, write_output, gnuplot_save_q_station, interp_pdf_tables
from sic4dvar_functions.sic4dvar_calculations import verify_name_length
from sic4dvar_functions.S841 import K as uz
from sic4dvar_modules.sic4dvar_prepare import prepare_params
from sic4dvar_modules.sic4dvar_launcher import sic4dvar_preprocessing, sic4dvar_set_prior, sic4dvar_compute_discharge
from lib.lib_log import set_logger, close_logger, append_to_principal_log, call_error_message
from lib.lib_dates import daynum_to_date
from lib.lib_indicators import compute_all_indicators_from_predict_true
from lib.lib_verif import reorder_ids_with_indices

def worker_fn_modules(j, param_dict, queue, upper):
    try:
        if param_dict['run_type'] == 'seq':
            reach_data = get_reach_dataset(param_dict, j)
            result = seq_run_modules(param_dict, reach_data)
            queue.put((j, result, None))
        elif param_dict['run_type'] == 'set':
            reach_data = get_reach_dataset(param_dict)
            result = set_run_modules(param_dict, reach_data, j, upper)
            queue.put((j, result, None))
    except Exception as e:
        error_info = traceback.format_exc()
        queue.put((j, None, error_info))

def execute_method(args):
    instance, method_name, arg = args
    method = getattr(instance, method_name)
    return method(arg)

def seq_run_modules(param_dict, reach_dict):
    try:
        log_path = param_dict['log_dir'].joinpath('sic4dvar_' + str(reach_dict['reach_id']) + '.log')
        set_logger(param_dict, log_path)
        logging.info(f'Run reach {reach_dict['reach_id']}')
        logging.info('Reach infos : ')
        for k, val in reach_dict.items():
            logging.info('  %s : %s' % (k, val))
        logging.info('Running SIC4DVAR sequentially on reaches one by one.')
        logging.info('Processing reach: ' + str(reach_dict['reach_id']))
        logging.info('Running reach %d' % reach_dict['reach_id'])
        logging.info('No data removed. Data suitable to estimate discharge.')
        input_data, flag_dict = get_input_data(param_dict, reach_dict)
        if type(input_data) != int:
            params.valid_min_z = input_data['valid_min_z']
            params.valid_min_dA = input_data['valid_min_dA']
            logging.info('Running SIC4DVAR (Algo315)!')
            sic4dvar_dict = prepare_params(input_data, flag_dict, param_dict, params)
            if sic4dvar_dict['output']['valid']:
                sic4dvar_dict = sic4dvar_preprocessing(sic4dvar_dict, params)
                sic4dvar_dict, flag_qwbm = sic4dvar_set_prior(sic4dvar_dict)
                sic4dvar_dict = sic4dvar_compute_discharge(sic4dvar_dict, params, flag_qwbm)
            else:
                return None
            if sic4dvar_dict['output']['valid']:
                logging.info('Results are valid !')
            else:
                logging.info('Results are INVALID!')
                if param_dict['aws']:
                    logging.info('Writing results to output directory for AWS.')
                    write_output(param_dict['output_dir'], param_dict, reach_dict['reach_id'], sic4dvar_dict['output'], algo5_results=sic4dvar_dict['algo5_results'], bb=sic4dvar_dict['bb'], reliability=sic4dvar_dict['reliability'])
            if sic4dvar_dict['output']['valid']:
                logging.info('Results are valid !')
                if 'station_q' in input_data.keys() and 'station_date' in input_data.keys():
                    logging.info('Indicators computation')
                    station_df = pd.DataFrame({'station_q': input_data['station_q'], 'station_date': input_data['station_date'], 'station_qt': input_data['station_qt']})
                    station_df['date_only'] = pd.to_datetime(station_df['station_date'], format='%Y-%m-%d').dt.date
                    epoch_0001 = datetime.datetime(1, 1, 1)
                    epoch_2000 = datetime.datetime(2000, 1, 1)
                    delta_days = (epoch_2000 - epoch_0001).days
                    station_df['station_qt_2000'] = [v - delta_days for v in station_df['station_qt']]
                    valid_idx1 = np.where(np.isfinite(sic4dvar_dict['output']['q_algo31']))
                    valid_idx2 = np.where(np.isfinite(sic4dvar_dict['output']['time']))
                    valid_idx = np.intersect1d(valid_idx1, valid_idx2)
                    sic4dvar_date = daynum_to_date(sic4dvar_dict['output']['time'][valid_idx], '2000-01-01')
                    sic4dvar_df = pd.DataFrame({'sic4dvar_q': sic4dvar_dict['output']['q_algo31'][valid_idx], 'sic4dvar_date': sic4dvar_date})
                    sic4dvar_df['date_only'] = pd.to_datetime(sic4dvar_df['sic4dvar_date'], format='%Y-%m-%d').dt.date
                    sic4dvar_df['sic4dvar_qt'] = sic4dvar_dict['output']['time'][valid_idx]
                    qa31_t = deepcopy(np.array(sic4dvar_df['sic4dvar_qt']))
                    if param_dict['gnuplot_saving']:
                        reach_id = str(reach_dict['reach_id'])
                        reach_id = verify_name_length(reach_id)
                        reach_id = verify_name_length(reach_id)
                        output_dir = param_dict['output_dir']
                        output_dir = output_dir.joinpath('gnuplot_data', reach_id)
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True, exist_ok=True)
                        gnuplot_save_q_station(station_df['station_q'], station_df['station_qt_2000'], output_dir.joinpath('q_station_original'))
                        gnuplot_save_q_station(sic4dvar_df['sic4dvar_q'], sic4dvar_df['sic4dvar_qt'], output_dir.joinpath('qa31_estimate_original'))
                    if params.kgokrgo:
                        Q_a31_mean = 0.0
                        time_scaling = 0.0
                        for t in range(1, len(sic4dvar_df['sic4dvar_qt'])):
                            Q_a31_mean += (sic4dvar_df['sic4dvar_q'].iloc[t] + sic4dvar_df['sic4dvar_q'].iloc[t - 1]) / 2 * (sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1])
                            time_scaling += sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1]
                        Q_a31_mean = Q_a31_mean / time_scaling
                        Q_a31_std = 0.0
                        for t in range(1, len(sic4dvar_df['sic4dvar_qt'])):
                            Q_a31_std += ((sic4dvar_df['sic4dvar_q'].iloc[t] - Q_a31_mean) ** 2 + (sic4dvar_df['sic4dvar_q'].iloc[t - 1] - Q_a31_mean) ** 2) / 2 * (sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1])
                        Q_a31 = np.array(sic4dvar_df['sic4dvar_q'])
                        it_pp_max = 5
                        it_pp = 0.0
                        if param_dict['q_prior_from_stations']:
                            q_std = sic4dvar_dict['input_data']['q_std_station'][0]
                        else:
                            q_std = sic4dvar_dict['input_data']['quant_var']
                        while Q_a31_std > q_std and it_pp < it_pp_max:
                            it_pp = it_pp + 1
                            cor_test = (qa31_t[-1] - qa31_t[0]) / (len(qa31_t) - 1)
                            Q_a31_2D = np.ones((len(qa31_t), len(qa31_t))) * np.nan
                            for t in range(0, len(qa31_t)):
                                Q_a31_2D[t, :] = Q_a31
                            Q_a31_2D = uz(dim=0, value0_array=Q_a31_2D, base0_array=np.array(qa31_t), max_iter=params.LSMT, cor=cor_test, always_run_first_iter=False, behaviour='', inter_behaviour=False, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=True)
                            Q_a31 = deepcopy(Q_a31_2D[0])
                            Q_a31_std = 0.0
                            for t in range(1, len(sic4dvar_df['sic4dvar_qt'])):
                                Q_a31_std += ((Q_a31[t] - Q_a31_mean) ** 2 + (Q_a31[t - 1] - Q_a31_mean) ** 2) / 2 * (sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1])
                                time_scaling += sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1]
                            Q_a31_std = np.sqrt(Q_a31_std / time_scaling)
                            print('it_pp, Q_a31_std:', it_pp, Q_a31_std)
                        sic4dvar_df['sic4dvar_q'] = deepcopy(Q_a31)
                    if param_dict['gnuplot_saving']:
                        reach_id = str(reach_dict['reach_id'])
                        reach_id = verify_name_length(reach_id)
                        reach_id = verify_name_length(reach_id)
                        output_dir = param_dict['output_dir']
                        output_dir = output_dir.joinpath('gnuplot_data', reach_id)
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True, exist_ok=True)
                        gnuplot_save_q_station(sic4dvar_df['sic4dvar_q'], sic4dvar_df['sic4dvar_qt'], output_dir.joinpath('qa31111'))
                else:
                    logging.info('No station data.')
                logging.info(f'seq_run : writing results to output directory {param_dict['output_dir']}')
                if not sic4dvar_dict['output']['valid_a5']:
                    logging.info('No algo5 results.')
                if sic4dvar_dict['reliability'] == 'unreliable':
                    pass
                write_output(param_dict['output_dir'], param_dict, reach_dict['reach_id'], sic4dvar_dict['output'], algo5_results=sic4dvar_dict['algo5_results'], bb=sic4dvar_dict['bb'], reliability=sic4dvar_dict['reliability'])
            else:
                logging.info('Results are INVALID!')
                if param_dict['aws']:
                    logging.info('Writing results to output directory for AWS.')
                    write_output(param_dict['output_dir'], param_dict, reach_dict['reach_id'], sic4dvar_dict['output'], algo5_results=sic4dvar_dict['algo5_results'], bb=sic4dvar_dict['bb'], reliability=sic4dvar_dict['reliability'])
            append_to_principal_log(param_dict, f'status of reach {reach_dict['reach_id']} : 0')
        close_logger(param_dict)
    except Exception as e:
        traceback.print_exception(e)
        logging.error(f'Error computing reach_id {reach_dict['reach_id']}: {e} ... skip')
        close_logger(param_dict)
        if not param_dict['safe_mode']:
            return -1

def seq_run(param_dict, reach_dict):
    try:
        log_path = param_dict['log_dir'].joinpath('sic4dvar_' + str(reach_dict['reach_id']) + '.log')
        set_logger(param_dict, log_path)
        logging.info(f'Run reach {reach_dict['reach_id']}')
        logging.info('Reach infos : ')
        for k, val in reach_dict.items():
            logging.info('  %s : %s' % (k, val))
        logging.info('Running SIC4DVAR sequentially on reaches one by one.')
        logging.info('Processing reach: ' + str(reach_dict['reach_id']))
        logging.info('Running reach %d' % reach_dict['reach_id'])
        logging.info('No data removed. Data suitable to estimate discharge.')
        input_data, flag_dict = get_input_data(param_dict, reach_dict)
        if type(input_data) != int:
            params.valid_min_z = input_data['valid_min_z']
            params.valid_min_dA = input_data['valid_min_dA']
            logging.info('Running SIC4DVAR (Algo315)!')
            processed = algorithms(input_data, flag_dict, param_dict)
            if processed.output['valid']:
                processed.launch_sic4dvar()
            else:
                return None
            if processed.output['valid']:
                logging.info('Results are valid !')
                if 'station_q' in input_data.keys() and 'station_date' in input_data.keys():
                    logging.info('Indicators computation')
                    station_df = pd.DataFrame({'station_q': input_data['station_q'], 'station_date': input_data['station_date'], 'station_qt': input_data['station_qt']})
                    station_df['date_only'] = pd.to_datetime(station_df['station_date'], format='%Y-%m-%d').dt.date
                    epoch_0001 = datetime.datetime(1, 1, 1)
                    epoch_2000 = datetime.datetime(2000, 1, 1)
                    delta_days = (epoch_2000 - epoch_0001).days
                    station_df['station_qt_2000'] = [v - delta_days for v in station_df['station_qt']]
                    valid_idx1 = np.where(np.isfinite(processed.output['q_algo31']))
                    valid_idx2 = np.where(np.isfinite(processed.output['time']))
                    valid_idx = np.intersect1d(valid_idx1, valid_idx2)
                    sic4dvar_date = daynum_to_date(processed.output['time'][valid_idx], '2000-01-01')
                    sic4dvar_df = pd.DataFrame({'sic4dvar_q': processed.output['q_algo31'][valid_idx], 'sic4dvar_date': sic4dvar_date})
                    sic4dvar_df['date_only'] = pd.to_datetime(sic4dvar_df['sic4dvar_date'], format='%Y-%m-%d').dt.date
                    sic4dvar_df['sic4dvar_qt'] = processed.output['time'][valid_idx]
                    qa31_t = deepcopy(np.array(sic4dvar_df['sic4dvar_qt']))
                    if param_dict['gnuplot_saving']:
                        reach_id = str(reach_dict['reach_id'])
                        reach_id = verify_name_length(reach_id)
                        reach_id = verify_name_length(reach_id)
                        output_dir = param_dict['output_dir']
                        output_dir = output_dir.joinpath('gnuplot_data', reach_id)
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True, exist_ok=True)
                        gnuplot_save_q_station(station_df['station_q'], station_df['station_qt_2000'], output_dir.joinpath('q_station_original'))
                        gnuplot_save_q_station(sic4dvar_df['sic4dvar_q'], sic4dvar_df['sic4dvar_qt'], output_dir.joinpath('qa31_estimate_original'))
                    if params.kgokrgo:
                        Q_a31_mean = 0.0
                        time_scaling = 0.0
                        for t in range(1, len(sic4dvar_df['sic4dvar_qt'])):
                            Q_a31_mean += (sic4dvar_df['sic4dvar_q'].iloc[t] + sic4dvar_df['sic4dvar_q'].iloc[t - 1]) / 2 * (sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1])
                            time_scaling += sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1]
                        Q_a31_mean = Q_a31_mean / time_scaling
                        Q_a31_std = 0.0
                        for t in range(1, len(sic4dvar_df['sic4dvar_qt'])):
                            Q_a31_std += ((sic4dvar_df['sic4dvar_q'].iloc[t] - Q_a31_mean) ** 2 + (sic4dvar_df['sic4dvar_q'].iloc[t - 1] - Q_a31_mean) ** 2) / 2 * (sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1])
                        Q_a31 = np.array(sic4dvar_df['sic4dvar_q'])
                        it_pp_max = 5
                        it_pp = 0.0
                        if param_dict['q_prior_from_stations']:
                            q_std = processed.input_data['q_std_station'][0]
                        else:
                            q_std = processed.input_data['quant_var']
                        while Q_a31_std > q_std and it_pp < it_pp_max:
                            it_pp = it_pp + 1
                            cor_test = np.array((qa31_t[-1] - qa31_t[0]) / (len(qa31_t) - 1))
                            Q_a31_2D = np.ones((len(qa31_t), len(qa31_t))) * np.nan
                            for t in range(0, len(qa31_t)):
                                Q_a31_2D[t, :] = Q_a31
                            Q_a31_2D = uz(dim=0, value0_array=Q_a31_2D, base0_array=np.array(qa31_t), max_iter=params.LSMT, cor=cor_test, always_run_first_iter=False, behaviour='', inter_behaviour=False, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False, time_integration=True)
                            Q_a31 = deepcopy(Q_a31_2D[0])
                            Q_a31_std = 0.0
                            for t in range(1, len(sic4dvar_df['sic4dvar_qt'])):
                                Q_a31_std += ((Q_a31[t] - Q_a31_mean) ** 2 + (Q_a31[t - 1] - Q_a31_mean) ** 2) / 2 * (sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1])
                                time_scaling += sic4dvar_df['sic4dvar_qt'].iloc[t] - sic4dvar_df['sic4dvar_qt'].iloc[t - 1]
                            Q_a31_std = np.sqrt(Q_a31_std / time_scaling)
                            print('it_pp, Q_a31_std:', it_pp, Q_a31_std)
                        sic4dvar_df['sic4dvar_q'] = deepcopy(Q_a31)
                    if param_dict['gnuplot_saving']:
                        reach_id = str(reach_dict['reach_id'])
                        reach_id = verify_name_length(reach_id)
                        reach_id = verify_name_length(reach_id)
                        output_dir = param_dict['output_dir']
                        output_dir = output_dir.joinpath('gnuplot_data', reach_id)
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True, exist_ok=True)
                        gnuplot_save_q_station(sic4dvar_df['sic4dvar_q'], sic4dvar_df['sic4dvar_qt'], output_dir.joinpath('qa31111111111111111'))
                    if params.indicators_experimental_computation:
                        if not params.force_specific_dates:
                            start_date = sic4dvar_df['date_only'].min()
                            end_date = sic4dvar_df['date_only'].max()
                        else:
                            start_date = params.start_date.date()
                            end_date = params.end_date.date()
                        filtered_station_df = station_df[(station_df['date_only'] >= start_date) & (station_df['date_only'] <= end_date)]
                        start_date_sic = sic4dvar_df['date_only'].min()
                        end_date_sic = sic4dvar_df['date_only'].max()
                        if filtered_station_df.empty:
                            print("No data for station on SWOT observations period. Can't compute indicators.")
                            data_df = pd.DataFrame()
                        elif not filtered_station_df.empty:
                            start_date = filtered_station_df['date_only'].min()
                            end_date = filtered_station_df['date_only'].max()
                            if start_date < start_date_sic:
                                start_date = start_date_sic
                            if end_date > end_date_sic:
                                end_date = end_date_sic
                            times_stations = filtered_station_df[(filtered_station_df['date_only'] >= start_date) & (filtered_station_df['date_only'] <= end_date)]
                            times_sic = sic4dvar_df[(sic4dvar_df['date_only'] >= start_date) & (sic4dvar_df['date_only'] <= end_date)]
                            q_ref = []
                            q_est = []
                            t_ref = []
                            time_system = np.arange(times_stations['station_qt_2000'].min(), times_stations['station_qt_2000'].max() + 1, 1.0)
                            for t in range(0, len(time_system)):
                                q_ref.append(interp_pdf_tables(len(filtered_station_df['station_q']) - 1, time_system[t], np.array(filtered_station_df['station_qt_2000']), np.array(filtered_station_df['station_q'])))
                                q_est.append(interp_pdf_tables(len(sic4dvar_df['sic4dvar_q']) - 1, time_system[t], np.array(sic4dvar_df['sic4dvar_qt']), np.array(sic4dvar_df['sic4dvar_q'])))
                                t_ref.append(time_system[t])
                            data_df = pd.DataFrame({'station_q': q_ref, 'station_date': t_ref, 'sic4dvar_q': q_est})
                            Q_est_mean = 0.0
                            time_scaling = 0.0
                            for t in range(1, len(data_df['station_q'])):
                                Q_est_mean += (data_df['sic4dvar_q'].iloc[t] + data_df['sic4dvar_q'].iloc[t - 1]) / 2 * (data_df['station_date'].iloc[t] - data_df['station_date'].iloc[t - 1])
                                time_scaling += data_df['station_date'].iloc[t] - data_df['station_date'].iloc[t - 1]
                            Q_est_mean = Q_est_mean / time_scaling
                            Q_ref_mean = processed.input_data['q_mean_station'][0]
                            Q_ref_norm = np.sum(data_df['station_q'] ** 2) ** (1 / 2)
                            bias = Q_est_mean - Q_ref_mean
                            nbias = bias / Q_ref_mean
                            absnbias = abs(nbias)
                            Q_prior = processed.input_data['reach_qwbm'][0]
                            bias_prior = Q_prior - Q_ref_mean
                            nbias_prior = bias_prior / Q_ref_mean
                            absnbias_prior = abs(nbias_prior)
                            nrmse_sum = 0.0
                            nrmse2_sum = 0.0
                            time_scaling = 0.0
                            for t in range(1, len(data_df['station_q'])):
                                nrmse_sum += ((data_df['sic4dvar_q'].iloc[t] - data_df['station_q'].iloc[t]) ** 2 + (data_df['sic4dvar_q'].iloc[t - 1] - data_df['station_q'].iloc[t - 1]) ** 2) / 2 * (data_df['station_date'].iloc[t] * 86400 - data_df['station_date'].iloc[t - 1] * 86400)
                                nrmse2_sum += ((data_df['sic4dvar_q'].iloc[t] - data_df['station_q'].iloc[t] - bias) ** 2 + (data_df['sic4dvar_q'].iloc[t - 1] - data_df['station_q'].iloc[t - 1] - bias) ** 2) / 2 * (data_df['station_date'].iloc[t] * 86400 - data_df['station_date'].iloc[t - 1] * 86400)
                                time_scaling += data_df['station_date'].iloc[t] * 86400 - data_df['station_date'].iloc[t - 1] * 86400
                            nrmse = np.sqrt(nrmse_sum / time_scaling) / Q_ref_mean
                            nrmse2 = np.sqrt(nrmse2_sum / time_scaling) / Q_ref_mean
                            spearman = 0.0
                            sp_q_est = np.zeros(len(data_df['station_q']))
                            sp_q_ref = np.zeros(len(data_df['station_q']))
                            for t in range(1, len(data_df['station_q'])):
                                sp_q_est[t] = (data_df['sic4dvar_q'].iloc[t] + data_df['sic4dvar_q'].iloc[t - 1]) / 2 * (data_df['station_date'].iloc[t] - data_df['station_date'].iloc[t - 1])
                                sp_q_ref[t] = (data_df['station_q'].iloc[t] + data_df['station_q'].iloc[t - 1]) / 2 * (data_df['station_date'].iloc[t] - data_df['station_date'].iloc[t - 1])
                            from lib.lib_indicators import spearman_correlation
                            sp_q_est = sp_q_est[1:-1]
                            sp_q_ref = sp_q_ref[1:-1]
                            spearman, _ = spearman_correlation(sp_q_ref, sp_q_est)
                            if param_dict['gnuplot_saving']:
                                reach_id = str(reach_dict['reach_id'])
                                reach_id = verify_name_length(reach_id)
                                reach_id = verify_name_length(reach_id)
                                output_dir = param_dict['output_dir']
                                output_dir = output_dir.joinpath('gnuplot_data', reach_id)
                                if not output_dir.is_dir():
                                    output_dir.mkdir(parents=True, exist_ok=True)
                                gnuplot_save_q_station(data_df['station_q'], data_df['station_date'], output_dir.joinpath('q_station_timesystem'))
                                gnuplot_save_q_station(data_df['sic4dvar_q'], data_df['station_date'], output_dir.joinpath('qa31_estimate_timesystem'))
                    elif not params.indicators_experimental_computation:
                        data_df = sic4dvar_df.merge(station_df, how='inner', left_on='sic4dvar_date', right_on='station_date')
                    if len(data_df) > 1:
                        indicators = {'reach_id': reach_dict['reach_id'], 'nb_dates': len(data_df), 'width': input_data['reach_w_mean'][0], 'reach_length': input_data['reach_length'][0], 'slope': input_data['reach_slope'][0]}
                        if params.indicators_experimental_computation:
                            indicators['nrmse'] = nrmse
                            indicators['nbias'] = nbias
                            indicators['absnbias'] = absnbias
                            indicators['spearman'] = spearman
                            indicators['nrmse2'] = nrmse2
                            indicators['absnbias_prior'] = absnbias_prior
                        else:
                            indicators.update(compute_all_indicators_from_predict_true(np.array(data_df['station_q']).astype('float'), np.array(data_df['sic4dvar_q']).astype('float')))
                        logging.info(f'Indicators name : {';'.join(list(indicators.keys()))}')
                        logging.info(f'Indicators values : {';'.join([str(e) for e in indicators.values()])}')
                    else:
                        logging.info('data_df < 1.')
                else:
                    logging.info("No station data. Can't compute indicators.")
                logging.info(f'seq_run : writing results to output directory {param_dict['output_dir']}')
                if not processed.output['valid_a5']:
                    logging.info('No algo5 results.')
                if processed.reliability == 'unreliable':
                    pass
                write_output(param_dict['output_dir'], param_dict, reach_dict['reach_id'], processed.output, algo5_results=processed.algo5_results, bb=processed.bb, reliability=processed.reliability)
            else:
                logging.info('Results are INVALID!')
                if param_dict['aws']:
                    logging.info('Writing results to output directory for AWS.')
                    write_output(param_dict['output_dir'], param_dict, reach_dict['reach_id'], processed.output, algo5_results=processed.algo5_results, bb=processed.bb, reliability=processed.reliability)
            append_to_principal_log(param_dict, f'status of reach {reach_dict['reach_id']} : 0')
        close_logger(param_dict)
    except Exception as e:
        traceback.print_exception(e)
        logging.error(f'Error computing reach_id {reach_dict['reach_id']}: {e} ... skip')
        close_logger(param_dict)
        if not param_dict['safe_mode']:
            return -1

def run_parallel(param_dict, down, upper, max_procs=4):
    manager = mp.Manager()
    queue = manager.Queue()
    results = {}
    total_jobs = upper - down
    active_procs = []
    for j in range(down, upper):
        p = mp.Process(target=worker_fn_modules, args=(j, param_dict, queue, upper))
        p.start()
        active_procs.append(p)
        if len(active_procs) >= max_procs:
            for proc in active_procs:
                proc.join()
            active_procs.clear()
        while not queue.empty():
            j_done, result, err = queue.get()
            if err:
                print(f'Job {j_done} failed: {err}')
                if not param_dict['safe_mode']:
                    return -1
            else:
                results[j_done] = result
    for proc in active_procs:
        proc.join()
    while not queue.empty():
        j_done, result, err = queue.get()
        if err:
            print(f'Job {j_done} failed: {err}')
            if not param_dict['safe_mode']:
                return -1
        else:
            results[j_done] = result
    return results

def sic4dvar_run(param_dict):
    t0 = datetime.datetime.utcnow()
    reach_dict = get_reach_dataset(param_dict)
    if param_dict['run_type'] == 'seq':
        append_to_principal_log(param_dict, 'Running SIC4DVAR sequentially one by one')
        append_to_principal_log(param_dict, f'Number of reaches is : {len(reach_dict)}')
    elif param_dict['run_type'] == 'set':
        append_to_principal_log(param_dict, 'Running SIC4DVAR in sets mode')
        append_to_principal_log(param_dict, f'Number of sets is : {len(reach_dict)}')
    if params.create_tables:
        reach_dict = get_reach_dataset(param_dict, 0)
        sword_table(params.input_dir, reach_dict)
    if param_dict['aws']:
        if 'index' in param_dict.keys():
            if param_dict['index'] == -256:
                down = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'))
                upper = down + 1
            elif param_dict['index'] >= 0:
                down = param_dict['index']
                upper = param_dict['index'] + 1
        else:
            down = 0
            upper = len(reach_dict)
    else:
        down = 0
        upper = len(reach_dict)
    append_to_principal_log(param_dict, f'down,upper: {down} {upper}')
    if param_dict['flag_parallel']:
        results = run_parallel(param_dict, down, upper, max_procs=params.max_cores)
    else:
        for j in range(down, upper):
            if param_dict['run_type'] == 'seq':
                seq_run_modules(param_dict, get_reach_dataset(param_dict, j))
            elif param_dict['run_type'] == 'set':
                set_run_modules(param_dict, get_reach_dataset(param_dict), j, upper)
    t1 = datetime.datetime.utcnow()
    enable_prints()

def set_check_activation(param_dict, reach_dict):
    append_to_principal_log(param_dict, 'Running SIC4DVAR sequentially on sets')
    append_to_principal_log(param_dict, 'Number of sets is : %d ' % len(reach_dict))
    if param_dict['aws'] == True and os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'):
        down = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'))
        upper = down + 1
    else:
        down = 0
        upper = len(reach_dict)
    if param_dict['deactivate_set_run']:
        append_to_principal_log(param_dict, 'Set run is deactivated, runs each reach of set sequentially')
        for i_set in range(down, upper):
            reach_dict_list = get_reach_dataset(param_dict, i_set)
            for j_reach, reach_dict in enumerate(reach_dict_list):
                print(j_reach, reach_dict['reach_id'])
                if str(reach_dict['reach_id'])[-1] == '1':
                    para_dict_tmp = param_dict.copy()
                    para_dict_tmp['run_type'] = 'seq'
                    seq_run(para_dict_tmp, reach_dict)
                else:
                    logging.error(call_error_message('105').format(reach_id=reach_dict['reach_id']))
                    append_to_principal_log(param_dict, f'status of reach {i_set} reach_id {reach_dict['reach_id']} : 101')
            append_to_principal_log(param_dict, f'status of set {i_set} reach_id {reach_dict_list[0]['reach_id']} to {reach_dict_list[-1]['reach_id']} : 2')
        return (0, 0)
    else:
        return (down, upper)

def set_reaches_init(nbr_reach, reach_dict, param_dict, processed, reaches_ids, i_set):
    for j in range(0, nbr_reach):
        if nbr_reach > 1:
            if str(reach_dict[i_set][j]['reach_id'])[-1] == '1':
                input_data, flag_dict = get_input_data(param_dict, reach_dict[i_set][j])
                sic4dvar_dict = prepare_params(input_data, flag_dict, param_dict, params)
                processed.append(sic4dvar_dict)
                reaches_ids.append(reach_dict[i_set][j]['reach_id'])
            else:
                logging.info(f'Reach {reach_dict[i_set][j]['reach_id']} not processed because reach type != 1')
        elif 'reach_id' in reach_dict[i_set]:
            if str(reach_dict[i_set]['reach_id'])[-1] == '1':
                input_data, flag_dict = get_input_data(param_dict, reach_dict[i_set])
                sic4dvar_dict = prepare_params(input_data, flag_dict, param_dict, params)
                processed.append(sic4dvar_dict)
                log_name = log_name + str(reach_dict[i_set]['reach_id']) + '_'
                reaches_ids.append(reach_dict[i_set]['reach_id'])
            else:
                pass
        elif 'reach_id' in reach_dict[i_set][0]:
            if str(reach_dict[i_set][0]['reach_id'])[-1] == '1':
                input_data, flag_dict = get_input_data(param_dict, reach_dict[i_set][0])
                sic4dvar_dict = prepare_params(input_data, flag_dict, param_dict, params)
                processed.append(sic4dvar_dict)
                log_name = log_name + str(reach_dict[i_set][0]['reach_id']) + '_'
                reaches_ids.append(reach_dict[i_set][0]['reach_id'])
            else:
                pass
    return (processed, reaches_ids)

def set_get_input_data(processed, keys, dim_n, dim_t, temp2, nbr_reach, run_flag):
    nb_lines = 0
    if np.array(processed).size == 0:
        run_flag = False
    get_dim_t = -1
    for i in range(0, nbr_reach):
        if np.array(processed).size == 0:
            break
        elif 'reach_z' in processed[i]['input_data']:
            get_dim_t = processed[i]['input_data']['reach_z'].shape[0]
            break
    if get_dim_t == -1:
        run_flag = False
    if nbr_reach != len(processed):
        nbr_reach = len(processed)
    if run_flag:
        for key in keys:
            temp = []
            tmp_t = []
            for i in range(0, nbr_reach):
                if np.array(processed[i]).size == 0:
                    pass
                if np.array(processed[i]['input_data']).size == 0:
                    pass
                if 'valid_min_z' in processed[i]['input_data']:
                    params.valid_min_z = processed[i]['input_data']['valid_min_z']
                    params.valid_min_dA = processed[i]['input_data']['valid_min_dA']
                else:
                    pass
                if key in processed[i]['input_data']:
                    if key == 'node_t':
                        dim_n.append(processed[i]['input_data'][key].shape[0])
                        dim_t.append(processed[i]['input_data'][key].shape[1])
                        nb_lines += processed[i]['input_data'][key].shape[0]
                    temp.append(processed[i]['input_data'][key])
                else:
                    if key == 'node_t':
                        dim_n.append(processed[i]['input_data']['node_length'].shape[0])
                        nb_lines += processed[i]['input_data']['node_length'].shape[0]
                    empty_array = np.full((processed[i]['input_data']['node_length'].shape[0], get_dim_t), np.nan)
                    temp.append(np.array(empty_array))
            temp2[key] = temp
    else:
        for key in keys:
            temp2[key] = []
    return (temp2, dim_n, dim_t, nb_lines)

def set_time_check(run_flag, temp2):
    All_T = []
    Unique_T = []
    tmp3 = []
    Generic_T = []
    if run_flag:
        for nb_process in range(0, len(temp2['node_t'])):
            All_T = []
            for n in range(0, temp2['node_t'][nb_process].shape[0]):
                All_T.append(temp2['node_t'][nb_process][n, :])
            if np.array(All_T).size > 0:
                All_T = np.sort(np.unique(np.hstack(All_T)))
            tmp3 = []
            tmp3.append(All_T)
            Unique_T = np.hstack((Unique_T, All_T))
        Unique_T = np.sort(np.unique(Unique_T[np.where(Unique_T > 0.0)]))
        if np.array(Unique_T).size > 0:
            Generic_T.append(Unique_T[0])
        else:
            logging.info("No time instant. Can't run.")
            run_flag = False
    return (run_flag, Unique_T, Generic_T)

def set_get_times(Unique_T, Generic_T, params):
    ig = 0
    for t in range(1, Unique_T.shape[0]):
        if Unique_T[t] >= Generic_T[ig]:
            index = np.where(Unique_T[t:] - Generic_T[ig] >= params.DT_obs)
            index = index[0]
            if np.array(index).size > 0:
                Generic_T.append(Unique_T[t + index[0]])
                ig += 1
    return Generic_T

def set_init_concat_dict(keys2, keys3, keys4, nb_lines, Generic_T, temp2):
    concat_dict = {}
    for key in keys2:
        maskCC = np.ones((nb_lines, len(Generic_T)), dtype=bool)
        arrCC = np.full((nb_lines, len(Generic_T)), np.nan)
        concat_dict[key] = np.ma.array(arrCC, mask=maskCC)
    for key in keys3:
        maskCC2 = []
        arrCC2 = []
        ma_arr = []
        for n in range(0, len(temp2['node_t'])):
            maskCC2 = np.ones(len(Generic_T))
            arrCC2 = np.full(len(Generic_T), np.nan)
            ma_arr.append(np.ma.array(arrCC2, mask=maskCC2))
        concat_dict[key] = ma_arr
    for key in keys4:
        concat_dict[key] = []
    concat_dict['reach_id'] = []
    concat_dict['dist_out_all'] = []
    concat_dict['node_length_all'] = []
    return concat_dict

def set_concatenate_spatial_data(concat_dict, temp2, nb_process, params):
    keys_tmp = ['node_id', 'node_length', 'dist_out', 'q_monthly_mean']
    for key in keys_tmp:
        if key == 'q_monthly_mean':
            concat_dict[key].append(np.array(temp2[key][nb_process]))
        elif key == 'dist_out':
            concat_dict[key] = np.hstack((concat_dict[key], temp2[key][nb_process]))
        elif np.array(temp2[key][nb_process]).size > 1:
            concat_dict[key] = np.hstack((concat_dict[key], temp2[key][nb_process]))
        else:
            concat_dict[key] = np.hstack((concat_dict[key], temp2[key][nb_process]))
    return concat_dict

def set_define_prior(temp2, nb_process, concat_dict, option):
    keys_tmp = ['reach_qwbm', 'facc']
    for nb_process_tmp in range(0, len(temp2['node_t'])):
        for key in keys_tmp:
            concat_dict[key] = np.hstack((concat_dict[key], temp2[key][nb_process_tmp]))
    if option == 2:
        concat_dict['reach_qwbm'] = np.nanmean(concat_dict['reach_qwbm'])
    elif option == 3:
        concat_dict['reach_qwbm'] = max(concat_dict['reach_qwbm'])
    masked_data = np.ma.masked_values(np.array([concat_dict['reach_qwbm']]), value=-9999.0)
    concat_dict['reach_qwbm'] = masked_data
    concat_dict['facc'] = np.nanmean(concat_dict['facc'])
    masked_data2 = np.ma.masked_values(np.array([concat_dict['facc']]), value=-9999.0)
    concat_dict['facc'] = masked_data2
    return concat_dict

def set_fill_matrices(temp2, nb_process, Generic_T, concat_dict, keys2, keys3, DI):
    n_obs = -9999
    for n in range(0, temp2['node_t'][nb_process].shape[0]):
        n_mat = n + DI
        index = []
        DT_obs = params.DT_obs
        for t in range(0, temp2['node_t'][nb_process].shape[1]):
            if calc.check_na(temp2['node_t'][nb_process][n][t]):
                continue
            elif temp2['node_t'][nb_process][n][t] > 0.0:
                if n_obs < 0:
                    n_obs = n
                index = np.array(np.where(abs(temp2['node_t'][nb_process][n][t] - Generic_T) <= DT_obs))
                while index.shape[1] != 1:
                    index = np.array(np.where(abs(temp2['node_t'][nb_process][n][t] - Generic_T) <= DT_obs))
                    if index.shape[1] < 1:
                        logging.warning('No time instant was kept!')
                        if DT_obs < 24.0 * 3600.0:
                            DT_obs = DT_obs * 2.0
                    elif index.shape[1] > 1:
                        logging.warning('more than one time instant was kept!')
                        DT_obsprint = DT_obs / 2.0
                for key in keys2:
                    if calc.check_na(temp2[key][nb_process][n, t]):
                        concat_dict[key][n_mat, index] = np.nan
                        concat_dict[key][n_mat].mask[index] = True
                    else:
                        concat_dict[key][n_mat, index] = temp2[key][nb_process][n, t]
                if n == n_obs:
                    for key in keys3:
                        if calc.check_na(temp2[key][nb_process][t]):
                            concat_dict[key][nb_process][index] = np.nan
                            concat_dict[key][nb_process].mask[index] = True
                        else:
                            concat_dict[key][nb_process][index] = temp2[key][nb_process][t]
    return concat_dict

def set_additional_params(concat_dict):
    concat_dict['reach_qsdev'] = 0.0
    q_monthly_mean_stack = np.stack(concat_dict['q_monthly_mean'])
    concat_dict['q_monthly_mean'] = np.nanmean(q_monthly_mean_stack, axis=0)
    masked_data3 = np.ma.masked_values(np.array([concat_dict['q_monthly_mean']]), value=-9999.0)
    concat_dict['q_monthly_mean'] = masked_data3[0]
    flag_dict = {}
    flag_dict['node'] = {}
    flag_dict['reach'] = {}
    flag_dict['nx'] = concat_dict['node_z'].shape[0]
    flag_dict['nt'] = concat_dict['node_z'].shape[1]
    return (concat_dict, flag_dict)

def set_write_output(run_flag, nbr_reach, new_process, output_dir, param_dict, reaches_ids, folder_name, dim_t):
    a5_counter = 0
    for nb_process in range(0, nbr_reach):
        if run_flag:
            if new_process['output']['valid']:
                if np.array(new_process['output']['valid_a5_sets']).size > 0:
                    if new_process['output']['valid_a5_sets'][nb_process]:
                        logging.info(f'sic4dvar_set_run : writing results to output directory {output_dir}')
                        if nbr_reach > 1:
                            write_output(output_dir, param_dict, reaches_ids[nb_process], new_process['output'], a5_counter)
                        if nbr_reach == 1:
                            write_output(output_dir, param_dict, reaches_ids[nb_process], new_process['output'])
                        a5_counter += 1
                        logging.info('Results are valid for algo5 and 3.1')
                else:
                    logging.info(f'sic4dvar_set_run : writing results to output directory {output_dir}')
                    if nbr_reach > 1:
                        write_output(output_dir, param_dict, reaches_ids[nb_process], new_process['output'], -1, folder_name)
                    if nbr_reach == 1:
                        write_output(output_dir, param_dict, reaches_ids[nb_process], new_process['output'], -1, folder_name)
                    logging.info('Results are valid for algo31 only')
            else:
                logging.info('Results are INVALID!')
                if param_dict['aws']:
                    if nbr_reach > 1:
                        write_output(param_dict['output_dir'], param_dict, reaches_ids[nb_process], new_process['output'], -1, folder_name)
                    if nbr_reach == 1:
                        write_output(param_dict['output_dir'], param_dict, reaches_ids[nb_process], new_process['output'], -1, folder_name)
                    logging.info('Force saving for AWS.')
        else:
            logging.info('No data to save.')
            if param_dict['aws']:
                if nbr_reach > 1:
                    write_output(param_dict['output_dir'], param_dict, reaches_ids[nb_process], [], -1, folder_name, max(dim_t))
                if nbr_reach == 1:
                    write_output(param_dict['output_dir'], param_dict, reaches_ids[nb_process], [], -1, folder_name, max(dim_t))
                logging.info('Force saving for AWS.')

def set_run_modules(param_dict, reach_dict, i_set, upper):
    reach_list = [reach['reach_id'] for reach in reach_dict[i_set]]
    reach_list_str = [str(reach) for reach in reach_list]
    if not all((reach_list[i] > reach_list[i + 1] for i in range(len(reach_list) - 1))):
        logging.warning(f'Reach_ids for set are not decreasing : {'_'.join(reach_list_str)}')
    reach_list_str = verify_name_length('_'.join(reach_list_str))
    log_name = 'sic4dvar_' + reach_list_str + '.log'
    log_path = param_dict['log_dir'].joinpath(log_name)
    set_logger(param_dict, log_path)
    logging.info('Processing set %d over %d with %d reaches: %s' % (i_set, upper, len(reach_list), reach_list_str))
    run_flag = True
    processed = []
    temp2 = {}
    concat_dict = {}
    dim_n = []
    dim_t = []
    keys = ['node_w', 'node_z', 'node_s', 'node_dA', 'reach_w', 'reach_z', 'reach_s', 'reach_dA', 'node_t', 'reach_qwbm', 'node_length', 'dist_out', 'reach_id', 'reach_t', 'facc', 'q_monthly_mean', 'node_id']
    keys2 = ['node_w', 'node_z', 'node_s', 'node_dA', 'node_t']
    keys3 = ['reach_w', 'reach_z', 'reach_s', 'reach_dA', 'reach_t']
    keys4 = ['dist_out', 'node_length', 'reach_qwbm', 'facc', 'q_monthly_mean', 'node_id']
    log_name = 'sic4dvar_'
    reaches_ids = []
    if len(reach_dict[i_set]) == 4 and 'reach_id' in reach_dict[i_set]:
        nbr_reach = 1
    else:
        nbr_reach = len(reach_dict[i_set])
    sic4dvar_dict, reaches_ids = set_reaches_init(nbr_reach, reach_dict, param_dict, processed, reaches_ids, i_set)
    nbr_reach = len(reaches_ids)
    logging.info(f'{nbr_reach} reaches to process in current set')
    if len(reaches_ids) == 0:
        logging.info('No valid reach to process in this set ... skip')
    logging.info('Running SIC4DVAR (Algo315) on sets !')
    logging.info('Loading parameters & data.')
    logging.info('Processing reaches: ' + str(reaches_ids))
    temp2, dim_n, dim_t, nb_lines = set_get_input_data(sic4dvar_dict, keys, dim_n, dim_t, temp2, nbr_reach, run_flag)
    run_flag, Unique_T, Generic_T = set_time_check(run_flag, temp2)
    if run_flag:
        Generic_T = set_get_times(Unique_T, Generic_T, params)
        concat_dict = set_init_concat_dict(keys2, keys3, keys4, nb_lines, Generic_T, temp2)
        for nb_process in range(0, len(temp2['node_t'])):
            concat_dict = set_concatenate_spatial_data(concat_dict, temp2, nb_process, params)
        np.set_printoptions(formatter={'float': '{:.2f}'.format})
        SWORD_id_reordered_index, SWORD_id_reordered = reorder_ids_with_indices(concat_dict['node_id'])
        concat_dict['node_id'] = np.array(SWORD_id_reordered)[SWORD_id_reordered_index]
        concat_dict['dist_out'] = np.array(concat_dict['dist_out'])[SWORD_id_reordered_index]
        concat_dict['node_length'] = np.array(concat_dict['node_length'])[SWORD_id_reordered_index]
        for nb_process in range(0, len(temp2['node_t'])):
            concat_dict['reach_id'].append(temp2['reach_id'][nb_process])
            concat_dict = set_define_prior(temp2, nb_process, concat_dict, option=2)
            if nb_process > 0:
                DI += temp2['node_t'][nb_process - 1].shape[0]
            else:
                DI = 0
            concat_dict = set_fill_matrices(temp2, nb_process, Generic_T, concat_dict, keys2, keys3, DI)
        concat_dict, flag_dict = set_additional_params(concat_dict)
        new_process = prepare_params(concat_dict, flag_dict, param_dict, params)
        new_process = sic4dvar_preprocessing(new_process, params)
        new_process, flag_qwbm = sic4dvar_set_prior(new_process)
        new_process = sic4dvar_compute_discharge(new_process, params, flag_qwbm)
    if param_dict['set_folder']:
        json_filename = param_dict['json_path'].stem
        output_dir = param_dict['output_dir'].joinpath(f'{json_filename}')
        folder_name = str(reaches_ids[0]) + '_' + str(reaches_ids[-1])
        output_dir = output_dir.joinpath(folder_name)
    else:
        output_dir = param_dict['output_dir']
        folder_name = 'set'
    set_write_output(run_flag, nbr_reach, new_process, output_dir, param_dict, reaches_ids, folder_name, dim_t)
    close_logger(param_dict)

def accumulate_node_length(dist_out, node_length):
    acc_node_len = np.cumsum(node_length)
    dist_out_0 = scipy.stats.mode(np.abs(acc_node_len - dist_out), axis=None).mode
    node_x = dist_out_0 + acc_node_len
    return node_x

def prepare_reaches(input_data, filtered_data, i):
    reach_dict = {}
    i = 1
    reach_info_list = []
    reach_node_x = []
    reach_node_x = accumulate_node_length(input_data['dist_out_all'][i], input_data['node_length_all'][i])
    reach_dict['node_z'] = np.empty((len(reach_node_x), len(input_data['separate_reach_t'][i])))
    reach_dict['node_z'][:] = np.nan
    reach_dict['node_t'] = np.empty((len(reach_node_x), len(input_data['separate_reach_t'][i])))
    reach_dict['node_t'][:] = np.nan
    for n in range(0, reach_dict['node_t'].shape[0]):
        reach_dict['node_t'][n] = input_data['separate_reach_t'][i]
    for n in range(0, filtered_data['node_t'].shape[0]):
        for t1 in range(0, filtered_data['node_t'].shape[1]):
            if calc.check_na(filtered_data['node_t'][n, t1]):
                pass
            else:
                t_value = filtered_data['node_t'][n, t1]

def densification_run(input_data, filtered_data):
    for i in range(0, len(input_data['separate_reach_t'])):
        pass
        prepare_reaches(input_data, filtered_data, i)