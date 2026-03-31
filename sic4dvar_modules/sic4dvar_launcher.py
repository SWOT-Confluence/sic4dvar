import logging
from copy import deepcopy
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brent, minimize_scalar
from pathlib import Path
import sic4dvar_params as params
from lib.lib_dates import daynum_to_date
from lib.lib_verif import verify_name_length
from sic4dvar_algos.algo3 import algo3, modified_manning_integrated
from sic4dvar_algos.algo5 import algo5
from sic4dvar_functions.sic4dvar_gnuplot_save import gnuplot_save_cs, gnuplot_save_q, gnuplot_save_var, gnuplot_save_cs, gnuplot_save_var
from sic4dvar_functions.sic4dvar_helper_functions import build_output_q_masked_array, build_output_2d_masked_array, compute_mean_var_from_2D_array, get_external_friction_data, grad_variance, interp_pdf_tables
from sic4dvar_modules.sic4dvar_compute_slope_and_bathymetry import call_func_APR, compute_bathymetry, compute_slope, compute_z_bed
from sic4dvar_modules.sic4dvar_create_filtered_arrays import create_filtered_arrays
from sic4dvar_modules.sic4dvar_extrapolation import Extrapolation
from sic4dvar_modules.sic4dvar_filtering import filter_based_on_config, remove_unuseable_nodes
from sic4dvar_modules.sic4dvar_launch_algo5 import launch_algo5
from sic4dvar_modules.sic4dvar_qwbm_replace import replace_prior

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
        sic4dvar_dict['output']['node_z'] = deepcopy(sic4dvar_dict['input_data']['node_z'])
        sic4dvar_dict['output']['node_w'] = deepcopy(sic4dvar_dict['input_data']['node_w'])
        sic4dvar_dict['output']['node_t'] = deepcopy(sic4dvar_dict['input_data']['node_t'])
        logging.info(f'After extrapolation: {valid_z} valid wse and {valid_w} valid width')
    logging.info('Finished extrapolating the data.')
    if sic4dvar_dict['data_is_useable']:
        sic4dvar_dict['data_is_useable'], sic4dvar_dict['observed_nodes'], sic4dvar_dict['list_to_keep'], sic4dvar_dict['removed_indices'] = remove_unuseable_nodes(sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['node_w'])
        sic4dvar_dict['output']['observed_nodes'] = sic4dvar_dict['observed_nodes']
    if sic4dvar_dict['data_is_useable'] and np.array(sic4dvar_dict['list_to_keep']).size > 0:
        sic4dvar_dict = create_filtered_arrays(sic4dvar_dict)
    else:
        logging.debug('Not enough usable nodes to process reach.')
        sic4dvar_dict['valid'] = 0
        sic4dvar_dict['output']['stopped_stage'] = 'densification'
        sic4dvar_dict['output']['node_z'] = sic4dvar_dict['input_data']['node_z']
        sic4dvar_dict['output']['node_w'] = sic4dvar_dict['input_data']['node_w']
        sic4dvar_dict['output']['node_t'] = sic4dvar_dict['input_data']['node_t']
    logging.info('Finished creating filtered data with useable nodes.')
    return sic4dvar_dict

def sic4dvar_set_prior(sic4dvar_dict):
    logging.info(f'Qwbm value: {sic4dvar_dict['input_data']['reach_qwbm']}')
    sic4dvar_dict, flag_qwbm = replace_prior(sic4dvar_dict)
    return (sic4dvar_dict, flag_qwbm)

def synchronize_records(sic4dvar_dict, params):
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
    return (last_time_instant, time_indexes_to_keep)

def sic4dvar_compute_discharge(sic4dvar_dict, params, flag_qwbm):
    if sic4dvar_dict['data_is_useable'] and flag_qwbm:
        logging.info('INFO: preparing to estimate discharge.')
        if params.densification:
            pass
        if not params.densification:
            last_time_instant, time_indexes_to_keep = synchronize_records(sic4dvar_dict, params)
            sic4dvar_dict['list_to_keep'] = sic4dvar_dict['list_to_keep'][time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['node_z'] = sic4dvar_dict['filtered_data']['node_z'][:, time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['node_w'] = sic4dvar_dict['filtered_data']['node_w'][:, time_indexes_to_keep]
            sic4dvar_dict['filtered_data']['reach_t'] = sic4dvar_dict['filtered_data']['reach_t'][time_indexes_to_keep]
            if sic4dvar_dict['param_dict']['run_type'] == 'seq':
                sic4dvar_dict['filtered_data']['reach_s'] = sic4dvar_dict['filtered_data']['reach_s'][time_indexes_to_keep]
            else:
                logging.error("Reach slope from SWOT is not available in set mode, can't keep it for algo3. Check your config file.")
            cort_slope = 0.0
            cort_tmp = []
            for t in range(0, len(sic4dvar_dict['filtered_data']['reach_t'])):
                cort_tmp.append(sic4dvar_dict['filtered_data']['reach_t'][t])
            sic4dvar_dict['cort_slope'] = np.mean(np.diff(cort_tmp))
            SLOPEM1, sic4dvar_dict = compute_slope(sic4dvar_dict, params)
            if sic4dvar_dict['reverse_order']:
                SLOPEM1 = -SLOPEM1
            sic4dvar_dict, bathymetry_array, apr_array = compute_bathymetry(sic4dvar_dict, params, SLOPEM1)
            sic4dvar_dict['output']['width'] = sic4dvar_dict['input_data']['node_xr']
            sic4dvar_dict['output']['elevation'] = sic4dvar_dict['input_data']['node_yr']
            sic4dvar_dict['output']['stopped_stage'] = 'bathymetry'
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
            if params.use_friction_external_file:
                friction_file = getattr(params, 'friction_external_file', '')
                if friction_file == '':
                    logging.warning('Friction external file option is True but no file provided.')
                    friction_values = None
                    friction_dates = None
                    friction_t = None
                else:
                    friction_values, friction_dates, friction_t = get_external_friction_data(friction_file, sic4dvar_dict['input_data']['reach_id'])
                friction_mean = 0
                time_scaling = 0
                for t in range(1, len(friction_t)):
                    friction_mean += (friction_values[t] + friction_values[t - 1]) / 2 * (friction_t[t] - friction_t[t - 1])
                    time_scaling += np.array(friction_t[t] - friction_t[t - 1])
                if time_scaling > 0:
                    friction_mean = friction_mean / time_scaling
                else:
                    logging.warning("Time scaling for friction external data is 0, can't compute mean friction.")
                    friction_mean = None
                if friction_mean is not None:
                    friction_mean = 1 / friction_mean
                sic4dvar_dict['input_data']['friction_value'] = friction_mean
            else:
                sic4dvar_dict['input_data']['friction_value'] = None
            sic4dvar_dict['output']['q_algo31'], sic4dvar_dict['output']['valid'], sic4dvar_dict['reliability'], sic4dvar_dict['output']['Kmi_acc'], sic4dvar_dict['output']['Zb_acc'] = algo3(apr_array, sic4dvar_dict['input_data']['reach_qwbm'], params, SLOPEM1, sic4dvar_dict['output']['valid'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['node_z_ini'], sic4dvar_dict['filtered_data']['node_x'], sic4dvar_dict['last_node_for_integral'], bathymetry_array, sic4dvar_dict['param_dict'], sic4dvar_dict['input_data']['reach_id'], sic4dvar_dict['filtered_data']['reach_t'], bb=sic4dvar_dict['bb'], reliability=sic4dvar_dict['reliability'], Qsdev=sic4dvar_dict['input_data']['reach_qsdev'], last_time_instant=last_time_instant, input_data=sic4dvar_dict['input_data'], time_indexes_to_keep=time_indexes_to_keep, friction_value=sic4dvar_dict['input_data']['friction_value'])
            sic4dvar_dict['output']['z_bed'] = compute_z_bed(apr_array['node_w_simp'], sic4dvar_dict['filtered_data']['node_z'], bathymetry_array['node_xr'], bathymetry_array['node_yr'], sic4dvar_dict['output']['Zb_acc'])
            sic4dvar_dict['output']['apr_array'] = apr_array
            if sic4dvar_dict['param_dict']['gnuplot_saving']:
                reach_id = str(reach_id)
                reach_id = verify_name_length(reach_id)
                node_x = sic4dvar_dict['filtered_data']['node_x']
                test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                nodes2 = (node_x - node_x[0]) / 1000
                times2 = test_t_array / 3600 / 24
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                gnuplot_save_q(sic4dvar_dict['output']['q_algo31'], times2, output_path.joinpath('q31_out'))
            if params.experimental_discharge_ML:
                ZB_update = sic4dvar_dict['output']['Zb_acc']
                Zb = np.zeros(len(sic4dvar_dict['filtered_data']['node_z']))
                Zb2 = params.algo_bounds[1][1]
                Wmin = 10000
                Wmean = []
                for i in range(len(apr_array['node_w_simp'])):
                    Wmean.append(min(apr_array['node_w_simp'][i]))
                    if np.min(apr_array['node_w_simp'][i]) < Wmin:
                        Wmin = np.min(apr_array['node_w_simp'][i])
                Wmean = np.average(Wmean)
                discharge_estimate = sic4dvar_dict['output']['q_algo31']
                time_instants = np.ma.masked_values(np.array(sic4dvar_dict['filtered_data']['reach_t']), value=-9999.0)
                if discharge_estimate.mask.size == 1:
                    valid_idx1 = np.where(discharge_estimate)[0]
                else:
                    valid_idx1 = np.where(np.isfinite(discharge_estimate))
                if time_instants.mask.size == 1:
                    valid_idx2 = np.where(time_instants)[0]
                else:
                    valid_idx2 = np.where(np.isfinite(time_instants))
                valid_idx3 = np.intersect1d(valid_idx1, valid_idx2)
                if discharge_estimate.mask.size == 1:
                    valid_idx1 = np.where(discharge_estimate)[0]
                else:
                    valid_idx1 = np.where(discharge_estimate.mask == False)
                if time_instants.mask.size == 1:
                    valid_idx1 = np.where(time_instants)[0]
                else:
                    valid_idx2 = np.where(time_instants.mask == False)
                valid_idx4 = np.intersect1d(valid_idx1, valid_idx2)
                valid_idx = np.intersect1d(valid_idx3, valid_idx4)
                time_instants = time_instants / 86400
                sic4dvar_date = daynum_to_date(time_instants[valid_idx], '2000-01-01')
                sic4dvar_df = pd.DataFrame({'sic4dvar_q': discharge_estimate[valid_idx], 'sic4dvar_date': sic4dvar_date})
                sic4dvar_df['date_only'] = pd.to_datetime(sic4dvar_df['sic4dvar_date'], format='%Y-%m-%d').dt.date
                sic4dvar_df['sic4dvar_qt'] = time_instants[valid_idx]
                qa31_t = deepcopy(np.array(sic4dvar_df['sic4dvar_qt']))
                if not params.force_specific_dates:
                    start_date = sic4dvar_df['date_only'].min()
                    end_date = sic4dvar_df['date_only'].max()
                else:
                    start_date = sic4dvar_df['date_only'].min()
                    end_date = sic4dvar_df['date_only'].max()
                ML_df = sic4dvar_dict['input_data']['ML_df']
                filtered_ML_df = ML_df[(ML_df['date_only'] >= start_date) & (ML_df['date_only'] <= end_date)]
                start_date_ML = filtered_ML_df['date_only'].min()
                end_date_ML = filtered_ML_df['date_only'].max()
                times_ML = filtered_ML_df[(filtered_ML_df['date_only'] >= start_date) & (filtered_ML_df['date_only'] <= end_date)]
                times_sic = sic4dvar_df[(sic4dvar_df['date_only'] >= start_date) & (sic4dvar_df['date_only'] <= end_date)]
                time_system = times_sic['sic4dvar_qt']
                if len(time_system) < 2:
                    logging.info("Time system size < 2, can't compute integrated estimated mean.")
                    return {}
                q_ref = []
                q_est = []
                t_ref = []
                q_ref_ML = []
                for t in range(0, len(time_system)):
                    q_ref_ML.append(interp_pdf_tables(len(filtered_ML_df['station_q']) - 1, time_system.iloc[t], np.array(filtered_ML_df['station_qt']), np.array(filtered_ML_df['station_q'])))
                    q_est.append(sic4dvar_df['sic4dvar_q'][t])
                    t_ref.append(time_system.iloc[t])
                ZM = 1.0
                ZM_opt_array = []
                for t in range(0, len(t_ref)):
                    Q_REF = q_ref_ML[t]

                    def objective_function_brent(ZM):
                        Q_EST, _, _ = modified_manning_integrated(bathymetry_array['node_xr'], bathymetry_array['node_yr'], apr_array['node_a'], apr_array['node_p'], Wmean, sic4dvar_dict['last_node_for_integral'], Zb2, ZB_update, Zb, t, sic4dvar_dict['filtered_data']['node_x'], SLOPEM1, ZM=ZM, KMI=sic4dvar_dict['output']['Kmi_acc'], option_recompute_area=True, sic4dvar_dict=sic4dvar_dict)
                        return (Q_REF - Q_EST) ** 2
                    ZM_opt = minimize_scalar(objective_function_brent, bounds=(0.33, 3.0), method='bounded', options={'maxiter': 10}).x
                    ZM_opt_array.append(ZM_opt)
                ZM_opt_array_new = np.array([float(x) for x in ZM_opt_array])
                if sic4dvar_dict['param_dict']['gnuplot_saving']:
                    reach_id = str(reach_id)
                    reach_id = verify_name_length(reach_id)
                    node_x = sic4dvar_dict['filtered_data']['node_x']
                    test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                    nodes2 = (node_x - node_x[0]) / 1000
                    times2 = test_t_array / 3600 / 24
                    output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                    if not Path(output_path).is_dir():
                        Path(output_path).mkdir(parents=True, exist_ok=True)
                    gnuplot_save_q(ZM_opt_array_new, times2, output_path.joinpath('ZM'))
                sigm0 = 0.2 * np.sqrt(100 / Wmean)
                ss = 0.0
                ss1 = 0.0
                for t in range(0, len(t_ref) - 1):
                    for n in range(0, len(sic4dvar_dict['filtered_data']['node_z'])):
                        ss += (((ZM_opt_array_new[t] - 1.0) * sic4dvar_dict['filtered_data']['node_z'][n, t]) ** 2 + ((ZM_opt_array_new[t + 1] - 1.0) * sic4dvar_dict['filtered_data']['node_z'][n, t + 1]) ** 2 / 2.0) * (t_ref[t + 1] - t_ref[t])
                        ss1 += t_ref[t + 1] - t_ref[t]
                alph0 = np.sqrt(ss1 * sigm0 ** 2 / ss)
                ZMS1 = np.zeros(len(t_ref))
                for t in range(0, len(t_ref)):
                    ss = 0.0
                    ss1 = 0.0
                    for n in range(0, len(sic4dvar_dict['filtered_data']['node_z'])):
                        ss += sic4dvar_dict['filtered_data']['node_z'][n, t]
                        ss1 += 1.0
                    ZMS1[t] = ss / ss1
                ss = 0.0
                ss1 = 0.0
                for t in range(0, len(t_ref) - 1):
                    ss += (((ZM_opt_array_new[t] - 1.0) * ZMS1[t]) ** 2 + ((ZM_opt_array_new[t + 1] - 1.0) * ZMS1[t + 1]) ** 2) / 2.0 * (t_ref[t + 1] - t_ref[t])
                    ss1 += t_ref[t + 1] - t_ref[t]
                alph1 = np.sqrt(ss1 * sigm0 ** 2 / ss)
                if alph1 > 1.0:
                    alph1 = 1.0
                sic4dvar_dict['output']['alph1'] = alph1
                for t in range(0, len(t_ref)):
                    ss = ZM_opt_array_new[t]
                    ZMS1[t] = 1.0 + alph1 * (ss - 1.0)
                ZM_opt_array_new = deepcopy(ZMS1)
                if sic4dvar_dict['param_dict']['gnuplot_saving']:
                    reach_id = str(reach_id)
                    reach_id = verify_name_length(reach_id)
                    node_x = sic4dvar_dict['filtered_data']['node_x']
                    test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                    nodes2 = (node_x - node_x[0]) / 1000
                    times2 = test_t_array / 3600 / 24
                    output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                    if not Path(output_path).is_dir():
                        Path(output_path).mkdir(parents=True, exist_ok=True)
                    gnuplot_save_q(ZM_opt_array_new, times2, output_path.joinpath('ZM2'))
                node_a2, node_p2, node_r2, node_w_simp2, _ = call_func_APR(sic4dvar_dict['filtered_data']['node_w'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_xr'], sic4dvar_dict['input_data']['node_yr'], params, sic4dvar_dict['param_dict'], coeff_array=ZM_opt_array_new)
                apr_array['node_a'] = deepcopy(node_a2)
                apr_array['node_p'] = deepcopy(node_p2)
                apr_array['node_r'] = deepcopy(node_r2)
                apr_array['node_w_simp'] = deepcopy(node_w_simp2)
                Q_EST_ARRAY = []
                if params.experimental_run_MMI:
                    for t in range(0, len(t_ref)):
                        ZM = ZM_opt_array_new[t]
                        Q_EST, _, _ = modified_manning_integrated(bathymetry_array['node_xr'], bathymetry_array['node_yr'], apr_array['node_a'], apr_array['node_p'], Wmean, sic4dvar_dict['last_node_for_integral'], Zb2, ZB_update, Zb, t, sic4dvar_dict['filtered_data']['node_x'], SLOPEM1, ZM=ZM, KMI=sic4dvar_dict['output']['Kmi_acc'], option_recompute_area=False, sic4dvar_dict=sic4dvar_dict)
                        Q_EST_ARRAY.append(Q_EST[0])
                else:
                    Q_EST_ARRAY, sic4dvar_dict['output']['valid'], sic4dvar_dict['reliability'], sic4dvar_dict['output']['Kmi_acc'], sic4dvar_dict['output']['Zb_acc'] = algo3(apr_array, sic4dvar_dict['input_data']['reach_qwbm'], params, SLOPEM1, sic4dvar_dict['output']['valid'], sic4dvar_dict['filtered_data']['node_z'], sic4dvar_dict['input_data']['node_z'], sic4dvar_dict['input_data']['node_z_ini'], sic4dvar_dict['filtered_data']['node_x'], sic4dvar_dict['last_node_for_integral'], bathymetry_array, sic4dvar_dict['param_dict'], sic4dvar_dict['input_data']['reach_id'], sic4dvar_dict['filtered_data']['reach_t'], bb=sic4dvar_dict['bb'], reliability=sic4dvar_dict['reliability'], Qsdev=sic4dvar_dict['input_data']['reach_qsdev'], last_time_instant=last_time_instant, input_data=sic4dvar_dict['input_data'], time_indexes_to_keep=time_indexes_to_keep, ZM_array=ZM_opt_array_new)
                Q_EST_ARRAY = np.array(Q_EST_ARRAY)
                sic4dvar_dict['output']['q_algo31'] = deepcopy(Q_EST_ARRAY)
                if True:
                    ss = 0.0
                    ss1 = 0.0
                    for t in range(0, len(Q_EST_ARRAY) - 1):
                        ss += (sic4dvar_dict['output']['q_algo31'][t + 1] + sic4dvar_dict['output']['q_algo31'][t]) / 2 * (t_ref[t + 1] - t_ref[t])
                        ss1 += t_ref[t + 1] - t_ref[t]
                    q_mean = ss / ss1
                    sic4dvar_dict['output']['q_algo31'] = deepcopy(sic4dvar_dict['output']['q_algo31'] * (sic4dvar_dict['input_data']['reach_qwbm'] / q_mean))
            if sic4dvar_dict['param_dict']['gnuplot_saving']:
                reach_id = str(reach_id)
                reach_id = verify_name_length(reach_id)
                node_x = sic4dvar_dict['filtered_data']['node_x']
                test_t_array = sic4dvar_dict['filtered_data']['reach_t']
                nodes2 = (node_x - node_x[0]) / 1000
                times2 = test_t_array / 3600 / 24
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                gnuplot_save_q(sic4dvar_dict['output']['q_algo31'], times2, output_path.joinpath('q31_out_2'))
                output_path = sic4dvar_dict['param_dict']['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
            logging.info(f'Finished discharge estimation.')
        Q31 = sic4dvar_dict['output']['q_algo31']
        if np.array(Q31).size == 0:
            sic4dvar_dict['output']['valid'] = 0
            return sic4dvar_dict
        sic4dvar_dict['output']['reach_length'] = sic4dvar_dict['input_data']['node_x'].max() - sic4dvar_dict['input_data']['node_x'].min()
        reach_slope = sic4dvar_dict['output']['SLOPEM1'] / sic4dvar_dict['output']['reach_length']
        sic4dvar_dict['output']['reach_slope'] = np.full(sic4dvar_dict['input_data']['reach_t'].size, np.nan)
        sic4dvar_dict['output']['reach_slope'][sic4dvar_dict['list_to_keep']] = reach_slope
        sic4dvar_dict['output']['q_algo31'] = build_output_q_masked_array(sic4dvar_dict, 'q_algo31')
        sic4dvar_dict['output']['time'] = build_output_q_masked_array(sic4dvar_dict, 'time')
        if 'apr_array' in sic4dvar_dict['output']:
            sic4dvar_dict['output']['apr_array']['node_a'] = build_output_2d_masked_array(sic4dvar_dict, sic4dvar_dict['output']['apr_array']['node_a'])
            sic4dvar_dict['output']['apr_array']['node_r'] = build_output_2d_masked_array(sic4dvar_dict, sic4dvar_dict['output']['apr_array']['node_r'])
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
    return sic4dvar_dict