import datetime
import logging
from copy import deepcopy
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import sic4dvar_params as params
from lib.lib_dates import seconds_to_date_old
from lib.lib_verif import check_na
from sic4dvar_functions import sic4dvar_calculations as calc

def prepare_params(input_data, flag_dict, param_dict, params):
    sic4dvar_dict = {'input_data': input_data, 'flag_dict': flag_dict, 'param_dict': param_dict, 'filtered_data': {}, 'output': {}, 'algo5_results': {}, 'bb': 9999.0, 'reliability': 'valid'}
    if sic4dvar_dict['param_dict']['run_type'] == 'set':
        test_t_array = np.ones(len(sic4dvar_dict['input_data']['node_t'][0]))
        for t in range(0, len(sic4dvar_dict['input_data']['node_t'][0])):
            test_t_array[t] = np.mean(sic4dvar_dict['input_data']['node_t'][:, t])
        sic4dvar_dict['input_data']['separate_reach_t'] = deepcopy(sic4dvar_dict['input_data']['reach_t'])
        sic4dvar_dict['input_data']['reach_t'] = deepcopy(test_t_array)
    sic4dvar_dict['output']['valid'] = 1
    sic4dvar_dict['output']['valid_a5'] = 1
    sic4dvar_dict['output']['valid_a5_sets'] = []
    if params.qsdev_activate:
        sic4dvar_dict['output']['q_std'] = input_data['reach_qsdev']
    sic4dvar_dict['output']['node_id'] = input_data['node_id']
    if params.force_specific_dates:
        logging.debug(f'reach_t: {sic4dvar_dict['input_data']['reach_t']}')
        dates = []
        for t in range(0, len(sic4dvar_dict['input_data']['reach_t'])):
            if check_na(sic4dvar_dict['input_data']['reach_t'][t]):
                dates.append(datetime.datetime(1, 1, 1))
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
            sic4dvar_dict['flag_dict']['nt'] = len(indexes)
            logging.debug(f'node_z shape: {sic4dvar_dict['input_data']['node_z'].shape}, reach_z shape: {sic4dvar_dict['input_data']['reach_z'].shape}')
    duplicate_t = np.ones(sic4dvar_dict['input_data']['node_t'].shape)
    for n in range(0, sic4dvar_dict['input_data']['node_t'].shape[0]):
        duplicate_t[n, :] = sic4dvar_dict['input_data']['reach_t']
    sic4dvar_dict['output']['reach_t_duplicate'] = deepcopy(duplicate_t)
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
        sic4dvar_dict['output']['time'] = np.array([])
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
        sic4dvar_dict['output']['time'] = np.array([])
        sic4dvar_dict['removed_nodes_ind'] = np.zeros(0)
        sic4dvar_dict['indices'] = []
    if not params.node_length:
        sic4dvar_dict['input_data']['node_x'] = sic4dvar_dict['input_data']['dist_out']
    if params.node_length:
        dist_out_1 = []
        dist_out_1.append(float(sic4dvar_dict['input_data']['dist_out'][0]))
        if not params.start_from_downstream:
            direction = -1
        else:
            direction = 1
        for i in range(1, len(sic4dvar_dict['input_data']['node_length'])):
            dist_out_1.append(dist_out_1[i - 1] + direction * sic4dvar_dict['input_data']['node_length'][i - 1])
        dist_out_1 = dist_out_1[0] - np.array(dist_out_1)
        sic4dvar_dict['input_data']['node_x'] = np.array(dist_out_1)
    nb_valid_z = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_z'].filled(np.nan)))
    nb_valid_w = np.sum(~np.isnan(sic4dvar_dict['input_data']['node_w'].filled(np.nan)))
    if nb_valid_z == 0 or nb_valid_w == 0:
        logging.debug(f'Not enough nodes with data to run SIC4DVAR: {nb_valid_z} valid WSE and {nb_valid_w} valid widths.')
        sic4dvar_dict['output']['valid'] = 0
        sic4dvar_dict['output']['stopped_stage'] = 'input_data_check'
        sic4dvar_dict['output']['node_z'] = sic4dvar_dict['input_data']['node_z']
        sic4dvar_dict['output']['node_w'] = sic4dvar_dict['input_data']['node_w']
    return sic4dvar_dict