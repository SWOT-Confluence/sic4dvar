import traceback
import logging
import datetime
import os
import numpy as np
import json
from sic4dvar_functions import sic4dvar_calculations as calc
from sic4dvar_functions.sic4dvar_helper_functions import get_input_data, get_reach_dataset, enable_prints, write_output
from sic4dvar_functions.sic4dvar_calculations import verify_name_length
from sic4dvar_algos.sic4dvar_algorithms import algorithms
import pandas as pd
import numpy as np
from lib.lib_log import set_logger, close_logger, append_to_principal_log, call_error_message
from lib.lib_dates import daynum_to_date
from lib.lib_indicators import compute_all_indicators_from_predict_true
import sic4dvar_params as params
import multiprocessing as mp
import traceback
from tqdm import tqdm

def worker_fn(j, param_dict, queue):
    try:
        reach_data = get_reach_dataset(param_dict, j)
        result = seq_run(param_dict, reach_data)
        queue.put((j, result, None))
    except Exception as e:
        error_info = traceback.format_exc()
        queue.put((j, None, error_info))

def execute_method(args):
    instance, method_name, arg = args
    method = getattr(instance, method_name)
    return method(arg)

def seq_run(param_dict, reach_dict):
    try:
        log_path = param_dict['log_dir'].joinpath('sic4dvar_' + str(reach_dict['reach_id']) + '.log')
        set_logger(logging.INFO, log_path)
        logging.info(f'Run reach {reach_dict['reach_id']}')
        logging.info('Reach infos : ')
        for k, val in reach_dict.items():
            logging.info('  %s : %s' % (k, val))
        logging.info('Running SIC4DVAR sequentially on reaches one by one.')
        logging.info('Processing reach: ' + str(reach_dict['reach_id']))
        logging.info('Running reach %d)' % reach_dict['reach_id'])
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
                    station_df = pd.DataFrame({'station_q': input_data['station_q'], 'station_date': input_data['station_date']})
                valid_idx1 = np.where(np.isfinite(processed.output['q_algo31']))
                valid_idx2 = np.where(np.isfinite(processed.output['time']))
                valid_idx = np.intersect1d(valid_idx1, valid_idx2)
                sic4dvar_date = daynum_to_date(processed.output['time'][valid_idx], '2000-01-01')
                sic4dvar_df = pd.DataFrame({'sic4dvar_q': processed.output['q_algo31'][valid_idx], 'sic4dvar_date': sic4dvar_date})
                data_df = sic4dvar_df.merge(station_df, how='inner', left_on='sic4dvar_date', right_on='station_date')
                if len(data_df) > 1:
                    indicators = {'reach_id': reach_dict['reach_id'], 'nb_dates': len(data_df), 'width': input_data['reach_w_mean'][0], 'reach_length': input_data['reach_length'][0], 'slope': input_data['reach_slope'][0], 'obs_filtering_ratio': processed.output['obs_filtering_ratio']}
                    indicators.update(compute_all_indicators_from_predict_true(np.array(data_df['station_q']).astype('float'), np.array(data_df['sic4dvar_q']).astype('float')))
                    logging.info(f'Indicators name : {';'.join(list(indicators.keys()))}')
                    logging.info(f'Indicators values : {';'.join([str(e) for e in indicators.values()])}')
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
    except Exception as e:
        traceback.print_exception(e)
        logging.error(f'Error computing reach_id {reach_dict['reach_id']}: {e} ... skip')
        if not param_dict['safe_mode']:
            return -1
    close_logger()

def run_parallel(param_dict, down, upper, max_procs=4):
    manager = mp.Manager()
    queue = manager.Queue()
    results = {}
    total_jobs = upper - down
    active_procs = []
    with tqdm(total=total_jobs, desc='Processing', unit='job') as pbar:
        for j in range(down, upper):
            p = mp.Process(target=worker_fn, args=(j, param_dict, queue))
            p.start()
            active_procs.append(p)
            if len(active_procs) >= max_procs:
                for proc in active_procs:
                    proc.join()
                active_procs.clear()
            while not queue.empty():
                print(queue.get())
                j_done, result, err = queue.get()
                if err:
                    print(f'\n❌ Job {j_done} failed:\n{err}')
                    if not param_dict['safe_mode']:
                        return -1
                else:
                    results[j_done] = result
                pbar.update(1)
        for proc in active_procs:
            proc.join()
        while not queue.empty():
            j_done, result, err = queue.get()
            if err:
                print(f'\n❌ Job {j_done} failed:\n{err}')
                if not param_dict['safe_mode']:
                    return -1
            else:
                results[j_done] = result
            pbar.update(1)
    return results

def sic4dvar_seq_run(param_dict):
    append_to_principal_log(param_dict, 'Running SIC4DVAR sequentially on json_path one by one')
    t0 = datetime.datetime.utcnow()
    with open(param_dict['json_path']) as jsonfile:
        dataj = json.load(jsonfile)
        append_to_principal_log(param_dict, f'Number of reaches is : {len(dataj)}')
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
            upper = len(dataj)
    else:
        down = 0
        upper = len(dataj)
    append_to_principal_log(param_dict, f'down,upper: {down} {upper}')
    if param_dict['flag_parallel']:
        results = run_parallel(param_dict, down, upper, max_procs=params.max_cores)
    else:
        for j in range(down, upper):
            seq_run(param_dict, get_reach_dataset(param_dict, j))
    t1 = datetime.datetime.utcnow()
    enable_prints()

def sic4dvar_set_run(param_dict):
    reach_dict = get_reach_dataset(param_dict)
    append_to_principal_log(param_dict, 'Running SIC4DVAR sequentially on sets')
    append_to_principal_log(param_dict, 'Number of sets is : %d ' % len(reach_dict))
    t0 = datetime.datetime.utcnow()
    if param_dict['aws'] == True and os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'):
        down = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX'))
        upper = down + 1
    else:
        down = 0
        upper = len(reach_dict)
    for i_set in range(down, upper):
        reach_dict_list = get_reach_dataset(param_dict, i_set)
        for j_reach, reach_dict in enumerate(reach_dict_list):
            if str(reach_dict['reach_id'])[-1] == '1':
                para_dict_tmp = param_dict.copy()
                para_dict_tmp['run_type'] = 'seq'
                seq_run(para_dict_tmp, reach_dict)
            else:
                logging.error(call_error_message('105').format(reach_id=reach_dict['reach_id']))
                append_to_principal_log(param_dict, f'status of reach {i_set} reach_id {reach_dict['reach_id']} : 101')
                return 0
        append_to_principal_log(param_dict, f'status of set {i_set} reach_id {reach_dict_list[0]['reach_id']} to {reach_dict_list[-1]['reach_id']} : 2')
    return 0
    for i_set in range(down, upper):
        t_reach0 = datetime.datetime.utcnow()
        reach_list = [str(reach['reach_id']) for reach in reach_dict[i_set]]
        reach_list_str = verify_name_length('_'.join(reach_list))
        log_name = 'sic4dvar_' + reach_list_str + '.log'
        log_path = param_dict['log_dir'].joinpath(log_name)
        set_logger(logging.INFO, log_path)
        logging.info('Processing set %d over %d with %d reaches: %s' % (i_set, upper, len(reach_list), ' '.join(reach_list)))
        run_flag = True
        processed = []
        temp2 = {}
        concat_dict = {}
        dim_n = []
        dim_t = []
        keys = ['node_w', 'node_z', 'node_s', 'node_dA', 'reach_w', 'reach_z', 'reach_s', 'reach_dA', 'node_t', 'reach_qwbm', 'node_length', 'dist_out', 'reach_id', 'reach_t', 'facc', 'q_monthly_mean']
        log_name = 'sic4dvar_'
        reaches_ids = []
        if len(reach_dict[i_set]) == 4 and 'reach_id' in reach_dict[i_set]:
            nbr_reach = 1
        else:
            nbr_reach = len(reach_dict[i_set])
        logging.info(f'{nbr_reach} reaches to process in current reach')
        for j in range(0, nbr_reach):
            if nbr_reach > 1:
                if str(reach_dict[i_set][j]['reach_id'])[-1] == '1':
                    input_data, flag_dict = get_input_data(param_dict, reach_dict[i_set][j])
                    processed.append(algorithms(input_data, flag_dict, param_dict))
                    reaches_ids.append(reach_dict[i_set][j]['reach_id'])
                else:
                    logging.info(f'Reach {reach_dict[i_set][j]['reach_id']} not processed because reach type != 1')
            elif 'reach_id' in reach_dict[i_set]:
                if str(reach_dict[i_set]['reach_id'])[-1] == '1':
                    input_data, flag_dict = get_input_data(param_dict, reach_dict[i_set])
                    processed.append(algorithms(input_data, flag_dict, param_dict))
                    log_name = log_name + str(reach_dict[i_set]['reach_id']) + '_'
                    reaches_ids.append(reach_dict[i_set]['reach_id'])
                else:
                    pass
            elif 'reach_id' in reach_dict[i_set][0]:
                if str(reach_dict[i_set][0]['reach_id'])[-1] == '1':
                    input_data, flag_dict = get_input_data(param_dict, reach_dict[i_set][0])
                    processed.append(algorithms(input_data, flag_dict, param_dict))
                    log_name = log_name + str(reach_dict[i_set][0]['reach_id']) + '_'
                    reaches_ids.append(reach_dict[i_set][0]['reach_id'])
                else:
                    pass
        if len(reaches_ids) == 0:
            logging.info('No valid reach to process in this set ... skip')
            continue
        logging.info('Running SIC4DVAR (Algo315)!')
        logging.info('Loading parameters & data.')
        logging.info('Processing reaches: ' + str(reaches_ids))
        nb_lines = 0
        if np.array(processed).size == 0:
            run_flag = False
        get_dim_t = -1
        for i in range(0, nbr_reach):
            if np.array(processed).size == 0:
                break
            elif 'reach_z' in processed[i].input_data:
                get_dim_t = processed[i].input_data['reach_z'].shape[0]
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
                    if np.array(processed[i].input_data).size == 0:
                        pass
                    if 'valid_min_z' in processed[i].input_data:
                        params.valid_min_z = processed[i].input_data['valid_min_z']
                        params.valid_min_dA = processed[i].input_data['valid_min_dA']
                    else:
                        pass
                    if key in processed[i].input_data:
                        if key == 'node_t':
                            dim_n.append(processed[i].input_data[key].shape[0])
                            dim_t.append(processed[i].input_data[key].shape[1])
                            nb_lines += processed[i].input_data[key].shape[0]
                        temp.append(processed[i].input_data[key])
                    else:
                        if key == 'node_t':
                            dim_n.append(processed[i].input_data['node_length'].shape[0])
                            nb_lines += processed[i].input_data['node_length'].shape[0]
                        empty_array = np.full((processed[i].input_data['node_length'].shape[0], get_dim_t), np.nan)
                        temp.append(np.array(empty_array))
                temp2[key] = temp
        All_T = []
        Unique_T = []
        tmp3 = []
        Generic_T = []
        if run_flag:
            for nb_process in range(0, len(temp2['node_t'])):
                All_T = []
                for i in range(0, temp2['node_t'][nb_process].shape[0]):
                    All_T.append(temp2['node_t'][nb_process][i, :])
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
        if run_flag:
            ig = 0
            for j in range(1, Unique_T.shape[0]):
                if Unique_T[j] >= Generic_T[ig]:
                    index = np.where(Unique_T[j:] - Generic_T[ig] >= params.DT_obs)
                    index = index[0]
                    if np.array(index).size > 0:
                        Generic_T.append(Unique_T[j + index[0]])
                        ig += 1
            keys2 = ['node_w', 'node_z', 'node_s', 'node_dA', 'node_t']
            keys3 = ['reach_w', 'reach_z', 'reach_s', 'reach_dA', 'reach_t']
            keys4 = ['dist_out', 'node_length', 'reach_qwbm', 'facc', 'q_monthly_mean']
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
            for nb_process in range(0, len(temp2['node_t'])):
                keys_tmp = ['node_length', 'dist_out', 'q_monthly_mean']
                for key in keys_tmp:
                    if key == 'q_monthly_mean':
                        concat_dict[key].append(np.array(temp2[key][nb_process]))
                    else:
                        if np.array(temp2[key][nb_process]).size > 1:
                            concat_dict[key] = np.sort(np.hstack((concat_dict[key], np.flipud(temp2[key][nb_process]))))
                        else:
                            concat_dict[key] = np.sort(np.hstack((concat_dict[key], temp2[key][nb_process])))
                        if params.densification:
                            if key == 'dist_out':
                                if np.array(temp2[key][nb_process]).size > 1:
                                    concat_dict['dist_out_all'].append(temp2[key][nb_process])
                                else:
                                    concat_dict['dist_out_all'].append(temp2[key][nb_process])
                            if key == 'node_length':
                                if np.array(temp2[key][nb_process]).size > 1:
                                    concat_dict['node_length_all'].append(temp2[key][nb_process])
                                else:
                                    concat_dict['node_length_all'].append(temp2[key][nb_process])
                concat_dict['reach_id'].append(temp2['reach_id'][nb_process])
                keys_tmp = ['reach_qwbm', 'facc']
                for nb_process_tmp in range(0, len(temp2['node_t'])):
                    for key in keys_tmp:
                        if nb_process == 0:
                            pass
                        else:
                            concat_dict[key] = np.hstack((concat_dict[key], temp2[key][nb_process_tmp]))
                concat_dict['reach_qwbm'] = np.nanmean(concat_dict['reach_qwbm'])
                masked_data = np.ma.masked_values(np.array([concat_dict['reach_qwbm']]), value=-9999.0)
                concat_dict['reach_qwbm'] = masked_data
                concat_dict['facc'] = np.nanmean(concat_dict['facc'])
                masked_data2 = np.ma.masked_values(np.array([concat_dict['facc']]), value=-9999.0)
                concat_dict['facc'] = masked_data2
                if nb_process > 0:
                    DI += temp2['node_t'][nb_process - 1].shape[0]
                else:
                    DI = 0
                i_obs = -9999
                for i in range(temp2['node_t'][nb_process].shape[0]):
                    i_mat = i + DI
                    index = []
                    DT_obs = params.DT_obs
                    for j in range(temp2['node_t'][nb_process].shape[1]):
                        if temp2['node_t'][nb_process][i][j] > 0.0:
                            if i_obs < 0:
                                i_obs = i
                            index = np.array(np.where(abs(temp2['node_t'][nb_process][i][j] - Generic_T) <= DT_obs))
                            while index.shape[1] != 1:
                                index = np.array(np.where(abs(temp2['node_t'][nb_process][i][j] - Generic_T) <= DT_obs))
                                if index.shape[1] < 1:
                                    if DT_obs < 24.0 * 3600.0:
                                        DT_obs = DT_obs * 2.0
                                elif index.shape[1] > 1:
                                    DT_obs = DT_obs / 2.0
                            for key in keys2:
                                if calc.check_na(temp2[key][nb_process][i, j]):
                                    concat_dict[key][i_mat, index] = min(params.valid_min_dA, params.valid_min_z)
                                    concat_dict[key][i_mat].mask[index] = True
                                else:
                                    concat_dict[key][i_mat, index] = temp2[key][nb_process][i, j]
                            if i == i_obs:
                                for key in keys3:
                                    if calc.check_na(temp2[key][nb_process][j]):
                                        concat_dict[key][nb_process][index] = min(params.valid_min_dA, params.valid_min_z)
                                        concat_dict[key][nb_process].mask[index] = True
                                    else:
                                        concat_dict[key][nb_process][index] = temp2[key][nb_process][j]
            q_monthly_mean_stack = np.stack(concat_dict['q_monthly_mean'])
            concat_dict['q_monthly_mean'] = np.nanmean(q_monthly_mean_stack, axis=0)
            masked_data3 = np.ma.masked_values(np.array([concat_dict['q_monthly_mean']]), value=-9999.0)
            concat_dict['q_monthly_mean'] = masked_data3[0]
            flag_dict = {}
            flag_dict['node'] = {}
            flag_dict['reach'] = {}
            flag_dict['nx'] = concat_dict['node_z'].shape[0]
            flag_dict['nt'] = concat_dict['node_z'].shape[1]
            new_process = algorithms(concat_dict, flag_dict, param_dict)
            tpy1 = datetime.datetime.utcnow()
            new_process.launch_sic4dvar(len(temp2['node_t']))
        if param_dict['set_folder']:
            json_filename = param_dict['json_path'].stem
            output_dir = param_dict['output_dir'].joinpath(f'{json_filename}')
            folder_name = str(reaches_ids[0]) + '_' + str(reaches_ids[-1])
            output_dir = output_dir.joinpath(folder_name)
        else:
            output_dir = param_dict['output_dir']
        a5_counter = 0
        for nb_process in range(0, nbr_reach):
            if run_flag:
                if new_process.output['valid']:
                    if new_process.output['valid_a5_sets'][nb_process]:
                        logging.info(f'sic4dvar_set_run : writing results to output directory {output_dir}')
                        if nbr_reach > 1:
                            write_output(output_dir, param_dict, reaches_ids[nb_process], new_process.output, a5_counter)
                        if nbr_reach == 1:
                            write_output(output_dir, param_dict, reaches_ids[nb_process], new_process.output)
                        a5_counter += 1
                        logging.info('Results are valid for algo5 and 3.1')
                    else:
                        logging.info(f'sic4dvar_set_run : writing results to output directory {output_dir}')
                        if nbr_reach > 1:
                            write_output(output_dir, param_dict, reaches_ids[nb_process], new_process.output, -1, folder_name)
                        if nbr_reach == 1:
                            write_output(output_dir, param_dict, reaches_ids[nb_process], new_process.output, -1, folder_name)
                        logging.info('Results are valid for algo31 only')
                else:
                    logging.info('Results are INVALID!')
                    if param_dict['aws']:
                        if nbr_reach > 1:
                            write_output(param_dict['output_dir'], param_dict, reaches_ids[nb_process], new_process.output, -1, folder_name)
                        if nbr_reach == 1:
                            write_output(param_dict['output_dir'], param_dict, reaches_ids[nb_process], new_process.output, -1, folder_name)
                        logging.info('Force saving for AWS.')
            else:
                logging.info('No data to save.')
                if param_dict['aws']:
                    if nbr_reach > 1:
                        write_output(param_dict['output_dir'], param_dict, reaches_ids[nb_process], [], -1, folder_name, max(dim_t))
                    if nbr_reach == 1:
                        write_output(param_dict['output_dir'], param_dict, reaches_ids[nb_process], [], -1, folder_name, max(dim_t))
                    logging.info('Force saving for AWS.')
        t_set1 = datetime.datetime.utcnow()
        logging.info('Run time for current set is = %s.', str(t_set1 - t_reach0))
        close_logger()
    t1 = datetime.datetime.utcnow()
    logging.info('Run time in total= %s.', str(t1 - t0))

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