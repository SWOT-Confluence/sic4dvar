from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import scipy
import sic4dvar_params as params
from sic4dvar_algos.algo5 import *
from sic4dvar_algos.algo3 import *
from sic4dvar_functions.sic4dvar_helper_functions import get_weighted_q_data, correlation_nodes
from sic4dvar_functions.sic4dvar_calculations import bathymetry_computation, compute_bb, call_func_APR, verify_name_length
from sic4dvar_functions.T338 import W, X
from sic4dvar_functions.l760 import L
from sic4dvar_functions.j.z753 import M
from lib.lib_swot_obs import get_flag_dict_from_config, filter_swot_obs_on_quality
from lib.lib_dates import get_swot_dates, seconds_to_date_old
try:
    import sys
    sys.path.append('/mnt/DATA/worksync/A trier/files/scripts/')
except:
    print('path /mnt/DATA/worksync/A trier/files/scripts/ not found')

class algorithms:

    def __init__(self, _input_data, flag_dict, param_dict):
        self.input_data = _input_data
        self.flag_dict = flag_dict
        self.param_dict = param_dict
        self.filtered_data = {}
        self.output = {}
        self.output['valid'] = 1
        self.output['valid_a5'] = 1
        self.output['valid_a5_sets'] = []
        self.intermediates_values = {}
        self.algo5_results = {}
        self.bb = 9999.0
        self.reliability = 'valid'
        if params.force_specific_dates:
            print(self.input_data['reach_t'])
            dates = []
            for t in range(0, len(self.input_data['reach_t'])):
                if calc.check_na(self.input_data['reach_t'][t]):
                    dates.append(datetime(1, 1, 1))
                else:
                    dates.append(seconds_to_date_old(self.input_data['reach_t'][t]))
            indexes = [i for i, date in enumerate(dates) if params.start_date <= date <= params.end_date]
            if np.array(indexes).size == 0:
                if self.param_dict['run_type'] == 'seq':
                    logging.info('All data removed by date filtering.')
                    logging.info("Sequential run, can't process without SWOT data.")
                    self.output['valid'] = 0
                    return None
            else:
                reach_keys = ['reach_dA', 'reach_s', 'reach_w', 'reach_z', 'reach_t']
                node_keys = ['node_w', 'node_z', 'node_dA', 'node_s', 'node_t']
                for key in reach_keys:
                    self.input_data[key] = self.input_data[key][indexes]
                for key in node_keys:
                    self.input_data[key] = self.input_data[key][:, indexes]
                for key in list(self.flag_dict['node'].keys()):
                    if not self.flag_dict['node'][key].shape == (0,):
                        self.flag_dict['node'][key] = self.flag_dict['node'][key][:, indexes]
                for key in list(self.flag_dict['reach'].keys()):
                    if key != 'reach_width_min' and key != 'reach_slope_min' and (key != 'reach_length_min'):
                        self.flag_dict['reach'][key] = self.flag_dict['reach'][key][indexes]
                self.flag_dict['nt'] = len(indexes)
                print(self.input_data['node_z'].shape, self.input_data['reach_z'].shape)
        if self.input_data == 0:
            raise IOError('Error reading inputs ... self.input_data is empty')
        if 'node_t' in self.input_data and 'reach_w' in self.input_data:
            if self.param_dict['run_type'] == 'set':
                mask = np.ones(self.input_data['node_t'].shape[1], dtype=bool)
                arr = np.full(self.input_data['node_t'].shape[1], np.nan)
            if self.param_dict['run_type'] == 'seq':
                mask = np.ones(len(self.input_data['reach_w']), dtype=bool)
                arr = np.full(len(self.input_data['reach_w']), np.nan)
            self.output['q_algo5'] = np.ma.array(arr, mask=mask)
            self.output['q_algo5_all'] = []
            self.output['q_algo31'] = np.ma.array(arr, mask=mask)
            self.output['a0'] = np.ma.array([np.nan], mask=[True])
            self.output['n'] = np.ma.array([np.nan], mask=[True])
            self.output['node_a0'] = []
            self.output['node_n'] = []
            self.lim_node_xr = []
            self.output['half_width'] = np.empty(len(self.input_data['node_w']), dtype=object)
            self.output['elevation'] = np.empty(len(self.input_data['node_w']), dtype=object)
            self.output['half_width'].fill(np.ma.array(arr, mask=mask))
            self.output['elevation'].fill(np.ma.array(arr, mask=mask))
            self.output['time'] = []
            self.removed_nodes_ind = np.zeros(0)
            self.indices = []
        elif self.param_dict['run_type'] == 'seq':
            logging.info("ERROR: Sequential run, can't process without SWOT data.")
            return 0
        elif self.param_dict['run_type'] == 'set':
            mask = []
            arr = []
            logging.info('INFO: Missing SWOT data. Set_run, processing.')
            self.output['q_algo5'] = []
            self.output['q_algo5_all'] = []
            self.output['q_algo31'] = []
            self.output['a0'] = []
            self.output['n'] = []
            self.output['node_a0'] = []
            self.output['node_n'] = []
            self.lim_node_xr = []
            self.output['half_width'] = []
            self.output['elevation'] = []
            self.output['time'] = []
            self.removed_nodes_ind = np.zeros(0)
            self.indices = []
        self.intermediates_values['node_t'] = self.input_data['node_t']
        self.intermediates_values['reach_t'] = self.input_data['reach_t']
        if not params.node_length:
            self.input_data['node_x'] = self.input_data['dist_out']
        if params.node_length:
            acc_node_len = np.cumsum(self.input_data['node_length'])
            dist_out_0 = scipy.stats.mode(np.abs(acc_node_len - self.input_data['dist_out']), axis=None).mode
            self.input_data['node_x'] = dist_out_0 + acc_node_len
        test = []
        for i in range(1, np.array(self.input_data['dist_out']).size):
            test.append(self.input_data['dist_out'][i] - self.input_data['dist_out'][i - 1])

    def launch_sic4dvar(self, reach_number=0):
        logging.info('Runs launch_sic4dvar over %d reaches' % reach_number)
        self.data_is_useable = True
        if self.param_dict['run_type'] == 'set':
            for i in range(0, reach_number):
                self.output['valid_a5_sets'].append(1)
        if params.pankaj_test:
            params.extrapolation = False
        logging.info('Discard reaches and nodes depending on SWOT quality values')
        val_swot_q_flag = get_flag_dict_from_config(self.param_dict)
        node_level_mask = filter_swot_obs_on_quality(self.flag_dict, val_swot_q_flag)
        valid_z = np.sum(~np.isnan(self.input_data['node_z']))
        valid_w = np.sum(~np.isnan(self.input_data['node_w']))
        logging.info(f'Loaded SWOT data : {valid_z} valid wse and {valid_w} valid')
        self.output['obs_filtering_ratio'] = np.sum(node_level_mask) * 100 // node_level_mask.size
        logging.info(f'Obs filtering ratio is {self.output['obs_filtering_ratio']}%')
        self.input_data['node_z'] = np.ma.masked_array(self.input_data['node_z'], mask=node_level_mask)
        self.input_data['node_w'] = np.ma.masked_array(self.input_data['node_w'], mask=node_level_mask)
        valid_z = np.sum(~np.isnan(self.input_data['node_z']))
        valid_w = np.sum(~np.isnan(self.input_data['node_w']))
        logging.info(f'After obs node and reach level filtering : {valid_z} valid wse and {valid_w} valid')
        if len(self.input_data['node_x']) == 0:
            self.data_is_useable = False
        if len(self.input_data['node_x']) != len(self.input_data['node_z']):
            self.data_is_useable = False
        if params.extrapolation and self.data_is_useable:
            logging.info('Extrapolating wse data')
            if params.old_extrapolation == True:
                self.old_Extrapolation()
            else:
                self.Extrapolation()
                valid_z = np.sum(~np.isnan(self.input_data['node_z']))
                valid_w = np.sum(~np.isnan(self.input_data['node_w']))
                logging.info(f'After extrapolation: {valid_z} valid wse and {valid_w} valid width')
        logging.info('Filtering data.')
        if self.data_is_useable:
            if not params.pankaj_test:
                pass
            if params.pankaj_test:
                for n in range(0, len(self.input_data['node_w'])):
                    self.input_data['node_w'][n, :] = np.ones(len(self.input_data['node_w'][0])) * 10
        if self.data_is_useable and params.run_algo31_v3:
            cs_float_atol = 0.01
            def_float_atol = 1e-08
            self.data_is_useable, self.observed_nodes, self.list_to_keep, self.removed_indices = tmp_filter_function(self.input_data['node_z'], self.input_data['node_w'])
            if np.array(self.observed_nodes).size == 0 or np.array(self.list_to_keep).size == 0:
                self.data_is_useable = False
            if self.data_is_useable:
                self.filtered_data['node_w'] = self.input_data['node_w'][self.observed_nodes]
                self.filtered_data['node_z'] = self.input_data['node_z'][self.observed_nodes]
                cs_max_n_points = self.bathymetry_computation()

                def array_of_arrays_to_regular(array_of_arrays):
                    max_length = max((len(subarray) for subarray in array_of_arrays))
                    result_array = np.full((len(array_of_arrays), max_length), np.nan)
                    for i, subarray in enumerate(array_of_arrays):
                        result_array[i, :len(subarray)] = subarray
                    return result_array
                self.input_data['node_xr'] = array_of_arrays_to_regular(self.input_data['node_xr'])
                self.input_data['node_yr'] = array_of_arrays_to_regular(self.input_data['node_yr'])
                cs_w, cs_z = cs_regularize(np.array(self.input_data['node_xr']), np.array(self.input_data['node_yr']), params.algo_bounds[1][2], cs_float_atol, def_float_atol, False, '', False, True)
        if self.data_is_useable and (not params.run_algo31_v3):
            self.data_is_useable, self.observed_nodes, self.list_to_keep, self.removed_indices = tmp_filter_function(self.input_data['node_z'], self.input_data['node_w'])
        if self.data_is_useable and np.array(self.list_to_keep).size > 0:
            if self.param_dict['use_reach_slope']:
                data_is_useable_sl, observed_nodes_sl, list_to_keep_sl, removed_indices_sl = tmp_filter_function(np.array([self.input_data['reach_s']]), np.array([self.input_data['reach_s']]))
                if data_is_useable_sl:
                    list_to_keep_intersection = np.intersect1d(self.list_to_keep, list_to_keep_sl)
                    total_array = np.arange(len(self.input_data['node_z'][0]))
                    removed_indices_intersection = np.setdiff1d(total_array, list_to_keep_intersection)
                    if np.array(list_to_keep_intersection).size > 0:
                        self.list_to_keep = list_to_keep_intersection
                        self.removed_indices = removed_indices_intersection
                    else:
                        self.data_is_useable = False
            self.filtered_data['node_w'] = self.input_data['node_w'][:, self.list_to_keep][self.observed_nodes]
            self.filtered_data['node_z'] = self.input_data['node_z'][:, self.list_to_keep][self.observed_nodes]
            self.filtered_data['node_t'] = self.input_data['node_t'][:, self.list_to_keep][self.observed_nodes]
            keys = ['reach_w', 'reach_z', 'reach_s', 'reach_dA']
            for key in keys:
                if self.param_dict['run_type'] == 'set':
                    self.filtered_data[key] = []
                    for i in range(0, len(self.input_data[key])):
                        self.filtered_data[key].append(self.input_data[key][i][self.list_to_keep])
                if self.param_dict['run_type'] == 'seq':
                    self.filtered_data[key] = []
                    self.filtered_data[key] = self.input_data[key][self.list_to_keep]
            self.filtered_data['reach_t'] = self.input_data['reach_t'][self.list_to_keep]
            self.filtered_data['n_obs'] = np.ones(self.filtered_data['node_w'].shape[0], dtype=int) * self.filtered_data['node_w'].shape[1]
            if not params.node_length:
                self.filtered_data['node_x'] = self.input_data['node_x'][self.observed_nodes]
            elif params.node_length:
                self.filtered_data['node_x'] = self.input_data['node_x'][self.observed_nodes]

            def find_array_with_most_valid_values(arrays):
                max_valid_count = -1
                max_valid_index = -1
                for i, arr in enumerate(arrays):
                    valid_count = np.sum(~np.isnan(arr))
                    if valid_count > max_valid_count:
                        max_valid_count = valid_count
                        max_valid_index = i
                return max_valid_index
            index = find_array_with_most_valid_values(self.filtered_data['node_t'][:])
            self.filtered_data['orig_time'] = self.filtered_data['node_t'][0]
            if not params.pankaj_test:
                self.output['time'] = self.input_data['reach_t'] / 86400
            if params.pankaj_test:
                self.output['time'] = self.filtered_data['orig_time'] / 86400
        if params.pankaj_test:
            self.list_to_keep = np.arange(0, len(self.input_data['node_z'][0]), 1, dtype=int)
            self.removed_indices = []
            valid_indices1 = ~np.isnan(self.input_data['node_t'][0])
            valid_indices2 = ~np.isnan(self.input_data['node_t'][1])
            valid_values1 = self.input_data['node_t'][0][valid_indices1]
            valid_values2 = self.input_data['node_t'][1][valid_indices2]
            combined_valid_values = np.concatenate((valid_values1, valid_values2))
            sorted_combined_values = np.unique(np.sort(combined_valid_values))
            self.filtered_data['orig_time'] = sorted_combined_values
            self.output['time'] = sorted_combined_values / 86400
            reference_date = datetime.datetime(2000, 1, 1, 0, 0, 0)
            time_in_seconds = self.filtered_data['orig_time']
            time_dates = [reference_date + datetime.timedelta(seconds=ts) for ts in time_in_seconds]
            time_strings = [time_date.strftime('%Y-%m-%d') for time_date in time_dates]
            self.filtered_data['node_w'] = self.input_data['node_w']
            self.filtered_data['node_z'] = self.input_data['node_z']
        valid_z = np.sum(~np.isnan(self.input_data['node_z']))
        valid_w = np.sum(~np.isnan(self.input_data['node_w']))
        logging.info(f'')
        flag_qwbm = True
        logging.info(f'')
        if (calc.check_na(self.input_data['reach_qwbm']) or self.input_data['reach_qwbm'] < 1.0) and (not self.param_dict['q_prior_from_stations']):
            logging.info('')
            logging.info('')
            flag_qwbm = False
            if not calc.check_na(self.input_data['facc']) or self.input_data['facc'] > 1.0:
                self.input_data['reach_qwbm'] = self.input_data['facc'] * 10 / 1000
                logging.info('')
                logging.info(f'')
                flag_qwbm = True
        if self.param_dict['override_q_prior']:
            self.input_data['reach_qwbm'] = np.ma.masked_values(float(self.param_dict['q_prior_value']), value=-9999.0)
            flag_qwbm = True
        if self.data_is_useable and self.param_dict['q_monthly_mean']:
            times = self.input_data['reach_t']
            count = 0
            for i in range(0, len(self.input_data['q_monthly_mean'])):
                if calc.check_na(self.input_data['q_monthly_mean'][i]):
                    count = count + 1
            if count == len(self.input_data['q_monthly_mean']):
                logging.info('')
                logging.info('all q monthly mean data was empty !')
            else:
                dates = get_swot_dates(self.filtered_data['node_t'])
                masked_data = np.ma.masked_values(np.array([get_weighted_q_data(dates, self.input_data['q_monthly_mean'])]), value=-9999.0)
                self.input_data['reach_qwbm'] = masked_data
                logging.info('INFO: replaced qwbm value with monthly mean computation')
                logging.info(f'')
        if self.data_is_useable and flag_qwbm:
            self.node_check = True
            if self.node_check:
                if self.input_data['reach_qwbm'].mask:
                    pass
                logging.info('INFO: Running algo31 to estimate discharge.')
                if params.cython_version:
                    self.output['q_algo31'] = self.algo31v2()
                elif not params.V32:
                    if not params.run_algo31_v3:
                        if params.densification:
                            pass
                        if not params.densification:
                            SLOPEM1 = self.slope_computation()
                            if not self.output['valid']:
                                return 0
                            self.input_data['node_xr'], self.input_data['node_yr'] = bathymetry_computation(node_w=self.filtered_data['node_w'], node_z=self.filtered_data['node_z'], param_dict=self.param_dict, params=params, input_data=self.input_data, filtered_data=self.filtered_data, slope=SLOPEM1)
                            node_x = self.input_data['node_x']
                            reach_id = str(self.input_data['reach_id'])
                            reach_id = verify_name_length(reach_id)
                            times2 = np.around(self.input_data['reach_t'] / 3600 / 24)
                            times2 = times2 - np.nanmin(times2)
                            nodes2 = (node_x - node_x[0]) / 1000
                            tmp_min = []
                            for i in range(0, len(self.input_data['node_yr'])):
                                tmp_min.append(np.nanmin(self.input_data['node_yr'][i]))
                            if self.param_dict['gnuplot_saving']:
                                output_path = self.param_dict['output_dir'].joinpath('gnuplot_data', reach_id)
                                if not Path(output_path).is_dir():
                                    Path(output_path).mkdir(parents=True, exist_ok=True)
                                gnuplot_save(nodes2, times2, self.input_data['node_z_ini'], self.input_data['node_w_ini'], output_path.joinpath('out_wse_w' + '_ini'), tmp_min, 2)
                                gnuplot_save(nodes2, times2, self.tmp_interp_values[0], self.input_data['node_w_ini'], output_path.joinpath('out_wse' + '_deviation'), tmp_min, 2)
                                gnuplot_save(nodes2, times2, self.tmp_interp_values[1], self.tmp_interp_values_w[0], output_path.joinpath('out_wse_w' + '_relax_space1'), tmp_min, 2)
                                gnuplot_save(nodes2, times2, self.tmp_interp_values[2], self.tmp_interp_values_w[1], output_path.joinpath('out_wse_w' + '_interpolated'), tmp_min, 2)
                                gnuplot_save(nodes2, times2, self.tmp_interp_values[3], self.tmp_interp_values_w[2], output_path.joinpath('out_wse_w' + '_relax_space2'), tmp_min, 2)
                                gnuplot_save(nodes2, times2, self.tmp_interp_values[4], self.tmp_interp_values_w[3], output_path.joinpath('out_wse_w' + '_final'), tmp_min, 1)
                                if len(self.tmp_interp_values) > 5:
                                    gnuplot_save(nodes2, times2, self.tmp_interp_values[5], self.input_data['node_w_ini'], output_path.joinpath('out_wse' + '_deviation_new'), tmp_min, 2)
                                gnuplot_save_list(nodes2, times2, self.input_data['node_yr'], self.input_data['node_xr'], output_path.joinpath('out_z_w' + '_bathy'), tmp_min, 2)
                            self.input_data['node_a'], self.input_data['node_p'], self.input_data['node_r'], self.input_data['node_w_simp'], self.bb = call_func_APR(self.filtered_data['node_w'], self.filtered_data['node_z'], self.input_data['node_xr'], self.input_data['node_yr'], params, self.param_dict)
                            tmp40 = []
                            tmp41 = []
                            tmp42 = []
                            tmp43 = []
                            tmp44 = []
                            for n in range(0, self.filtered_data['node_z'].shape[0]):
                                tmp41.append(self.filtered_data['node_z'][n, self.filtered_data['node_z'][n, :].argmax()])
                                tmp43.append(self.filtered_data['node_w'][n, self.filtered_data['node_w'][n, :].argmax()])
                                tmp40.append(self.input_data['node_xr'][n][self.input_data['node_xr'][n].argmin()])
                                tmp44.append(self.input_data['node_xr'][n].argmin())
                            t0_a31 = datetime.utcnow()
                            apr_array = {'node_w_simp': self.input_data['node_w_simp'], 'node_a': self.input_data['node_a'], 'node_p': self.input_data['node_p'], 'node_r': self.input_data['node_r']}
                            bathymetry_array = {'node_xr': self.input_data['node_xr'], 'node_yr': self.input_data['node_yr']}
                            self.output['q_algo31'], self.output['valid'], self.reliability = algo3(apr_array, self.input_data['reach_qwbm'], params, SLOPEM1, self.output['valid'], self.filtered_data['node_z'], self.input_data['node_z'], self.input_data['node_z_ini'], self.filtered_data['node_x'], self.last_node_for_integral, bathymetry_array, self.param_dict, self.input_data['reach_id'], self.input_data['reach_t'], bb=self.bb, reliability=self.reliability, Qsdev=self.input_data['reach_qsdev'])
                            self.intermediates_values['q_da'] = self.output['q_algo31']
                            t1_a31 = datetime.utcnow()
                    else:
                        cs_max_n_points = self.bathymetry_computation()
                        SLOPEM1 = self.slope_computation()
                        def_slope_max: float = np.inf
                        def_slope_min: float = 1e-12
                        def_float_atol = 1e-08
                        xr_array = np.full([self.filtered_data['node_w'].shape[0], cs_max_n_points + 1], fill_value=np.nan)
                        yr_array = np.full([self.filtered_data['node_z'].shape[0], cs_max_n_points + 1], fill_value=np.nan)
                        for node_n in range(self.filtered_data['node_w'].shape[0]):
                            xr_array[node_n, 0:len(self.input_data['node_xr'][node_n])] = np.array(self.input_data['node_xr'][node_n])
                            yr_array[node_n, 0:len(self.input_data['node_yr'][node_n])] = np.array(self.input_data['node_yr'][node_n])
                        logging.info(f'node_x={self.filtered_data['node_x']}')
                        self.output['q_algo31'] = algo31_v3(self.filtered_data['node_z'], self.filtered_data['node_x'], SLOPEM1, xr_array, yr_array, self.input_data['reach_qwbm'], params.algo_bounds[0][0], params.algo_bounds[1][0], params.algo_bounds[1][1], params.algo_bounds[1][2], params.algo_bounds[2][0], params.algo_bounds[2][1], params.algo_bounds[2][2], params.val1, params.val2, params.shape03, params.shape13, params.shape23, params.kDim, params.local_QM1, def_slope_min, def_slope_max, params.slope_smoothing_num_passes, def_float_atol, False, '', False, True)[0]
                elif params.V32:
                    logging.info('Entering algo32 for discharge prediction.')
                    self.output['q_algo31'] = self.algo32()
                Q31 = self.output['q_algo31']
                if np.array(Q31).size == 0:
                    self.output['valid'] = 0
                    return 0
                self.output['q_algo31'] = self.build_output_q_masked_array('q_algo31')
                self.output['time'] = self.build_output_q_masked_array('time')
                self.output['width'] = self.input_data['node_xr']
                self.output['elevation'] = self.input_data['node_yr']
                if self.param_dict['run_type'] == 'set':
                    NBR_REACHES = len(self.filtered_data['reach_w'])
                if self.param_dict['run_type'] == 'seq':
                    NBR_REACHES = 1
                if self.param_dict['run_algo5']:
                    if self.input_data['reach_dA'].mask.all():
                        cross_section = bathymetry_computation([self.input_data['reach_w']], [self.input_data['reach_z']], self.param_dict, params)
                        self.filtered_data['reach_dA'], _, _, _, _ = call_func_APR([self.filtered_data['reach_w']], [self.filtered_data['reach_z']], [cross_section[0][0]], [cross_section[1][0]], params, self.param_dict)
                        self.filtered_data['reach_dA'] = self.filtered_data['reach_dA'][0]
                        self.filtered_data['reach_dA'] = np.ma.array(self.filtered_data['reach_dA'], mask=False)
                    for i in range(0, NBR_REACHES):
                        if self.param_dict['run_type'] == 'seq':
                            self.reach_check = self.check_reach_data_SET()
                        if self.param_dict['run_type'] == 'set':
                            self.reach_check = self.check_reach_data_SET(i)
                        if self.reach_check:
                            logging.info('Running algo5 at REACH level to estimate discharge, A0 & n.')
                            if self.param_dict['run_type'] == 'set':
                                self.output['q_algo5'], a0, n = algo5(self.output['q_algo31_masked'], self.filtered_data['reach_dA'][i], self.filtered_data['reach_w'][i], self.filtered_data['reach_s'][i], self.input_data['reach_id'][i])
                                self.output['node_a0'] += [a0]
                                self.output['node_n'] += [n]
                                self.output['q_algo5'] = algo5_fill_removed_data(self.output['q_algo5'], self.removed_indices, len(self.filtered_data['node_z'][0]))
                                self.output['q_algo5'] = self.build_output_q_masked_array('q_algo5')
                                self.output['q_algo5_all'] += [self.output['q_algo5']]
                            if params.node_run:
                                logging.info('Running algo5 at NODE level to estimate discharge, A0 & n.')
                                for ni in range(len(self.filtered_data['node_w'])):
                                    q_algo5, a0, n = algo5(self.output['q_algo31'][self.list_to_keep], self.filtered_data['node_dA'][ni], self.filtered_data['node_w'][ni], self.filtered_data['node_s'][ni])
                                    self.output['node_a0'] += [a0]
                                    self.output['node_n'] += [n]
                                logging.info('Preparing node half-width & elevation data to be output.')
                                if params.cython_version:
                                    self.prepare_output_half_width_elevation_v2()
                                else:
                                    self.prepare_output_half_width_elevation()
                            if self.param_dict['run_type'] == 'seq':
                                t0_a5 = datetime.utcnow()
                                self.output['q_algo5'], self.algo5_results = algo5(self.output['q_algo31_masked'], self.filtered_data['reach_dA'], self.filtered_data['reach_w'], self.filtered_data['reach_s'], self.input_data['reach_id'], equation='ManningLW')
                                t1_a5 = datetime.utcnow()
                                self.output['q_algo5'] = algo5_fill_removed_data(self.output['q_algo5'], self.removed_indices, len(self.filtered_data['node_z'][0]))
                                self.output['q_algo5'] = self.build_output_q_masked_array('q_algo5')
                            self.intermediates_values['q_mm'] = self.output['q_algo5']
                        else:
                            self.output['q_algo31'] = self.build_output_q_masked_array('q_algo31')
                            logging.info('WARNING: not enough REACH data found to process reach.')
                            if self.param_dict['run_type'] == 'seq':
                                self.output['valid_a5'] = 0
                            if self.param_dict['run_type'] == 'set':
                                self.output['valid_a5_sets'][i] = 0
            else:
                logging.info('')
                self.output['valid'] = 0
        else:
            logging.info('WARNING: not enough consecutive data available to process reach.')
            self.output['valid'] = 0
        if self.param_dict['write_intermediate_products'] == True:
            output_path = self.param_dict['output_dir'].joinpath('%d_sic4dvar_intermediates_values' % self.input_data['reach_id'])
            np.savez(output_path, **self.intermediates_values)

    def Extrapolation(self):
        self.input_data['node_t_ini'] = deepcopy(self.input_data['node_t'])
        self.input_data['node_z_ini'] = deepcopy(self.input_data['node_z'])
        self.input_data['node_w_ini'] = deepcopy(self.input_data['node_w'])
        node_x = self.input_data['node_x']
        corx = np.mean(np.diff(node_x))
        test_swot_z_obs = self.input_data['node_z']
        test_swot_w_obs = self.input_data['node_w']
        test_x_array = node_x
        if self.param_dict['run_type'] == 'seq':
            test_t_array = self.input_data['reach_t']
            if params.force_create_reach_t:
                test_t_array = np.ones(len(self.input_data['node_t'][0]))
                for t in range(0, len(self.input_data['node_t'][0])):
                    test_t_array[t] = np.mean(self.input_data['node_t'][:, t])
                self.input_data['reach_t'] = test_t_array
        if self.param_dict['run_type'] == 'set':
            test_t_array = np.ones(len(self.input_data['node_t'][0]))
            for t in range(0, len(self.input_data['node_t'][0])):
                test_t_array[t] = np.mean(self.input_data['node_t'][:, t])
            self.input_data['separate_reach_t'] = deepcopy(self.input_data['reach_t'])
            self.input_data['reach_t'] = test_t_array
        cort = params.cort
        sections_plot = False
        self.tmp_interp_values = []
        self.tmp_interp_values_w = []
        arr = self.input_data['node_z_ini']
        arr[arr < -100000] = np.nan
        self.intermediates_values['z0'] = test_swot_z_obs.filled(np.nan)
        self.intermediates_values['w0'] = test_swot_w_obs.filled(np.nan)
        if params.large_deviations:
            test_swot_z_obs = large_deviations_removal(self.input_data['node_x'], self.input_data['node_z_ini'])
            self.tmp_interp_values.append(test_swot_z_obs)
            test_swot_z_obs2, c1, c2 = global_large_deviations_removal(self.input_data['node_x'], self.input_data['node_z_ini'])
            nodes2 = (node_x - node_x[0]) / 1000
            times2 = test_t_array / 3600 / 24
            if self.param_dict['gnuplot_saving']:
                reach_id = str(self.input_data['reach_id'])
                reach_id = verify_name_length(reach_id)
                output_path = self.param_dict['output_dir'].joinpath('gnuplot_data', str(reach_id))
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                output_path = self.param_dict['output_dir'].joinpath('gnuplot_data', str(reach_id), 'c1c2')
                gnuplot_save_c1c2(nodes2, c1, c2, times2, output_path)
            test_swot_z_obs = test_swot_z_obs2
        self.intermediates_values['z1'] = test_swot_z_obs.filled(np.nan)
        self.intermediates_values['w1'] = self.input_data['node_w'].filled(np.nan)
        test_swot_z_obs = L(dim=1, value0_array=test_swot_z_obs, base0_array=test_x_array, max_iter=params.LSMX, cor=corx, always_run_first_iter=True, behaviour='', inter_behaviour=False, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='', min_change_v_thr=0.0001, plot=False, plot_title='', clean_run=True, debug_mode=False)
        self.intermediates_values['z2'] = test_swot_z_obs.filled(np.nan)
        times = np.arange(0, len(self.input_data['node_z'][0]))
        nodes = np.arange(0, len(self.input_data['node_z']))
        times2 = np.around(test_t_array / 3600 / 24)
        times2 = times2 - min(times2)
        nodes2 = (node_x - node_x[0]) / 1000
        reach_id = str(self.input_data['reach_id'])
        reach_id = verify_name_length(reach_id)
        if self.param_dict['gnuplot_saving']:
            output_path = self.param_dict['output_dir'].joinpath('gnuplot_data', reach_id)
            if not Path(output_path).is_dir():
                Path(output_path).mkdir(parents=True, exist_ok=True)
        if False:
            t = len(test_swot_z_obs[0])
            n = len(test_swot_z_obs[:, 0])
            base_value = 61.15
            import random
            min_value = base_value - 2
            max_value = base_value + 2
            array_test = []
            for i in range(0, n):
                if i % 2 == 0:
                    test_swot_z_obs[i, 0] = random.uniform(min_value, max_value)
                    test_swot_z_obs[i, 1] = random.uniform(min_value, max_value)
                else:
                    test_swot_z_obs[i, 0] = np.nan
                    test_swot_z_obs[i, 1] = np.nan
                test_swot_z_obs[i, 2] = np.nan
            test_swot_z_obs[int(n / 2), 2] = random.uniform(min_value, max_value)
            array_test = test_swot_z_obs[:, 0:3]
            array_test = W(values0_array=array_test, space0_array=test_x_array, dx_max_in=params.DX_max_in, dx_max_out=params.DX_max_out, dw_min=0.01, clean_run=False, debug_mode=False)
        self.tmp_interp_values.append(test_swot_z_obs)
        test_swot_z_obs = W(values0_array=test_swot_z_obs, space0_array=test_x_array, dx_max_in=params.DX_max_in, dx_max_out=params.DX_max_out, dw_min=0.01, clean_run=False, debug_mode=False)
        self.intermediates_values['z3'] = test_swot_z_obs.filled(np.nan)
        self.tmp_interp_values.append(test_swot_z_obs)
        test_swot_z_obs = L(dim=1, value0_array=test_swot_z_obs, base0_array=test_x_array, max_iter=params.LSMX, cor=corx, always_run_first_iter=True, behaviour='decrease', inter_behaviour=True, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='force', min_change_v_thr=0.0001, plot=False, plot_title='', debug_mode=False)
        self.tmp_interp_values.append(test_swot_z_obs)
        test_swot_z_obs = L(dim=0, value0_array=test_swot_z_obs, base0_array=test_t_array, max_iter=params.LSMT, cor=cort, always_run_first_iter=False, behaviour='', inter_behaviour=False, inter_behaviour_min_thr=params.def_float_atol, inter_behaviour_max_thr=params.DX_max_in, check_behaviour='', min_change_v_thr=0.0001, plot=False, plot_title='', debug_mode=False)
        self.intermediates_values['z4'] = test_swot_z_obs.filled(np.nan)
        self.tmp_interp_values.append(test_swot_z_obs)
        self.tmp_interp_values.append(test_swot_z_obs2)
        test_swot_w_obs = L(dim=1, value0_array=test_swot_w_obs, base0_array=test_x_array, max_iter=params.LSMX, cor=corx, always_run_first_iter=True, behaviour='', inter_behaviour=False, check_behaviour='', min_change_v_thr=0.01, plot=False, plot_title='', debug_mode=False)
        self.intermediates_values['w2'] = test_swot_w_obs.filled(np.nan)
        self.tmp_interp_values_w.append(test_swot_w_obs)
        test_swot_w_obs = X(values0_array=test_swot_w_obs, space0_array=test_x_array, weight0_array=test_swot_z_obs, dx_max_in=params.DX_max_in, dx_max_out=params.DX_max_out, dw_min=0.1, clean_run=False, debug_mode=False)
        self.intermediates_values['w3'] = test_swot_w_obs.filled(np.nan)
        self.tmp_interp_values_w.append(test_swot_w_obs)
        test_swot_w_obs = L(dim=1, value0_array=test_swot_w_obs, base0_array=test_x_array, max_iter=params.LSMX, cor=corx, always_run_first_iter=True, behaviour='', inter_behaviour=False, check_behaviour='', min_change_v_thr=0.01, plot=False, plot_title='', debug_mode=False)
        self.tmp_interp_values_w.append(test_swot_w_obs)
        test_swot_w_obs = L(dim=0, value0_array=test_swot_w_obs, base0_array=test_t_array, max_iter=params.LSMT, cor=cort, always_run_first_iter=True, behaviour='', inter_behaviour=False, check_behaviour='', min_change_v_thr=0.01, plot=False, plot_title='', debug_mode=False)
        self.intermediates_values['w4'] = test_swot_w_obs.filled(np.nan)
        self.tmp_interp_values_w.append(test_swot_w_obs)
        self.input_data['node_z'] = test_swot_z_obs.filled(np.nan)
        self.input_data['node_w'] = test_swot_w_obs.filled(np.nan)


    def check_node_data(self):
        unuseable_indices = {}
        node_indices = {}
        for key in self.filtered_data:
            if key == 'node_w' or key == 'node_z' or key == 'node_dA' or (key == 'node_s'):
                unuseable_indices[key] = []
                for i in range(len(self.filtered_data[key])):
                    if self.filtered_data[key][i].mask.any():
                        unuseable_indices[key] += [i]
        if unuseable_indices['node_w'] == unuseable_indices['node_z'] and unuseable_indices['node_dA'] == unuseable_indices['node_w'] and (unuseable_indices['node_s'] == unuseable_indices['node_w']):
            diff_between_indices = []
            for i in range(len(unuseable_indices['node_w']) - 1):
                diff_between_indices += [unuseable_indices['node_w'][i + 1] - unuseable_indices['node_w'][i]]
            biggest_diff = [diff_between_indices[0], 0]
            for i in range(len(diff_between_indices)):
                if biggest_diff[0] < diff_between_indices[i]:
                    biggest_diff = [diff_between_indices[i], i]
            start_node_index = unuseable_indices['node_w'][biggest_diff[1]] + 1
            end_node_index = unuseable_indices['node_w'][biggest_diff[1] + 1] - 1
            node_indices = [start_node_index, end_node_index]
        tmp = {}
        for key in unuseable_indices:
            tmp[key] = []
            for i in range(self.filtered_data[key].shape[0]):
                if i >= node_indices[0] and i <= node_indices[1]:
                    tmp[key].append(self.filtered_data[key][i])
        for key in tmp:
            self.filtered_data[key] = tmp[key]
        if len(self.filtered_data['node_w']) < 5:
            return False
        else:
            return True

    def check_reach_data_SET(self, iR=0):
        self.output['q_algo31_masked'] = deepcopy(self.output['q_algo31'][self.list_to_keep])
        reach = dict()
        if self.param_dict['run_type'] == 'seq':
            reach['dA'] = deepcopy([self.filtered_data['reach_dA']])
            reach['w'] = deepcopy([self.filtered_data['reach_w']])
            reach['s'] = deepcopy([self.filtered_data['reach_s']])
            reach['z'] = deepcopy([self.filtered_data['reach_z']])
        if self.param_dict['run_type'] == 'set':
            reach['dA'] = deepcopy(self.filtered_data['reach_dA'])
            reach['w'] = deepcopy(self.filtered_data['reach_w'])
            reach['s'] = deepcopy(self.filtered_data['reach_s'])

        def return_seq_filtered_data(dA, w, s):
            return (dA, w, s)

        def output(dA, w, s):
            if self.param_dict['run_type'] == 'seq':
                self.filtered_data['reach_dA'], self.filtered_data['reach_w'], self.filtered_data['reach_s'] = return_seq_filtered_data(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
            if self.param_dict['run_type'] == 'set':
                self.filtered_data['reach_dA'][iR], self.filtered_data['reach_w'][iR], self.filtered_data['reach_s'][iR] = return_seq_filtered_data(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
        if self.output['q_algo31'].mask.all() or reach['dA'][iR].mask.all() or reach['w'][iR].mask.all() or reach['s'][iR].mask.all():
            return False
        elif self.output['q_algo31'].mask.any() or reach['dA'][iR].mask.any() or reach['w'][iR].mask.any() or reach['s'][iR].mask.any():
            remove_indices = {}
            keys = ['q_algo31', 'dA', 'w', 's']
            for key in keys:
                if key == 'q_algo31':
                    data_mask = []
                    for i in range(0, len(self.output[key][self.list_to_keep])):
                        data_mask.append(calc.check_na(self.output[key][self.list_to_keep][i]))
                else:
                    data_mask = []
                    for i in range(0, len(reach[key][iR])):
                        data_mask.append(calc.check_na(reach[key][iR][i]))
                i = 0
                for row in data_mask:
                    if row:
                        remove_indices[i] = 1
                    i += 1
            arr_to_remove = []
            for index in remove_indices:
                arr_to_remove.append(index)
            arr_to_remove = np.array(arr_to_remove)
            self.removed_indices = arr_to_remove
            if arr_to_remove.size > 0:
                mask = np.ones(len(self.output['q_algo31'][self.list_to_keep]), dtype=bool)
                mask[arr_to_remove] = False
                self.output['q_algo31_masked'] = self.output['q_algo31'][self.list_to_keep][mask]
                reach['dA'][iR] = reach['dA'][iR][mask]
                reach['w'][iR] = reach['w'][iR][mask]
                reach['s'][iR] = reach['s'][iR][mask]
                keys = ['dA', 'w', 's', 'q_algo31_masked']
                for key in keys:
                    full_nan = 0
                    for t in range(0, len(self.output['q_algo31_masked'])):
                        if not key == 'q_algo31_masked':
                            if calc.check_na(reach[key][iR][t]):
                                full_nan += 1
                        elif calc.check_na(self.output[key][t]):
                            full_nan += 1
                    if full_nan == len(self.output['q_algo31_masked']):
                        output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
                        return False
        if len(self.output['q_algo31_masked']) < self.param_dict['min_obs']:
            logging.info('')
            output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
            return False
        OUTLOG = []
        for ind in range(len(reach['w'][iR])):
            if not params.cython_version:
                if not calc.check_suitability(0.1, abs(reach['dA'][iR].min()) * 1.5, reach['dA'][iR][ind], reach['w'][iR][ind], reach['s'][iR][ind]):
                    OUTLOG += [True]
                else:
                    OUTLOG += [False]
        if np.array(OUTLOG).all():
            logging.info('')
            output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
            return False
        else:
            output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
            return True
        output(reach['dA'][iR], reach['w'][iR], reach['s'][iR])
        return True

    def build_output_q_masked_array(self, str):
        if str == 'q_algo31' or str == 'time':
            mask = np.ones(self.input_data['node_z'].shape[1], dtype=bool)
            arr = np.ones(self.input_data['node_z'].shape[1]) * np.nan
            if np.array(self.list_to_keep).size > 0:
                if len(self.list_to_keep) == len(self.output[str]):
                    arr[self.list_to_keep] = self.output[str]
                else:
                    arr[self.list_to_keep] = self.output[str][self.list_to_keep]
            if np.array(self.removed_indices).size > 0:
                arr[self.removed_indices] = np.nan
        elif str == 'q_algo5':
            mask = np.ones(self.input_data['node_z'].shape[1], dtype=bool)
            arr = np.ones(self.input_data['node_z'].shape[1]) * np.nan
            if np.array(self.list_to_keep).size > 0:
                if len(self.list_to_keep) == len(self.output[str]):
                    arr[self.list_to_keep] = self.output[str]
                else:
                    arr[self.list_to_keep] = self.output[str][self.list_to_keep]
            index_tmp = np.where(self.output[str] <= 0.0)
            arr[index_tmp[0]] = np.nan
        for ind in range(0, len(arr)):
            if not calc.check_na(arr[ind]):
                mask[ind] = False
        q_masked = np.ma.array(arr, mask=mask)
        return q_masked

    def readd_removed_data(self):
        self.output['q_algo31'] = self.build_output_q_masked_array('q_algo31')
        self.output['q_algo5'] = self.build_output_q_masked_array('q_algo5')
        self.output['time'] = self.build_output_q_masked_array('time')

    def slope_computation(self):
        if self.param_dict['use_reach_slope']:
            if self.param_dict['run_type'] == 'set':
                for t in range(self.filtered_data['node_z'][0].shape[0]):
                    SLOPEM1 = np.zeros(self.filtered_data['node_z'][0].shape[0])
                    if self.filtered_data['reach_s'][0][t] > 0.0:
                        if not params.node_length:
                            SLOPEM1[t] = self.filtered_data['reach_s'][0][t] * (self.filtered_data['node_x'][-1] - self.filtered_data['node_x'][0])
                            for i in range(1, self.filtered_data['node_x'].size):
                                pass
                        elif params.node_length:
                            SLOPEM1[t] = self.filtered_data['reach_s'][0][t] * np.sum(self.filtered_data['node_x'])
                    else:
                        if len(self.filtered_data['node_z']) < 2:
                            raise ''
                        if len(self.filtered_data['node_z']) >= 2 and len(self.filtered_data['node_z']) <= 3:
                            SS1 = self.filtered_data['node_z'][0][t]
                            SS2 = self.filtered_data['node_z'][-1][t]
                        if len(self.filtered_data['node_z']) > 3:
                            SS1 = (self.filtered_data['node_z'][0][t] + 2.0 * self.filtered_data['node_z'][1][t] + self.filtered_data['node_z'][2][t]) / 4.0
                            index_x = np.where(abs(self.filtered_data['node_x'][0] - self.filtered_data['node_x'][0:]) < params.DX_Length)
                            index_x = index_x[0]
                            SS2 = (self.filtered_data['node_z'][index_x[-3]][t] + 2.0 * self.filtered_data['node_z'][index_x[-2]][t] + self.filtered_data['node_z'][index_x[-1]][t]) / 4.0
                        SLOPEM1[t] = SS1 - SS2
            if self.param_dict['run_type'] == 'seq':
                if not params.node_length:
                    SLOPEM1 = self.filtered_data['reach_s'] * (self.filtered_data['node_x'][-1] - self.filtered_data['node_x'][0])
                elif params.node_length:
                    SLOPEM1 = self.filtered_data['reach_s'] * (self.filtered_data['node_x'][-1] - self.filtered_data['node_x'][0])
                    self.last_node_for_integral = len(self.filtered_data['node_x'])
        else:
            SLOPEM1 = np.zeros(self.filtered_data['node_z'][0].shape[0])
            for t in range(self.filtered_data['node_z'][0].shape[0]):
                if len(self.filtered_data['node_z']) < 2:
                    logging.info('')
                    self.output['valid'] = 0
                    self.last_node_for_integral = 0
                    return np.ones(self.filtered_data['node_z'][0].shape[0]) * np.nan
                if len(self.filtered_data['node_z']) == 2:
                    SS1 = self.filtered_data['node_z'][0][t]
                    SS2 = self.filtered_data['node_z'][-1][t]
                    self.last_node_for_integral = 2
                if len(self.filtered_data['node_z']) >= 3:
                    SS1 = (self.filtered_data['node_z'][0][t] + 2.0 * self.filtered_data['node_z'][1][t] + self.filtered_data['node_z'][2][t]) / 4.0
                    if abs(self.filtered_data['node_x'][2] - self.filtered_data['node_x'][0]) > params.DX_Length:
                        params.DX_Length = abs(self.filtered_data['node_x'][2] - self.filtered_data['node_x'][0])
                    index_x = np.where(abs(self.filtered_data['node_x'][0] - self.filtered_data['node_x'][0:]) <= params.DX_Length)
                    index_x = index_x[0]
                    self.last_node_for_integral = len(index_x)
                    SS2 = (self.filtered_data['node_z'][index_x[-3]][t] + 2.0 * self.filtered_data['node_z'][index_x[-2]][t] + self.filtered_data['node_z'][index_x[-1]][t]) / 4.0
                SLOPEM1[t] = SS1 - SS2
            if self.param_dict['gnuplot_saving']:
                reach_id = verify_name_length(str(self.input_data['reach_id']))
                output_path = self.param_dict['output_dir'].joinpath('gnuplot_data', reach_id)
                if not Path(output_path).is_dir():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                times2 = np.around(self.input_data['reach_t'] / 3600 / 24)
                times2 = times2 - min(times2)
                gnuplot_save_slope(SLOPEM1, times2, output_path.joinpath('slope'))
        self.intermediates_values['slope'] = SLOPEM1
        return SLOPEM1

    def prepare_output_half_width_elevation_v2(self):
        self.output['half_width'] = []
        self.output['elevation'] = []
        for node in range(len(self.filtered_data['node_x'])):
            tmp_xr = self.input_data['node_xr'][node][:self.lim_node_xr[node]]
            tmp_yr = self.input_data['node_yr'][node][:self.lim_node_xr[node]]
            h0 = self.output['node_a0'][node] / tmp_xr.min() * 2
            self.output['elevation'].append(np.insert(tmp_yr, 0, h0))
            self.output['half_width'].append(np.insert(tmp_xr, 0, tmp_xr[0]))
        self.output['elevation'] = np.array(self.output['elevation'], dtype=object)
        self.output['half_width'] = np.array(self.output['half_width'], dtype=object)
        if 0 < self.removed_nodes_ind.shape[0]:
            arr = np.array(np.full(len(self.output['elevation'][0]), np.nan))
            self.output['elevation'] = np.insert(self.output['elevation'], self.removed_nodes_ind, arr)
            self.output['half_width'] = np.insert(self.output['half_width'], self.removed_nodes_ind, arr)

    def prepare_output_half_width_elevation(self):
        self.output['half_width'] = []
        self.output['elevation'] = []
        for node in range(len(self.input_data['node_xr'])):
            h0 = self.output['node_a0'][node] / self.input_data['node_xr'][node].min() * 2
            self.output['elevation'].append(np.insert(self.input_data['node_yr'][node], 0, h0))
            self.output['half_width'].append(np.insert(self.input_data['node_xr'][node], 0, self.input_data['node_xr'][node][0]))
        self.output['elevation'] = np.array(self.output['elevation'], dtype=object)
        self.output['half_width'] = np.array(self.output['half_width'], dtype=object)
        if 0 < self.removed_nodes_ind.shape[0]:
            arr = np.array(np.full(len(self.output['elevation'][0]), np.nan))
            self.output['elevation'] = np.insert(self.output['elevation'], self.removed_nodes_ind, arr)
            self.output['half_width'] = np.insert(self.output['half_width'], self.removed_nodes_ind, arr)
