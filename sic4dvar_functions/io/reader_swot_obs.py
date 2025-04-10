import copy
import pathlib
from datetime import datetime
from typing import Tuple, Dict, Literal
import netCDF4 as nc4
import numpy as np
import pandas as pd
from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array, check_shape, array_as_row_vector, array_as_col_vector, datetime_array_set_to_freq_and_filter

def get_vars_from_swot_nc(reach_ids: Tuple[str | int, ...], swot_file_pattern: str | pathlib.PurePath, node_vars: Tuple[str, ...]=(), reach_vars: Tuple[str, ...]=(), val_swot_q_flag: Tuple[int, ...]=SIC4DVarLowCostDefaults().def_val_swot_q_flag, freq_datetime: str=SIC4DVarLowCostDefaults().def_freq_datetime, dup_datetime: Literal['drop', 'raise']='raise', start_datetime: datetime | float | int | None=None, end_datetime: datetime | float | int | None=None, clean_run: bool=False, debug_mode: bool=False) -> Tuple[Dict[Dict, Dict], datetime]:
    if clean_run:
        debug_mode = False
    if debug_mode:
        clean_run = False
    msg = 'loading SWOT observations'
    if not clean_run:
        print(f'\n{msg}')
    if len(node_vars) == 0 and len(reach_vars) == 0:
        raise TypeError('must specify either node_vars or reach_vars')
    node_vars1 = list(copy.deepcopy(node_vars))
    reach_vars1 = list(copy.deepcopy(reach_vars))
    if len(node_vars1) > 0:
        if 'node_q' not in node_vars1:
            node_vars1.append('node_q')
    if len(reach_vars1) > 0:
        if 'reach_q' not in reach_vars1:
            reach_vars1.append('reach_q')
    reach_ids = [int(i) for i in reach_ids]
    if len(reach_ids) == 0:
        raise TypeError('must specify at lest one reach_id')
    reach_path_dict = {}
    swot_file_pattern = str(swot_file_pattern)
    for reach_id in reach_ids:
        reach_file_path = pathlib.Path(swot_file_pattern.format(reach_id))
        if not reach_file_path.exists():
            raise FileNotFoundError(f'swot file for reach {reach_id} {reach_file_path} does not exist')
        reach_path_dict[reach_id] = reach_file_path
    swot_dict_arrays = {'node': {k_: [] for k_ in node_vars1}, 'reach': {k_: [] for k_ in reach_vars1}}
    swot_list_var = list(node_vars1) + list(reach_vars1)
    ref_datetime = SIC4DVarLowCostDefaults().def_ref_datetime
    for r_i in reach_ids:
        reach_msg = f'loading {swot_list_var} data from SWOT for reach {r_i}'
        if debug_mode:
            print(reach_msg)
        with nc4.Dataset(reach_path_dict[r_i]) as swot_ds:
            ref_datetime = datetime.fromisoformat(swot_ds['reach']['time'].units.split('since ')[1])
            reach_times = masked_array_to_nan_array(swot_ds['reach']['time'][:])
            if any([start_datetime is not None, end_datetime is not None, freq_datetime]):
                time_mask_bool, time_as_datetime, time_as_sec_array = datetime_array_set_to_freq_and_filter(data_dt=reach_times, ref_datetime=ref_datetime, freq_datetime=freq_datetime, duplicates=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime)
                time_ids = np.nonzero(time_mask_bool)[0]
            else:
                time_ids = np.array(range(reach_times.size), dtype=np.int32)
                time_as_sec_array = np.array(np.round(reach_times, 0), dtype=np.int64)
            total_n_nodes = swot_ds['node']['node_id'][:].size
            total_n_times = swot_ds['reach']['time'][:].size
            for node_var in swot_dict_arrays['node'].keys():
                swot_var = swot_ds['node'][node_var]
                swot_var_array = masked_array_to_nan_array(swot_var[:])
                if node_var not in ['reach_id', 'node_id']:
                    swot_var_array = check_shape(swot_var_array, expected_shape=(total_n_nodes, total_n_times), force_shape=False)
                    swot_var_array = swot_var_array[:, time_ids]
                elif node_var == 'reach_id':
                    swot_var_array.shape = (1, 1)
                else:
                    swot_var_array = array_as_col_vector(swot_var_array)
                swot_dict_arrays['node'][node_var].append(swot_var_array)
            for reach_var in swot_dict_arrays['reach'].keys():
                swot_var = swot_ds['reach'][reach_var]
                swot_var_array = masked_array_to_nan_array(swot_var[:])
                if reach_var != 'reach_id':
                    swot_var_array = array_as_row_vector(swot_var_array)
                    swot_var_array = swot_var_array[:, time_ids]
                else:
                    swot_var_array.shape = (1, 1)
                if reach_var == 'time':
                    swot_var_array = time_as_sec_array
                swot_dict_arrays['reach'][reach_var].append(swot_var_array)
            if debug_mode:
                print(f'total number of nodes  {total_n_nodes}, total number of time instances: {total_n_times}')
        if debug_mode:
            print(msg, 'done')
    try:
        node_q_list = swot_dict_arrays['node']['node_q']
    except KeyError:
        pass
    else:
        for k_ in node_vars:
            if k_ in ['node_q', 'node_id']:
                continue
            for v_n, v_array in enumerate(swot_dict_arrays['node'][k_]):
                swot_dict_arrays['node'][k_][v_n] = np.where(np.isin(node_q_list[v_n], val_swot_q_flag), v_array, np.nan)
    try:
        reach_q_list = swot_dict_arrays['reach']['reach_q']
    except KeyError:
        pass
    else:
        for k_ in reach_vars:
            if k_ in ['reach_q', 'reach_id']:
                continue
            for v_n, v_array in enumerate(swot_dict_arrays['reach'][k_]):
                swot_dict_arrays['reach'][k_][v_n] = np.where(np.isin(reach_q_list[v_n], val_swot_q_flag), v_array, np.nan)
    if 'node_q' in swot_dict_arrays['node'].keys() and 'node_q' not in node_vars:
        swot_dict_arrays['node'].pop('node_q')
    if 'reach_q' in swot_dict_arrays['reach'].keys() and 'reach_q' not in reach_vars:
        swot_dict_arrays['reach'].pop('reach_q')
    swot_dict_arrays = {'node': {k_: tuple(v_) for k_, v_ in swot_dict_arrays['node'].items()}, 'reach': {k_: tuple(v_) for k_, v_ in swot_dict_arrays['reach'].items()}}
    if not clean_run:
        print(msg, 'done')
    return (swot_dict_arrays, ref_datetime)

def get_array_dict_from_swot_nc(reach_ids: Tuple[str | int, ...], swot_file_pattern: str | pathlib.PurePath, use_node_z: bool, use_node_w: bool, use_reach_slope: bool, compute_swot_q: bool, use_uncertainty: bool, val_swot_q_flag: Tuple[int, ...]=SIC4DVarLowCostDefaults().def_val_swot_q_flag, freq_datetime: str=SIC4DVarLowCostDefaults().def_freq_datetime, dup_datetime: Literal['drop', 'raise']='raise', start_datetime: datetime | float | int | None=None, end_datetime: datetime | float | int | None=None, clean_run: bool=False, debug_mode: bool=False) -> Tuple[Dict, datetime]:
    if clean_run:
        debug_mode = False
    if debug_mode:
        clean_run = False
    swot_lut = pd.DataFrame(index=['t', 'node_z', 'node_w', 'node_z_u', 'node_w_u', 'node_id', 'reach_s', 'reach_da', 'reach_w'], columns=['reach', 'node'])
    swot_lut.loc['t', 'reach'] = 'time'
    swot_lut.loc['node_z', 'node'] = 'wse'
    swot_lut.loc['node_w', 'node'] = 'width'
    swot_lut.loc['node_z_u', 'node'] = 'wse_u'
    swot_lut.loc['node_w_u', 'node'] = 'width_u'
    swot_lut.loc['node_id', 'node'] = 'node_id'
    swot_lut.loc['reach_s', 'reach'] = 'slope2'
    swot_lut.loc['reach_da', 'reach'] = 'd_x_area'
    swot_lut.loc['reach_w', 'reach'] = 'width'
    swot_node_list_var, swot_reach_list_var = (['node_id'], ['t'])
    if use_node_z:
        swot_node_list_var.append('node_z')
    if use_node_w:
        swot_node_list_var.append('node_w')
    if use_uncertainty:
        if use_node_z:
            swot_node_list_var.append('node_z_u')
        if use_node_w:
            swot_node_list_var.append('node_w_u')
    if use_reach_slope or compute_swot_q:
        swot_reach_list_var.append('reach_s')
    if compute_swot_q:
        swot_reach_list_var.extend(['reach_da', 'reach_w'])
    swot_dict_arrays = {k_: [] for k_ in swot_node_list_var + swot_reach_list_var}
    swot_dict, ref_datetime = get_vars_from_swot_nc(reach_ids=reach_ids, swot_file_pattern=swot_file_pattern, node_vars=swot_lut.loc[swot_node_list_var][['node']].to_numpy().flatten(), reach_vars=swot_lut.loc[swot_reach_list_var][['reach']].to_numpy().flatten(), val_swot_q_flag=val_swot_q_flag, freq_datetime=freq_datetime, dup_datetime=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime, clean_run=clean_run, debug_mode=debug_mode)
    for k_i, tup_arr_i in swot_dict['node'].items():
        swot_dict_arrays[swot_lut.loc[swot_lut['node'] == k_i].index[0]] = list(tup_arr_i)
    for k_i, tup_arr_i in swot_dict['reach'].items():
        swot_dict_arrays[swot_lut.loc[swot_lut['reach'] == k_i].index[0]] = list(tup_arr_i)
    for t_n, t_arr in enumerate(swot_dict_arrays['t']):
        swot_dict_arrays['t'][t_n] = np.round(t_arr).astype(np.int64)
    for n_n, n_arr in enumerate(swot_dict_arrays['node_id']):
        swot_dict_arrays['node_id'][n_n] = n_arr.astype(np.int64)
    return (swot_dict_arrays, ref_datetime)
if __name__ == '__main__':
    base_path = pathlib.Path('C:\\Users\\isadora.rezende\\PhD')
    test_output, _ = get_array_dict_from_swot_nc(reach_ids=(74265000081,), swot_file_pattern=base_path / 'Datasets' / 'PEPSI' / 'Ohio' / '{}_SWOT.nc', use_node_z=True, use_node_w=False, use_reach_slope=False, compute_swot_q=False, use_uncertainty=False, freq_datetime='3D', start_datetime=datetime(2010, 10, 1), clean_run=False, debug_mode=False)
    for k, v in test_output.items():
        print(k, [v_i.shape for v_i in v])