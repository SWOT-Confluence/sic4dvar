"""
Created on September 11th 2023 at 16:00
by @Isadora Silva

Last modified on May 15th 2024 at 14:00
by @Isadora Silva

@authors: Isadora Silva
"""

import copy
import itertools
import math
import operator
from datetime import datetime
from typing import Iterable, Tuple, Any, Literal

import numpy as np
import pandas as pd

from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults
from sic4dvar_functions.helpers.helpers_generic import pairwise, dt_as_pd_datetime, PdTimeDeltaFreq


def iterable_to_flattened_array(
        my_iterable: Iterable,
) -> np.ndarray:
    
    my_array1 = copy.deepcopy(my_iterable)

    if (not isinstance(my_array1, np.ndarray)) or not (isinstance(my_array1, np.ma.MaskedArray)):
        my_array1 = np.array(my_array1)
    else:
        my_array1 = my_array1

    if my_array1.ndim != 1:
        my_array1 = my_array1.flatten()

    return my_array1


def masked_array_to_nan_array(
        my_array: np.ma.MaskedArray,
):
    my_array1 = copy.deepcopy(my_array)

    if my_array1.dtype == np.int32 or my_array1.dtype == np.int64:

        if my_array1.size == 0:
            my_array1 = np.empty(0, dtype=np.float32)
            return my_array1

        if isinstance(my_array1, np.ma.MaskedArray):
            max_value = np.ma.max(my_array1)
        else:
            max_value = np.nanmax(my_array1)

        if my_array1.dtype == np.int32:
            
            my_array1 = my_array1.astype(np.float32)
            return my_array1

        if my_array1.dtype == np.int64:
            
            my_array1 = my_array1.astype(np.float64)
            return my_array1

    if isinstance(my_array1, np.ma.MaskedArray):
        my_array1 = my_array1.filled(np.nan)

    return my_array1


def nan_array_to_masked_array(
        my_array: np.ndarray | np.ma.MaskedArray,
        fill_value: None | float | int = None,
) -> np.ma.MaskedArray:
    
    my_array1 = copy.deepcopy(my_array)

    if not isinstance(my_array1, np.ma.MaskedArray):
        my_array1 = np.ma.masked_invalid(my_array1)
    if fill_value is not None:
        my_array1.set_fill_value(fill_value)
    return my_array1


def arrays_rmv_nan_pair(
        x0: Iterable,
        y0: Iterable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    x, y = iterable_to_flattened_array(x0), iterable_to_flattened_array(y0)

    x = masked_array_to_nan_array(x)
    y = masked_array_to_nan_array(y)

    df = pd.DataFrame({'x': x, 'y': y})

    df.dropna(axis=0, how='any', inplace=True)

    x, y, i = np.array(df['x']), np.array(df['y']), np.array(df.index, dtype=np.int32)

    return x, y, i


def arrays_rmv_next_same(
        x0: np.ndarray | Iterable,
        y0: np.ndarray | Iterable,
        float_atol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    x, y = [iterable_to_flattened_array(i_) for i_ in [x0, y0]]

    x_is_int = True if np.issubdtype(x.dtype, np.integer) else False
    y_is_int = True if np.issubdtype(y.dtype, np.integer) else False

    no_dup_x_list = [x[0]]
    no_dup_y_list = [y[0]]
    no_dup_n_list = [0]

    n = 0
    for x_i, y_i in zip(x[1:], y[1:]): 
        n += 1

        if x_is_int:
            x_comp = x_i == no_dup_x_list[-1]
        else:
            x_comp = math.isclose(x_i, no_dup_x_list[-1], abs_tol=float_atol)

        if x_comp:

            if y_is_int:
                y_comp = y_i == no_dup_y_list[-1]
            else:
                y_comp = math.isclose(y_i, no_dup_y_list[-1], abs_tol=float_atol)

            if y_comp:
                continue

        no_dup_x_list.append(x_i)
        no_dup_y_list.append(y_i)
        no_dup_n_list.append(n)

    x = np.array(no_dup_x_list)
    y = np.array(no_dup_y_list)
    n = np.array(no_dup_n_list, dtype=np.int32)

    return x, y, n


def sort_by_distance_to_value(
        my_array: np.ndarray | Iterable,
        value: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)

    my_array_1 = copy.deepcopy(my_array)

    difs = np.abs(my_array_1 - value)

    ids = np.argsort(difs)

    my_array_1 = my_array_1[ids]

    return ids, my_array_1


def find_nearest(
        my_array: np.ndarray | Iterable,
        value,
) -> Tuple[int, float]:
    
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)
    i = (np.abs(my_array - value)).argmin()
    v = my_array[i]
    return i, v


def find_n_nearest(
        my_array: np.ndarray | Iterable,
        value: Any,
        n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    ids, my_array = sort_by_distance_to_value(my_array, value)

    return ids[:n], my_array[:n]


def find_normalized_nearest(
        my_array0: np.ndarray,
        value: float,
        norm_diff_thr: float = 0.1,
        min_n: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    
    my_array = copy.deepcopy(my_array0).flatten()

    n_valid_pts = np.count_nonzero(np.isfinite(my_array))

    req_n_pts = min(max(1, min_n), n_valid_pts)

    diff_array = np.full_like(my_array, fill_value=np.nan)
    diff_array[np.isfinite(my_array)] = np.abs(my_array[np.isfinite(my_array)] - value)

    min_, max_ = np.nanmin(diff_array), np.nanmax(diff_array)

    if min_ == max_:
        diff_array[np.isfinite(diff_array)] = 0.
    else:
        diff_array = ((diff_array - min_) / (max_ - min_))

    idxs_array = np.nonzero(diff_array.flatten() <= norm_diff_thr)[0]

    if idxs_array.size < req_n_pts:
        idxs_array = np.argsort(diff_array.flatten())[:req_n_pts]

    return idxs_array, 1 - diff_array.flatten()[idxs_array]


def find_relative_nearest(
        my_array0: np.ndarray,
        value: float,
        rel_diff_thr: float = 0.1,
        min_n: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    
    my_array = copy.deepcopy(my_array0).flatten()

    n_valid_pts = np.count_nonzero(np.isfinite(my_array))

    req_n_pts = min(max(1, min_n), n_valid_pts)

    diff_array = np.full_like(my_array, fill_value=np.nan)
    diff_array[np.isfinite(my_array)] = np.abs(my_array[np.isfinite(my_array)] - value)

    diff_array /= value

    idxs_array = np.nonzero(diff_array.flatten() <= rel_diff_thr)[0]

    if idxs_array.size < req_n_pts:
        idxs_array = np.argsort(diff_array.flatten())[:req_n_pts]

    return idxs_array, np.exp(-diff_array[idxs_array].flatten())


def _arrays_de_in_crease(
        my_array: np.ndarray | Iterable,
        comp_op: operator,
        check: bool,
        force: bool,
        remove_nan: bool = True,
) -> np.ndarray | bool:
    
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)

    # make a copy of the array
    my_array1 = copy.deepcopy(my_array)

    if remove_nan:
        my_array1 = my_array1[np.isfinite(my_array1)]

    for n_0i, n_1i in pairwise(range(my_array1.shape[0])):

        v_0i, v_1i = my_array1[n_0i], my_array1[n_1i]

        if comp_op(v_1i, v_0i):
            if check:
                return False
            my_array1[n_1i] = v_0i

    if check:
        return True

    return my_array1


def arrays_check_decrease(
        my_array: np.ndarray | Iterable,
        remove_nan: bool = True,
) -> bool:
    return _arrays_de_in_crease(my_array, comp_op=operator.gt, check=True, force=False, remove_nan=remove_nan)


def arrays_check_increase(
        my_array: np.ndarray | Iterable,
        remove_nan: bool = True,
) -> bool:
    return _arrays_de_in_crease(my_array, comp_op=operator.lt, check=True, force=False, remove_nan=remove_nan)


def arrays_force_decrease(
        my_array: np.ndarray | Iterable,
        remove_nan: bool = True,
) -> np.ndarray:
    return _arrays_de_in_crease(my_array, comp_op=operator.gt, check=False, force=True, remove_nan=remove_nan)


def arrays_force_increase(
        my_array: np.ndarray | Iterable,
        remove_nan: bool = True,
) -> np.ndarray:
    return _arrays_de_in_crease(my_array, comp_op=operator.lt, check=False, force=True, remove_nan=remove_nan)


def get_mask_nan_across_arrays(
        *my_arrays: np.ndarray | np.ma.MaskedArray,
) -> np.ndarray:
    expected_shape = None
    sum_array = np.empty(0)

    if not isinstance(my_arrays, tuple):
        my_arrays = (my_arrays,)

    for arr in my_arrays:

        arr1 = copy.deepcopy(arr)

        arr1 = masked_array_to_nan_array(arr1)

        if expected_shape is None:
            expected_shape = arr.shape
            sum_array = copy.deepcopy(arr1)
            continue

        sum_array += arr1

        if np.any(np.isfinite(sum_array)):
            sum_array /= np.nanmin(sum_array)

    nan_bool_array = np.isnan(sum_array)

    return nan_bool_array


def get_index_valid_data(
        *my_arrays: np.ndarray | np.ma.MaskedArray,
        drop_na_how: Literal["any", "all"] = "any",
        axis: int | tuple = (0, 1),
) -> tuple:
    
    if isinstance(axis, int):
        axis = (axis,)

    if not isinstance(my_arrays, tuple):
        my_arrays = (my_arrays,)

    nan_bool_array = get_mask_nan_across_arrays(*my_arrays)

    expected_shape = nan_bool_array.shape

    tmp_array = np.full_like(nan_bool_array, fill_value=1., dtype=np.float32)
    tmp_array[nan_bool_array] = np.nan

    if not np.any(nan_bool_array):
        idx_dim0 = np.array(range(expected_shape[0]))

        if len(expected_shape) == 1:
            return (idx_dim0,)

        return idx_dim0, np.array(range(expected_shape[1]))

    if np.all(nan_bool_array):
        return (np.array([], dtype=np.int32),) * len(expected_shape)

    if len(expected_shape) == 1:
        return ((~nan_bool_array).nonzero()[0],)

    df_valid = pd.DataFrame(copy.deepcopy(tmp_array))

    for ax in axis:
        df_valid = df_valid.dropna(axis=ax, how=drop_na_how)

    if df_valid.empty:
        return (np.array([], dtype=np.int32),) * len(expected_shape)

    return np.array(df_valid.index, dtype=np.int32), np.array(df_valid.columns, dtype=np.int32)


def array_as_row_vector(
        my_array: np.ndarray | np.ma.MaskedArray,
):
    my_vector = copy.deepcopy(my_array)
    try:
        my_vector.shape = (1, my_array.size)
    except AttributeError as ae:
        if "'list' object has no attribute 'size'" in ae.args[0]:
            my_vector = np.array(my_vector)
            my_vector.shape = (my_vector.size, 1)
        else:
            raise ae
    return my_vector


def array_as_col_vector(
        my_array: np.ndarray | np.ma.MaskedArray,
):
    my_vector = copy.deepcopy(my_array)
    try:
        my_vector.shape = (my_array.size, 1)
    except AttributeError as ae:
        if "'list' object has no attribute 'size'" in ae.args[0]:
            my_vector = np.array(my_vector)
            my_vector.shape = (my_vector.size, 1)
        else:
            raise ae
    return my_vector


def check_shape(
        my_array: np.ndarray | np.ma.MaskedArray,
        expected_shape: tuple,
        force_shape: bool = False,
):
    if (not isinstance(my_array, np.ndarray)) or (not isinstance(my_array, np.ma.MaskedArray)):
        my_array = np.array(my_array)

    init_shape = my_array.shape

    if isinstance(expected_shape, int):
        expected_shape = (expected_shape,)

    if init_shape == expected_shape:
        return copy.deepcopy(my_array)

    if len(expected_shape) == 1:
        is_vector = False
        is_1d = True
    elif len(expected_shape) == 2:
        is_vector = np.any([i_ == 1 for i_ in expected_shape])
        is_1d = False

    if not (is_1d or is_vector):
        print("debug helper")
        print(my_array.shape)

    if is_1d:
        major_size = expected_shape[0]
    else:
        try:
            major_size = [i_ for i_ in expected_shape if i_ != 1][0]
        except IndexError:
            major_size = 1

    tmp_match_dim = [i_ == major_size for i_ in init_shape]

    if not np.any(tmp_match_dim):
        pass

    if len(tmp_match_dim) == 1:
        my_array_out = copy.deepcopy(my_array)

        if is_1d:
            pass

    else:

        if np.all(tmp_match_dim):
            pass

        elif tmp_match_dim[0]:
            my_array_out = copy.deepcopy(my_array[:, 0])

        else:
            my_array_out = copy.deepcopy(my_array[0, :])

    if is_vector:
        if expected_shape[0] == 1:
            my_array_out = array_as_row_vector(my_array_out)
        elif expected_shape[1] == 1:
            my_array_out = array_as_col_vector(my_array_out)
        else:
            pass

    # sanity check
    if my_array_out.shape != expected_shape:
        pass

    return my_array_out


def array_filter_by_index(
        my_array: np.ndarray,
        row_0_ids: np.ndarray | None = None,
        col_0_ids: np.ndarray | None = None,
):
    if my_array.size == 0:
        return copy.deepcopy(my_array)

    if my_array.ndim > 2:
        pass

    row_ids = copy.deepcopy(row_0_ids)
    col_ids = copy.deepcopy(col_0_ids)

    if row_ids is None and col_ids is None:
        pass

    if row_ids is not None:
        row_ids = row_ids.astype(np.int32)
        row_ids = row_ids.flatten()
    if col_ids is not None:
        col_ids = col_ids.astype(np.int32)
        col_ids = col_ids.flatten()

    if my_array.ndim == 1:
        if (row_ids is not None) and (col_ids is not None):
            pass

        for id_array in row_ids, col_ids:
            if id_array is not None:
                if id_array.ndim != 1:
                    if np.any([i_ == 1 for i_ in id_array.shape]):
                        id_array = id_array.flatten()
                    else:
                        pass

                return copy.deepcopy(my_array[id_array])

    if row_ids is None or np.array_equal(row_ids, np.array(range(my_array.shape[0]))):
        out_array = copy.deepcopy(my_array)
        row_ids = np.array(range(my_array.shape[0]))
    else:
        try:
            out_array = copy.deepcopy(my_array[row_ids, :])
        except IndexError as ie:
            print('debugger helper')
            print('array', my_array.shape)
            print(my_array)
            print('row ids', row_ids.shape)
            print(row_ids)
            raise ie

    if col_ids is None or np.array_equal(col_ids, np.array(range(my_array.shape[1]))):
        col_ids = np.array(range(my_array.shape[1]))
    else:
        try:
            out_array = out_array[:, col_ids]
        except IndexError as ie:
            print('debugger helper')
            print('array', out_array.shape)
            print(out_array)
            print('col ids', col_ids.shape)
            print(col_ids)
            raise ie

    return check_shape(out_array, (row_ids.size, col_ids.size), force_shape=True)


def array_fill_with_nan_from_ids(
        my_array: np.ndarray,
        valid_0_ids: np.ndarray,
        expected_shape: tuple | None,
):
    if not isinstance(valid_0_ids, np.ndarray):
        valid_0_ids = np.array(valid_0_ids)

    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)

    if my_array.ndim != valid_0_ids.ndim:
        pass

    if valid_0_ids.size > 0:
        if valid_0_ids.ndim == 1:
            if expected_shape is None:
                expected_shape = (np.max(valid_0_ids) + 1,)
        else:
            if expected_shape is None:
                expected_shape = (np.max(valid_0_ids[:, 0]) + 1, np.max(valid_0_ids[:, 1]) + 1)
    else:
        if expected_shape is None:
            pass

    expected_size = 1
    for i in expected_shape:
        expected_size *= i

    if valid_0_ids.size > 0:
        if valid_0_ids.ndim == 1:
            valid_flat_ids = copy.deepcopy(valid_0_ids)
        else:
            # Converts a tuple of index arrays into an array of flat indices
            valid_flat_ids = []
            for pair_ids in valid_0_ids:
                try:
                    valid_flat_ids.append(np.ravel_multi_index(pair_ids, dims=expected_shape))
                except ValueError as ve:
                    print('debug helper')
                    print('expected_shape', expected_shape)
                    print('pair ids', pair_ids)
                    raise ve
            valid_flat_ids = np.array(valid_flat_ids)
    else:
        valid_flat_ids = np.empty(0, dtype=np.int32)

    out_array = np.full(expected_size, fill_value=np.nan, dtype=np.float32)
    out_array[valid_flat_ids] = my_array.flatten()  # flatten already copies the array

    out_array.shape = expected_shape

    return check_shape(out_array, expected_shape, force_shape=True)


def array_fix_next_same(
        my_array: np.ndarray,
        float_atol: float = SIC4DVarLowCostDefaults().def_float_atol,
) -> np.ndarray:
    out_array = copy.deepcopy(my_array)

    is_next_same = True
    while is_next_same:

        is_next_same = False

        for (n0, v0), (n1, v1) in pairwise(enumerate(out_array)):

            if np.isclose(v0, v1, rtol=0., atol=float_atol):
                out_array[n1] = v1 + float_atol
                is_next_same = True

    return out_array


def arrays_bounds(
        ref_array: np.ndarray,
        value0_low_bound_array: np.ndarray | np.ma.MaskedArray | None = None,
        value0_up_bound_array: np.ndarray | np.ma.MaskedArray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    def_low_array, def_up_array = [
        np.full_like(ref_array, fill_value=inf_, dtype=ref_array.dtype) for inf_ in [-np.inf, np.inf]]

    if value0_low_bound_array is None:
        value_0_low_bound_array = def_low_array

    else:
        value_0_low_bound_array = check_shape(
            masked_array_to_nan_array(value0_low_bound_array),
            expected_shape=ref_array.shape,
            force_shape=False)
        value_0_low_bound_array[np.isnan(value_0_low_bound_array)] = -np.inf

    if value0_up_bound_array is None:
        value_0_up_bound_array = def_up_array

    else:
        value_0_up_bound_array = check_shape(
            masked_array_to_nan_array(value0_up_bound_array),
            expected_shape=ref_array.shape,
            force_shape=False)
        value_0_up_bound_array[np.isnan(value_0_up_bound_array)] = np.inf

    return value_0_low_bound_array, value_0_up_bound_array


class DataDateTimeArrays(PdTimeDeltaFreq):
    def __init__(
            self,
            data_dt: np.ndarray | pd.Series | tuple | list,
            ref_datetime: datetime,
            freq_datetime: str):

        super().__init__(freq_datetime=freq_datetime)

        data_dt = np.array(data_dt).flatten()

        if not isinstance(data_dt[0], np.datetime64):
            data_dt = pd.to_timedelta(data_dt, unit='second') + ref_datetime

        self._data_dt_timerange: pd.DatetimeIndex = pd.to_datetime(data_dt).round(self._freq_dt_str)

        self._data_dt_sec_array = np.array((self._data_dt_timerange - ref_datetime).total_seconds(), dtype=np.int64)

    @property
    def data_dt_timerange(self) -> pd.DatetimeIndex:
        return copy.deepcopy(self._data_dt_timerange)

    @property
    def data_dt_sec_array(self) -> np.ndarray:
        return copy.deepcopy(self._data_dt_sec_array)


def datetime_create_time_range(
        start_datetime: datetime | float | int,
        end_datetime: datetime | float | int,
        ref_datetime: datetime,
        freq_datetime: str,
):
    freq_o = PdTimeDeltaFreq(freq_datetime=freq_datetime)
    freq_dt_int, freq_dt_str, freq_datetime = freq_o.freq_datetime_int, freq_o.freq_datetime_str, freq_o.freq_datetime

    start_datetime = dt_as_pd_datetime(dt=start_datetime, ref_datetime=ref_datetime, pd_freq=freq_dt_str)
    end_datetime = dt_as_pd_datetime(dt=end_datetime, ref_datetime=ref_datetime, pd_freq=freq_dt_str)

    stat_dt = start_datetime.normalize() 
    end_dt = end_datetime.normalize() + ( 
            pd.to_timedelta(1, unit="D") + pd.to_timedelta(freq_dt_int, unit=freq_dt_str))

    all_dt_timerange = pd.date_range(start=stat_dt, end=end_dt, freq=freq_datetime, inclusive="both", )
    all_dt_timerange = all_dt_timerange[all_dt_timerange <= end_datetime]

    return all_dt_timerange


def datetime_array_set_to_freq_and_filter(
        data_dt: np.ndarray | pd.Series | tuple | list,
        ref_datetime: datetime,
        freq_datetime: str,
        duplicates: Literal["drop", "keep", "raise"] = "raise",
        start_datetime: datetime | float | int | None = None,
        end_datetime: datetime | float | int | None = None,
):
    
    dt_o = DataDateTimeArrays(
        data_dt=data_dt,
        ref_datetime=ref_datetime,
        freq_datetime=freq_datetime,
    )
    freq_dt_int, freq_dt_str, freq_datetime = dt_o.freq_datetime_int, dt_o.freq_datetime_str, dt_o.freq_datetime
    data_dt_timerange = dt_o.data_dt_timerange
    data_dt_sec_array = dt_o.data_dt_sec_array
    del data_dt, dt_o

    start_datetime = dt_as_pd_datetime(dt=start_datetime, ref_datetime=ref_datetime, pd_freq=freq_dt_str)
    end_datetime = dt_as_pd_datetime(dt=end_datetime, ref_datetime=ref_datetime, pd_freq=freq_dt_str)

    mask_bool = np.full(len(data_dt_timerange), fill_value=True, dtype=bool)
    if start_datetime is not None:
        mask_bool = data_dt_timerange >= start_datetime
    if end_datetime is not None:
        mask_bool = mask_bool & (data_dt_timerange <= end_datetime)
    data_dt_timerange = data_dt_timerange[mask_bool]
    data_dt_sec_array = data_dt_sec_array[mask_bool]

    if data_dt_timerange.size == 0:
        return mask_bool, np.empty(0, dtype=np.datetime64), np.empty(0, dtype=np.int64)

    mask_id0 = np.argmax(mask_bool)

    if freq_datetime.startswith("1s"):
        return mask_bool, data_dt_timerange, data_dt_sec_array

    all_dt_timerange = datetime_create_time_range(
        start_datetime=data_dt_timerange[0], end_datetime=data_dt_timerange[-1],
        ref_datetime=ref_datetime, freq_datetime=freq_datetime, )

    all_dt_sec_array = np.array((all_dt_timerange - ref_datetime).total_seconds(), dtype=np.int64)

    bisect_low_id, keep_ids = 0, []
    for n_sec, t_sec in enumerate(data_dt_sec_array):
        all_id = np.argmin(np.abs(all_dt_sec_array[bisect_low_id:] - t_sec)) + bisect_low_id
        
        if n_sec > 0 and all_id == bisect_low_id:
            if duplicates == "keep":
                pass
            elif duplicates == "drop":
                mask_bool[mask_id0 + n_sec] = False
                continue
            else:
                pass
        keep_ids.append(all_id)
        bisect_low_id = all_id

    n_val_elems = np.count_nonzero(mask_bool)
    n_kept_elems = len(keep_ids)
    if n_val_elems != n_kept_elems:
        pass

    all_dt_timerange = all_dt_timerange[keep_ids]
    all_dt_sec_array = all_dt_sec_array[keep_ids]

    return mask_bool, all_dt_timerange, all_dt_sec_array


if __name__ == "__main__":
    test_array = np.array(range(4))
    print('1D array', test_array)
    print('as row vector:', array_as_row_vector(test_array))
    print('as column vector:', array_as_col_vector(test_array))

    test_val_array = np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
    test_val_2d_ids = np.array([[r_n, t_n] for r_n, t_n in itertools.product([0, 1, 2], [6, 7, 8])])
    print(array_fill_with_nan_from_ids(test_val_array, valid_0_ids=test_val_2d_ids, expected_shape=None, ))
    test_val_2d_ids = np.array([[r_n, t_n] for r_n, t_n in itertools.product([5, 6, 7], [1, 2, 3])])
    print(array_fill_with_nan_from_ids(test_val_array, valid_0_ids=test_val_2d_ids, expected_shape=None, ))
    test_val_2d_ids = np.array([[r_n, t_n] for r_n, t_n in itertools.product([1, 3, 6], [0, 2, 5])])
    print(array_fill_with_nan_from_ids(test_val_array, valid_0_ids=test_val_2d_ids, expected_shape=None, ))
