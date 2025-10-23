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

import pathlib
from typing import Tuple
import netCDF4 as nc4
import numpy as np
import pandas as pd
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array

def get_vars_from_iris_file(reach_ids: Tuple[str | int, ...] | None, iris_file_path: str | pathlib.PurePath, reach_vars: Tuple[str, ...], raise_missing_reach: bool=True, return_m_m: bool=False, clean_run: bool=False) -> pd.DataFrame:
    msg = 'loading data from IRIS'
    if not clean_run:
        print(f'\n{msg}')
    if reach_ids is not None:
        reach_ids = np.array([int(i) for i in reach_ids])
    iris_dict = {k: None for k in reach_vars}
    with nc4.Dataset(iris_file_path) as iris_ds:
        iris_reach_ids = iris_ds['reach_id'][:]
        if reach_ids is None:
            reach_ids = iris_reach_ids
        iris_reach_index_list = []
        for reach_id in reach_ids:
            try:
                iris_reach_index_list.append(np.nonzero(iris_reach_ids == reach_id)[0][0])
            except IndexError:
                if raise_missing_reach:
                    raise KeyError(f'reach {reach_id} missing in IRIS file')
                iris_reach_index_list.append(None)
        for reach_var in iris_dict.keys():
            reach_var_array = np.full(len(iris_reach_index_list), fill_value=np.nan)
            iris_reach_var = iris_ds[reach_var]
            for n_, iris_reach_id in enumerate(iris_reach_index_list):
                if iris_reach_id is None:
                    continue
                reach_var_array[n_] = masked_array_to_nan_array(iris_reach_var[iris_reach_id])
            iris_dict[reach_var] = reach_var_array
    if return_m_m:
        for k in iris_dict.keys():
            iris_dict[k] = iris_dict[k] * 1e-06
    df_iris = pd.DataFrame(iris_dict)
    df_iris['reach_id'] = np.array(reach_ids)
    df_iris.set_index('reach_id', inplace=True)
    if not clean_run:
        print(msg, 'done')
    return df_iris
if __name__ == '__main__':
    iris_slope_opts_test = ('across', 'combined', 'along')
    df_test = get_vars_from_iris_file(reach_ids=(21406100031, 21406100041, 21406100051), iris_file_path='C:\\Users\\isadora.rezende\\PhD\\Datasets\\Dahiti\\Slope\\IRIS_netcdf_v2.nc', reach_vars=[f'avg_{sl}_slope' for sl in iris_slope_opts_test], clean_run=False)
    df_test = df_test.rename(columns={f'avg_{sl}_slope': sl for sl in iris_slope_opts_test})
    print(df_test)