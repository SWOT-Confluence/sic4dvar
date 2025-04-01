"""
Created on September 18th 2023 at 19:00
by @Isadora Silva

Last modified on February 23rd 2024 at 18:00
by @Isadora Silva

@authors: Isadora Silva
"""

import pathlib
from datetime import datetime
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults
from sic4dvar_low_cost.sic4dvar_functions.helpers.helpers_arrays import array_as_row_vector, array_as_col_vector, \
    datetime_array_set_to_freq_and_filter
from sic4dvar_low_cost.sic4dvar_functions.interpolation_linear_piecewise import piecewise_linear_interpolation
from sic4dvar_low_cost.sic4dvar_functions.io.reader_sword import get_vars_from_sword_file


def get_array_dict_from_csv_files(
        reach_ids: Tuple[str, ...],
        node_z_file_pattern: str | pathlib.PurePath | None = None,
        node_z_u_file_pattern: str | pathlib.PurePath | None = None,
        node_w_file_pattern: str | pathlib.PurePath | None = None,
        node_w_u_file_pattern: str | pathlib.PurePath | None = None,
        reach_data_file_pattern: str | pathlib.PurePath | None = None,  # if None, does not run algo 5 (swot q)
        dist_col: str | None = None,
        node_id_col: str | None = None,
        reach_time_col: str | None = None,
        reach_s_col: str | None = None,
        reach_w_col: str | None = None,
        reach_da_col: str | None = None,
        miss_node_in_sword: str | None = None,
        no_data_value: float | str | int | None = None,
        dist_dif_obs: int | None = None,
        sword_file_path: str | pathlib.PurePath | None = None,
        x_ref: Literal["node_length", "dist_out"] = "node_length",
        add_facc: bool = False,
        ref_datetime: datetime = SIC4DVarLowCostDefaults().def_ref_datetime,
        freq_datetime: str = SIC4DVarLowCostDefaults().def_freq_datetime,
        dup_datetime: Literal["drop", "raise"] = "raise",
        start_datetime: datetime | float | int | None = None,
        end_datetime: datetime | float | int | None = None,
        clean_run: bool = False,
        debug_mode: bool = False,
):
    
    if clean_run:
        debug_mode = False
    if debug_mode:
        clean_run = False

    # one of them must be defined otherwise we won't know the nodes or the distances
    if all([i_ is None for i_ in [node_z_file_pattern, node_w_file_pattern]]):
        raise TypeError(
            "at least one of node_z_file_pattern, node_w_file_pattern must be specified")

    reach_ids = [int(i) for i in reach_ids]

    if len(reach_ids) == 0:
        raise TypeError("must specify at lest one reach_id")

    if dist_dif_obs is None:
        dist_dif_obs = SIC4DVarLowCostDefaults().def_dist_dif_obs

    # check for reference cols
    if (dist_col is None) and (node_id_col is None):
        raise TypeError("must define one of node_id_col or dist_col")

    # check for sword_file
    if sword_file_path is not None:
        if node_id_col is None:
            raise TypeError("if sword_file_path is defined, node_id_col must be defined")
        sword_file_path = pathlib.Path(sword_file_path)
        if not sword_file_path.exists():
            raise FileNotFoundError(f"sword_file_path {sword_file_path} does not exist")
    else:
        add_facc = False
    #     if dist_col is None:
    #         raise TypeError("if sword_file_path is not defined, dist_col must be defined")

    # list and paths of variables
    df_node_list_var, df_reach_list_var, path_dict = [], [], {}
    if node_z_file_pattern is not None:
        path_dict["node_z"] = str(node_z_file_pattern)
        df_node_list_var.append("node_z")
    if node_w_file_pattern is not None:
        path_dict["node_w"] = str(node_w_file_pattern)
        df_node_list_var.append("node_w")
    if node_z_u_file_pattern is not None:
        path_dict["node_z_u"] = str(node_z_u_file_pattern)
        df_node_list_var.append("node_z_u")
    if node_w_u_file_pattern is not None:
        path_dict["node_w_u"] = str(node_w_u_file_pattern)
        df_node_list_var.append("node_w_u")
    if (reach_data_file_pattern is not None) and np.any(
            [i_ is not None for i_ in [reach_s_col, reach_w_col, reach_da_col]]):
        if reach_time_col is None:
            raise TypeError("to extract the time form reach csv, must specify reach_time_col")
        path_dict["reach_data"] = str(reach_data_file_pattern)
        if reach_s_col is not None:
            df_reach_list_var.append("reach_s")
        if np.all(
                [i_ is not None for i_ in [reach_s_col, reach_w_col, reach_da_col]]):
            df_reach_list_var.append("reach_w")
            df_reach_list_var.append("reach_da")

    # dict to store the array data
    df_dict_arrays = {k_: [] for k_ in df_node_list_var + df_reach_list_var}
    df_dict_arrays["t"] = []
    df_dict_arrays["x"] = []
    df_dict_arrays["node_id"] = []

    # loop in reaches to fill the dict
    for n_reach, reach_id in enumerate(reach_ids):

        # get node variables data
        for n_var, node_var in enumerate(df_node_list_var):

            # get path
            f_pat = path_dict[node_var]
            # format path and open as csv
            df_var = pd.read_csv(f_pat.format(reach_id))
            # set datatype to np.float64
            df_var = df_var.astype(np.float64)
            # mask no data values
            if no_data_value is not None:
                df_var[df_var == no_data_value] = pd.NA

            # get dist and node_id arrays
            dist_array = np.empty(0) if dist_col is None else array_as_col_vector(
                np.round(df_var[dist_col].to_numpy(dtype=np.float64), 0).astype(np.int64))
            node_id_array = np.empty(0) if node_id_col is None else array_as_col_vector(
                df_var[node_id_col].to_numpy(dtype=np.int64))

            # remove dist and node_id from cols
            df_var = df_var[[c_ for c_ in df_var.columns if c_ not in [node_id_col, dist_col]]]

            # columns: change type
            datetime_sec_df_array = np.round(df_var.columns.to_numpy(dtype=np.float64), 0).astype(np.int64)
            df_var.columns = datetime_sec_df_array
            # columns: sort
            datetime_sec_df_array = np.sort(datetime_sec_df_array)
            df_var = df_var[datetime_sec_df_array]

            if any([start_datetime is not None, end_datetime is not None, freq_datetime]):
                # match dts to the specified frequency and get boolean array for filtering between start and end
                cols_mask_bool, _, datetime_sec_df_array = datetime_array_set_to_freq_and_filter(
                    data_dt=datetime_sec_df_array, ref_datetime=ref_datetime, freq_datetime=freq_datetime,
                    duplicates=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime)

                # Filter the data to include only the values between start and end dates
                df_var = df_var[df_var.columns[cols_mask_bool]]

                # change name of columns
                df_var.columns = datetime_sec_df_array

            # get time array
            time_array = array_as_row_vector(datetime_sec_df_array)

            # if first node variable, add time, distance and/or node_id to the dict
            if n_var == 0:
                df_dict_arrays["t"].append(time_array)
                df_dict_arrays["x"].append(dist_array)
                df_dict_arrays["node_id"].append(node_id_array)

            # if not, check if shape and size matches the ones defined by previous node variable
            else:
                error_msg = f"reach {reach_id} variable {node_var},"
                if df_dict_arrays["t"][n_reach].shape != time_array.shape:
                    raise IndexError(f"{error_msg} time array shape mismatch")
                if not np.allclose(df_dict_arrays["t"][n_reach], time_array, rtol=0, atol=0):
                    raise ValueError(f"{error_msg} time array value mismatch")
                if df_dict_arrays["x"][n_reach].shape != dist_array.shape:
                    raise IndexError(f"{error_msg} dist array shape mismatch")
                if not np.allclose(df_dict_arrays["x"][n_reach], dist_array, rtol=0., atol=dist_dif_obs):
                    raise ValueError(f"{error_msg} dist array value mismatch")
                if df_dict_arrays["node_id"][n_reach].shape != node_id_array.shape:
                    raise IndexError(f"{error_msg} node id array shape mismatch")
                if not np.all(np.equal(df_dict_arrays["node_id"][n_reach], node_id_array)):
                    raise ValueError(f"{error_msg} node id array value mismatch")

            # add data to dict
            df_dict_arrays[node_var].append(df_var.to_numpy(dtype=np.float32))

        if "reach_data" in path_dict.keys():

            # get path
            f_pat = path_dict["reach_data"]
            # format path and open as csv
            df_reach = pd.read_csv(f_pat.format(reach_id))
            # mask no data values
            if no_data_value is not None:
                df_reach.loc[df_reach == no_data_value] = pd.NA
            # set datatype to np.float64
            df_reach = df_reach.astype(np.float64)

            # sort by datetime
            df_reach.sort_values(reach_time_col, inplace=True)

            # get time array
            datetime_sec_df_array = np.round(
                df_reach[reach_time_col].to_numpy(dtype=np.float64).flatten(), 0).astype(np.int64)

            if any([start_datetime is not None, end_datetime is not None, freq_datetime]):
                # match dts to the specified frequency and get boolean array for filtering between start and end
                rows_mask_bool, _, datetime_sec_df_array = datetime_array_set_to_freq_and_filter(
                    data_dt=datetime_sec_df_array, ref_datetime=ref_datetime, freq_datetime=freq_datetime,
                    duplicates=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime)

                # Filter the data to include only the values between start and end dates
                df_reach = df_reach.loc[df_reach.index[rows_mask_bool]]

            # change to row vector
            time_array = array_as_row_vector(datetime_sec_df_array)

            # remove time array from columns
            df_reach = df_reach[[c_ for c_ in df_reach.columns if c_ != reach_time_col]]

            # time check
            error_msg = f"reach {reach_id} reach variables,"
            if df_dict_arrays["t"][n_reach].shape != time_array.shape:
                raise IndexError(f"{error_msg} time array shape mismatch")
            if not np.allclose(df_dict_arrays["t"][n_reach], time_array, rtol=0, atol=0):
                raise ValueError(f"{error_msg} time array value mismatch")

            # add columns to data dict
            for reach_var in df_reach_list_var:
                df_dict_arrays[reach_var].append(
                    array_as_row_vector(df_reach[reach_var].to_numpy(dtype=np.float32)))

    # add reach ids to df obs dict
    df_dict_arrays["reach_id"] = [np.full((1, 1), fill_value=i_, dtype=np.int64) for i_ in reach_ids]

    # if sword path was defined, give preference to it
    if (sword_file_path is not None) and (node_id_col is not None):

        # get node info from SWORD
        sword_dict = get_vars_from_sword_file(
            reach_ids=reach_ids, sword_file_path=sword_file_path,
            node_vars=("node_length", "dist_out", "node_id"),
            reach_vars=("facc",) if add_facc else (),
            x_ref=x_ref, clean_run=clean_run)

        # force node_id type to int64
        sword_dict["nodes"]["node_id"] = sword_dict["nodes"]["node_id"].astype(np.int64)

        # add the distance (max_dist - dist_out) from the SWORD nodes to the df observation dict
        for reach_n in range(len(df_dict_arrays["reach_id"])):

            # create empy array to store distances for the nodes of this reach
            df_dict_arrays["x"][reach_n] = np.full(
                df_dict_arrays["node_id"][reach_n].shape, fill_value=np.nan, dtype=np.float64, )

            # loop to search per node
            for node_idx_in_df, node_n in enumerate(df_dict_arrays["node_id"][reach_n][:, 0]):

                # get the indexes of the node in SWORD
                try:
                    node_idx_in_sword = np.nonzero(sword_dict["nodes"]["node_id"] == node_n)[0][0]

                except IndexError:
                    error_msg = f"could not find node {node_n} in SWORD for reach " + \
                                str(df_dict_arrays["reach_id"][reach_n][0, 0])
                    # raise KeyError if node was not found in SWORD
                    if (miss_node_in_sword is None) or any([i_ in miss_node_in_sword for i_ in ["raise", "error"]]):
                        raise KeyError(error_msg)
                    if not clean_run:
                        print(error_msg)

                else:
                    # assign distance to node
                    df_dict_arrays["x"][reach_n][node_idx_in_df] = sword_dict["nodes"]["dist_out"][
                        node_idx_in_sword]

            # linear interpolate distance for missing nodes in SWORD
            if np.any(np.isnan(df_dict_arrays["x"][reach_n])):
                df_dict_arrays["x"][reach_n] = piecewise_linear_interpolation(
                    values_in_array=df_dict_arrays["x"][reach_n],
                    base_in_array=np.array(range(0, len(df_dict_arrays["x"][reach_n]))),
                    limits="linear", check_nan=False)

        # correct distance to refer to most upstream point
        for reach_n in range(len(df_dict_arrays["reach_id"])):
            df_dict_arrays["x"][reach_n] = np.nanmax(sword_dict["nodes"]["dist_out"]) - df_dict_arrays["x"][reach_n]
    else:
        sword_dict = dict()

    # if node_id was not defined, remove it from dict
    if df_dict_arrays["node_id"][0].size == 0:
        df_dict_arrays.pop("node_id")

    # if distance was not defined, remove it from dict
    if df_dict_arrays["x"][0].size == 0:
        df_dict_arrays.pop("x")

    if add_facc:
        df_dict_arrays["reach_facc"] = [
            np.full((1, 1), fill_value=i_, dtype=np.float32) for i_ in sword_dict["reaches"]["facc"]]

    return df_dict_arrays


if __name__ == "__main__":
    from pathlib import Path

    base_path = Path(r"C:\Users\isadora.rezende\PhD")
    swot_sim_path = base_path / "Discharge_paper" / "Po" / "SWOT_sim" / "csvs" / "observations"

    test_output = get_array_dict_from_csv_files(
        reach_ids=(
            # 21406100011,  # !
            # 21406100021,  # !
            21406100071,  # !
            21406100031,
            21406100041,
            # 21406100051,
            # 21406100061,
            # 21406100081,
            # 21406100101,
            # 21406100111,
        ),
        node_z_file_pattern=swot_sim_path / "{}_z_SWOT_sim.csv",
        node_w_file_pattern=swot_sim_path / "{}_w_SWOT_sim.csv",
        node_id_col="node_id",
        miss_node_in_sword="linear",
        no_data_value=-9999.,
        sword_file_path=base_path / "Datasets" / "SWORD" / "v15" / "netcdf" / "eu_sword_v15.nc",
        start_datetime=datetime(2008, 5, 1),
        end_datetime=datetime(2009, 6, 30),
    )
    for k, v in test_output.items():
        print(k, [v_i.shape for v_i in v])
