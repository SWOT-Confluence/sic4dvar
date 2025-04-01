"""
Created on September 18th 2023 at 19:00
by @Isadora Silva

Last modified on February 23rd 2024 at 18:00
by @Isadora Silva

@authors: Isadora Silva
"""

import copy
import pathlib
from datetime import datetime
from typing import Tuple, Dict, Literal

import netCDF4 as nc4
import numpy as np
import pandas as pd

from sic4dvar_classes.sic4dvar_0_defaults import SIC4DVarLowCostDefaults
from sic4dvar_functions.helpers.helpers_arrays import masked_array_to_nan_array, check_shape, \
    array_as_row_vector, array_as_col_vector, datetime_array_set_to_freq_and_filter


def get_vars_from_swot_nc(
        reach_ids: Tuple[str | int, ...],
        swot_file_pattern: str | pathlib.PurePath,
        node_vars: Tuple[str, ...] = (),
        reach_vars: Tuple[str, ...] = (),
        val_swot_q_flag: Tuple[int, ...] = SIC4DVarLowCostDefaults().def_val_swot_q_flag,
        freq_datetime: str = SIC4DVarLowCostDefaults().def_freq_datetime,
        dup_datetime: Literal["drop", "raise"] = "raise",
        start_datetime: datetime | float | int | None = None,
        end_datetime: datetime | float | int | None = None,
        clean_run: bool = False,
        debug_mode: bool = False,
) -> Tuple[Dict[Dict, Dict], datetime]:
    """
    Read SWOT-like observations of the specified node and reach variables of the specified reach_ids.
    1D arrays are forced to either row or column vectors.

    Parameters
    ----------
    reach_ids : Tuple[str | int, ...]
        The reach ids to be read from the SWOT file
    swot_file_pattern :
        The pattern to the SWORD netCDF files, to be formatted using the reach_ids.
    node_vars : Tuple[str, ...]
        The node variables to be loaded
    reach_vars : Tuple[str, ...]
        The reach variables to be loaded
    val_swot_q_flag: Tuple[int, ...]
        The quality flags for valid data (0: nominal, 1: suspect, 2: degraded quality, and 3: bad).
    freq_datetime : datetime
        The time of observations is approximated every freq when loading the data.
        Multiplier and str "D", "h", "min", "s". E.g.: "3h", "D". Set as "" to skip this.
        https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    dup_datetime: Literal["drop", "raise"]
        What to do when the data is assigned to the same datetime. Either "drop", "raise": "drop" drops all
         occurrences except the first one; "raise" raises IndexError.
    start_datetime: datetime | float | int | None
        start datetime for filtering the data (>=). Either as datetime or as seconds from reference (float or int).
    end_datetime: datetime | float | int | None
        end datetime for filtering the data (<=). Either as datetime or as seconds from reference (float or int).
    clean_run : bool
        Whether to print statements while running this function
    debug_mode : bool
        Whether to print debug statements while running this function

    Returns
    -------
    Tuple[Dict[Dict, Dict], datetime]:
        (i) Dictionary with the unmasked arrays. Structure {"nodes": {node_dict}, "reaches": {reach_dict}}.
         Each inner dict contains a tuple whose elements are the unmasked arrays for each reach_id.
        (ii) Reference datetime
    """
    if clean_run:
        debug_mode = False
    if debug_mode:
        clean_run = False

    msg = "loading SWOT observations"
    if not clean_run:
        print(f"\n{msg}")

    if (len(node_vars) == 0) and (len(reach_vars) == 0):
        raise TypeError("must specify either node_vars or reach_vars")

    node_vars1 = list(copy.deepcopy(node_vars))
    reach_vars1 = list(copy.deepcopy(reach_vars))
    if len(node_vars1) > 0:
        if "node_q" not in node_vars1:
            node_vars1.append("node_q")
    if len(reach_vars1) > 0:
        if "reach_q" not in reach_vars1:
            reach_vars1.append("reach_q")

    reach_ids = [int(i) for i in reach_ids]

    if len(reach_ids) == 0:
        raise TypeError("must specify at lest one reach_id")

    # check for swot files per reach
    reach_path_dict = {}
    swot_file_pattern = str(swot_file_pattern)
    for reach_id in reach_ids:
        reach_file_path = pathlib.Path(swot_file_pattern.format(reach_id))
        if not reach_file_path.exists():
            raise FileNotFoundError(f"swot file for reach {reach_id} {reach_file_path} does not exist")
        reach_path_dict[reach_id] = reach_file_path

    # initialize array to hold variables
    swot_dict_arrays = {"node": {k_: [] for k_ in node_vars1}, "reach": {k_: [] for k_ in reach_vars1}}
    swot_list_var = list(node_vars1) + list(reach_vars1)

    ref_datetime = SIC4DVarLowCostDefaults().def_ref_datetime

    # loop in reach_ids
    for r_i in reach_ids:

        reach_msg = f"loading {swot_list_var} data from SWOT for reach {r_i}"
        if debug_mode:
            print(reach_msg)

        # open netcdf file for reach
        with nc4.Dataset(reach_path_dict[r_i]) as swot_ds:

            # get reference datetime from SWOT metadata
            ref_datetime = datetime.fromisoformat(swot_ds["reach"]["time"].units.split("since ")[1])

            # get all times for that reach
            reach_times = masked_array_to_nan_array(swot_ds["reach"]["time"][:])

            if any([start_datetime is not None, end_datetime is not None, freq_datetime]):
                # Match the datetimes to the specified frequency, get boolean array for filtering between start and end
                time_mask_bool, time_as_datetime, time_as_sec_array = datetime_array_set_to_freq_and_filter(
                    data_dt=reach_times, ref_datetime=ref_datetime, freq_datetime=freq_datetime,
                    duplicates=dup_datetime, start_datetime=start_datetime, end_datetime=end_datetime)

                # get indexes to filter the data by start and end date
                time_ids = np.nonzero(time_mask_bool)[0]
            else:
                # get all indexes
                time_ids = np.array(range(reach_times.size), dtype=np.int32)

                # round and set as integer
                time_as_sec_array = np.array(np.round(reach_times, 0), dtype=np.int64)

            # get total number of nodes and total number of times
            total_n_nodes = swot_ds["node"]["node_id"][:].size
            total_n_times = swot_ds["reach"]["time"][:].size

            for node_var in swot_dict_arrays["node"].keys():
                # read data from SWOT file
                swot_var = swot_ds["node"][node_var]
                swot_var_array = masked_array_to_nan_array(swot_var[:])

                # shape check
                if node_var not in ["reach_id", "node_id"]:
                    # all node variables should be 2D arrays
                    swot_var_array = check_shape(
                        swot_var_array, expected_shape=(total_n_nodes, total_n_times), force_shape=False)
                    # filter by valid times
                    swot_var_array = swot_var_array[:, time_ids]
                elif node_var == "reach_id":
                    swot_var_array.shape = (1, 1)
                else:
                    swot_var_array = array_as_col_vector(swot_var_array)

                # append to dict
                swot_dict_arrays["node"][node_var].append(swot_var_array)

            for reach_var in swot_dict_arrays["reach"].keys():
                # read data from SWOT file
                swot_var = swot_ds["reach"][reach_var]
                swot_var_array = masked_array_to_nan_array(swot_var[:])

                # shape check
                if reach_var != "reach_id":
                    # all reach variables should be row cols with time
                    swot_var_array = array_as_row_vector(swot_var_array)
                    # filter by valid times
                    swot_var_array = swot_var_array[:, time_ids]
                else:
                    swot_var_array.shape = (1, 1)

                if reach_var == "time":
                    swot_var_array = time_as_sec_array

                # append to dict
                swot_dict_arrays["reach"][reach_var].append(swot_var_array)

            if debug_mode:
                print(f"total number of nodes  {total_n_nodes}, total number of time instances: {total_n_times}")

        if debug_mode:
            print(msg, "done")

    # mask node arrays according to quality of the measurements
    try:
        node_q_list = swot_dict_arrays["node"]["node_q"]
    except KeyError:
        pass
    else:
        for k_ in node_vars:
            if k_ in ["node_q", "node_id"]:
                continue
            for v_n, v_array in enumerate(swot_dict_arrays["node"][k_]):
                swot_dict_arrays["node"][k_][v_n] = np.where(
                    np.isin(node_q_list[v_n], val_swot_q_flag), v_array, np.nan)

    # mask reach arrays according to quality of the measurements
    try:
        reach_q_list = swot_dict_arrays["reach"]["reach_q"]
    except KeyError:
        pass
    else:
        for k_ in reach_vars:
            if k_ in ["reach_q", "reach_id"]:
                continue
            for v_n, v_array in enumerate(swot_dict_arrays["reach"][k_]):
                swot_dict_arrays["reach"][k_][v_n] = np.where(
                    np.isin(reach_q_list[v_n], val_swot_q_flag), v_array, np.nan)

    # remove quality flags if not required by user
    if ("node_q" in swot_dict_arrays["node"].keys()) and ("node_q" not in node_vars):
        swot_dict_arrays["node"].pop("node_q")
    if ("reach_q" in swot_dict_arrays["reach"].keys()) and ("reach_q" not in reach_vars):
        swot_dict_arrays["reach"].pop("reach_q")

    # list (mutable) to tuple (immutable)
    swot_dict_arrays = {
        "node": {k_: tuple(v_) for k_, v_ in swot_dict_arrays["node"].items()},
        "reach": {k_: tuple(v_) for k_, v_ in swot_dict_arrays["reach"].items()},
    }

    if not clean_run:
        print(msg, "done")

    return swot_dict_arrays, ref_datetime


def get_array_dict_from_swot_nc(
        reach_ids: Tuple[str | int, ...],
        swot_file_pattern: str | pathlib.PurePath,
        use_node_z: bool,
        use_node_w: bool,
        use_reach_slope: bool,
        compute_swot_q: bool,
        use_uncertainty: bool,
        val_swot_q_flag: Tuple[int, ...] = SIC4DVarLowCostDefaults().def_val_swot_q_flag,
        freq_datetime: str = SIC4DVarLowCostDefaults().def_freq_datetime,
        dup_datetime: Literal["drop", "raise"] = "raise",
        start_datetime: datetime | float | int | None = None,
        end_datetime: datetime | float | int | None = None,
        clean_run: bool = False,
        debug_mode: bool = False,
) -> Tuple[Dict, datetime]:
    """
    Read SWOT-like observations of the specified node and reach variables of the specified reach_ids.
    1D arrays are forced to either row or column vectors.

    Parameters
    ----------
    reach_ids : Tuple[str | int, ...]
        The reach ids to be read from the SWOT file
    swot_file_pattern : str | pathlib.PurePath
        The pattern to the SWORD netCDF files, to be formatted using the reach_ids.
    use_node_z : bool
        Whether to load the water surface elevation data.
    use_node_w : bool
        Whether to load the width data
    use_reach_slope : bool
        Whether to load the reach slope data
    compute_swot_q : bool
        Whether to load the reach delta area and width data
    use_uncertainty : bool
        Whether to load the uncertainty data.
    val_swot_q_flag: Tuple[int, ...]
        The quality flags for valid data (0: nominal, 1: suspect, 2: degraded quality, and 3: bad).
    freq_datetime : datetime
        The time of observations is approximated every freq when loading the data.
        Multiplier and str "D", "h", "min", "s". E.g.: "3h", "D". Set as "" to skip this.
        https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    dup_datetime: Literal["drop", "raise"]
        What to do when the data is assigned to the same datetime. Either "drop", "raise": "drop" drops all
         occurrences except the first one; "raise" raises IndexError.
    start_datetime: datetime | float | int | None
        start datetime for filtering the data (>=). Either as datetime or as seconds from reference (float or int).
    end_datetime: datetime | float | int | None
        end datetime for filtering the data (<=). Either as datetime or as seconds from reference (float or int).
    clean_run : bool
        Whether to print statements while running this function
    debug_mode : bool
        Whether to print debug statements while running this function

    Returns
    -------
    Dict
        Dict with arrays
    """
    if clean_run:
        debug_mode = False
    if debug_mode:
        clean_run = False

    # look up table for swot variables
    swot_lut = pd.DataFrame(
        index=["t", "node_z", "node_w", "node_z_u", "node_w_u", "node_id", "reach_s", "reach_da", "reach_w"],
        columns=["reach", "node"],
    )
    swot_lut.loc["t", "reach"] = "time"
    swot_lut.loc["node_z", "node"] = "wse"
    swot_lut.loc["node_w", "node"] = "width"
    swot_lut.loc["node_z_u", "node"] = "wse_u"
    swot_lut.loc["node_w_u", "node"] = "width_u"
    swot_lut.loc["node_id", "node"] = "node_id"
    swot_lut.loc["reach_s", "reach"] = "slope2"
    swot_lut.loc["reach_da", "reach"] = "d_x_area"
    swot_lut.loc["reach_w", "reach"] = "width"

    # define reach and node variables
    swot_node_list_var, swot_reach_list_var = ["node_id", ], ["t"]
    if use_node_z:
        swot_node_list_var.append("node_z")
    if use_node_w:
        swot_node_list_var.append("node_w")
    if use_uncertainty:
        if use_node_z:
            swot_node_list_var.append("node_z_u")
        if use_node_w:
            swot_node_list_var.append("node_w_u")
    if use_reach_slope or compute_swot_q:
        swot_reach_list_var.append("reach_s")
    if compute_swot_q:
        swot_reach_list_var.extend(["reach_da", "reach_w"])

    # create dict to store the arrays with the observations
    swot_dict_arrays = {k_: [] for k_ in swot_node_list_var + swot_reach_list_var}

    # get arrays from SWOT observation files
    swot_dict, ref_datetime = get_vars_from_swot_nc(
        reach_ids=reach_ids,
        swot_file_pattern=swot_file_pattern,
        node_vars=swot_lut.loc[swot_node_list_var][["node"]].to_numpy().flatten(),
        reach_vars=swot_lut.loc[swot_reach_list_var][["reach"]].to_numpy().flatten(),
        val_swot_q_flag=val_swot_q_flag,
        freq_datetime=freq_datetime,
        dup_datetime=dup_datetime,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        clean_run=clean_run,
        debug_mode=debug_mode,
    )

    for k_i, tup_arr_i in swot_dict["node"].items():
        swot_dict_arrays[swot_lut.loc[swot_lut["node"] == k_i].index[0]] = list(tup_arr_i)
    for k_i, tup_arr_i in swot_dict["reach"].items():
        swot_dict_arrays[swot_lut.loc[swot_lut["reach"] == k_i].index[0]] = list(tup_arr_i)

    # round time instances (no need if lower than second precision)
    for t_n, t_arr in enumerate(swot_dict_arrays["t"]):
        swot_dict_arrays["t"][t_n] = np.round(t_arr).astype(np.int64)  # int 64 because we expect big numbers

    # convert node id to int
    for n_n, n_arr in enumerate(swot_dict_arrays["node_id"]):
        swot_dict_arrays["node_id"][n_n] = n_arr.astype(np.int64)  # int 64 because we expect big numbers

    return swot_dict_arrays, ref_datetime


if __name__ == "__main__":
    base_path = pathlib.Path(r"C:\Users\isadora.rezende\PhD")

    test_output, _ = get_array_dict_from_swot_nc(
        reach_ids=(
            74265000081,
            # 74265000091,
            # 74265000101,
            # 74265000111,
            # 74265000121,
        ),
        swot_file_pattern=base_path / "Datasets" / "PEPSI" / "Ohio" / "{}_SWOT.nc",
        use_node_z=True,
        use_node_w=False,
        use_reach_slope=False,
        compute_swot_q=False,
        use_uncertainty=False,
        freq_datetime="3D",
        start_datetime=datetime(2010, 10, 1),
        clean_run=False,
        debug_mode=False,
    )
    for k, v in test_output.items():
        print(k, [v_i.shape for v_i in v])
    # pprint(test_output[0])
    # print(np.concatenate(test_output[0]["t"], axis=1).shape)
    # print(np.concatenate(test_output[0]["node_id"], axis=0).shape)
