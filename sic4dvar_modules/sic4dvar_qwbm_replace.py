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
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import scipy
import logging
import numpy as np
import sic4dvar_params as params
import sic4dvar_functions.sic4dvar_calculations as calc
from sic4dvar_functions.sic4dvar_helper_functions import get_weighted_q_data
from lib.lib_dates import get_swot_dates

def replace_prior(sic4dvar_dict):
    flag_qwbm = True
    if (calc.check_na(sic4dvar_dict['input_data']['reach_qwbm']) or sic4dvar_dict['input_data']['reach_qwbm'] < 1.0) and (not sic4dvar_dict['param_dict']['q_prior_from_stations']) and params.activate_facc:
        logging.warning('qwbm value invalid/too small.')
        logging.info('INFO: trying to replace qwbm value with facc * specific discharge.')
        flag_qwbm = False
        if not calc.check_na(sic4dvar_dict['input_data']['facc']) or sic4dvar_dict['input_data']['facc'] > 1.0:
            sic4dvar_dict['input_data']['reach_qwbm'] = sic4dvar_dict['input_data']['facc'] * 10 / 1000
            logging.info('INFO: replaced qwbm value with facc * specific discharge.')
            logging.info(f'facc * SD value: {sic4dvar_dict['input_data']['reach_qwbm']}')
            flag_qwbm = True
    else:
        logging.info('QWBM suitable to use.')
    if sic4dvar_dict['param_dict']['override_q_prior']:
        sic4dvar_dict['input_data']['reach_qwbm'] = np.ma.masked_values(float(sic4dvar_dict['param_dict']['q_prior_value']), value=-9999.0)
        flag_qwbm = True
    if sic4dvar_dict['data_is_useable'] and sic4dvar_dict['param_dict']['q_monthly_mean']:
        times = sic4dvar_dict['input_data']['reach_t']
        count = 0
        for i in range(0, len(sic4dvar_dict['input_data']['q_monthly_mean'])):
            if calc.check_na(sic4dvar_dict['input_data']['q_monthly_mean'][i]):
                count = count + 1
        if count == len(sic4dvar_dict['input_data']['q_monthly_mean']):
            logging.info('INFO: tried to replaced qwbm value with q monthly mean computation')
            logging.info('But all q monthly mean data was empty !')
        else:
            dates = get_swot_dates(sic4dvar_dict['filtered_data']['node_t'])
            masked_data = np.ma.masked_values(np.array([get_weighted_q_data(dates, sic4dvar_dict['input_data']['q_monthly_mean'])]), value=-9999.0)
            sic4dvar_dict['input_data']['reach_qwbm'] = masked_data
            logging.info('INFO: replaced qwbm value with monthly mean computation')
            logging.info(f'qwbm value: {sic4dvar_dict['input_data']['reach_qwbm']}')
    return (sic4dvar_dict, flag_qwbm)