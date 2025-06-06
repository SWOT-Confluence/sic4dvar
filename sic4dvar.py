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
import argparse, configparser
import pathlib
import datetime
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import sys
import sic4dvar_params as params
from sic4dvar_functions.sic4dvar_runs import *
from lib.lib_log import append_to_principal_log, call_error_message
from lib.lib_config import read_config
from sic4dvar_functions.sic4dvar_helper_functions import get_run_type

def init_parameters(args):
    param_dict = {}
    if args['json_path']:
        param_dict['json_path'] = pathlib.Path(args['json_path']).absolute()
    if args['swot_dir']:
        param_dict['swot_dir'] = pathlib.Path(args['swot_dir']).absolute()
    if args['output_dir']:
        param_dict['output_dir'] = pathlib.Path(args['output_dir']).absolute()
    if args['log_dir']:
        param_dict['log_dir'] = pathlib.Path(args['log_dir']).absolute()
    if args['run_type']:
        if args['run_type'] == 'seq':
            param_dict['set_run'] = False
            param_dict['seq_run'] = True
        elif args['run_type'] == 'set':
            param_dict['set_run'] = True
            param_dict['seq_run'] = False
        else:
            print('WARNING : run_type %s not known' % args['run_type'])
    else:
        param_dict['set_run'] = params.set_run
        param_dict['seq_run'] = params.seq_run
    if args.AWS:
        param_dict['sos_dir'] = param_dict['input_dir'].joinpath('sos')
        param_dict['swot_dir'] = param_dict['input_dir'].joinpath('swot')
        param_dict['sword_dir'] = param_dict['input_dir'].joinpath('sword')
    elif args.sos_dir:
        param_dict['sos_dir'] = args.sos_dir
    else:
        param_dict['sos_dir'] = pathlib.Path(args['sos_dir']).joinpath(param_dict['constraint'])
    if args['swot_dir']:
        param_dict['swot_dir'] = pathlib.Path(args['swot_dir'])
    if args['sword_dir']:
        param_dict['sword_dir'] = pathlib.Path(args['sword_dir'])
    if not param_dict['output_dir'].exists():
        param_dict['output_dir'].mkdir(parents=True, exist_ok=True)
    if not param_dict['log_dir'].exists():
        param_dict['log_dir'].mkdir(parents=True, exist_ok=True)
    for param in args.keys():
        if param.startswith('node_level_') or param.startswith('reach_level_'):
            param_dict[param] = args[param]
    return param_dict

def get_explicit_args(parser):
    explicit = set()
    for action in parser._actions:
        if any((opt in sys.argv for opt in action.option_strings)):
            explicit.add(action.dest)
    return explicit

def main():
    parser = argparse.ArgumentParser(description='Sic4dvar algorithm.')
    parser.add_argument('--config_file_path', help='json file name to run.', default=pathlib.Path(__file__).parent / 'sic4dvar_param.ini', type=pathlib.Path)
    parser.add_argument('-r', '--json_path', help='json file name to run.', type=pathlib.Path)
    parser.add_argument('--output_dir', help='Output directory', type=pathlib.Path)
    parser.add_argument('--log_dir', help='Log directory', type=pathlib.Path)
    parser.add_argument('--swot_dir', help='Input swot directory', type=pathlib.Path)
    parser.add_argument('--sos_dir', help='SWORD Of Science dir', type=pathlib.Path)
    parser.add_argument('--sword_dir', help='SWORD dir', type=pathlib.Path)
    parser.add_argument('--input_dir', help='Input directory with swot, sos, and sword directories inside', type=pathlib.Path)
    parser.add_argument('-p', '--flag_parallel', action='store_true', help='Parallel processing')
    parser.add_argument('-v', '--verbose', help='Be verbose, log level debug', default=logging.INFO, action='store_const', dest='loglevel', const=logging.DEBUG)
    parser.add_argument('-i', '--index', help='json file index for reach/set', type=int)
    args = parser.parse_args()
    if params.replace_config:
        args.config_file_path = pathlib.Path(params.config_file_path)
    param_dict = read_config(args.config_file_path)
    explicit_args = get_explicit_args(parser)
    args_dict = vars(args)
    for key in explicit_args:
        param_dict[key] = args_dict[key]
    param_dict['log_path'] = param_dict['log_dir'].joinpath('sic4dvar.log')
    append_to_principal_log(param_dict, 'Running SIC4DVAR low cost')
    if param_dict["aws"] and "json_path" not in param_dict:
        param_dict["json_path"] = pathlib.Path(param_dict["input_dir"],"reaches.json")
        param_dict["sos_dir"] = param_dict["input_dir"].joinpath('sos')
        param_dict["swot_dir"] = param_dict["input_dir"].joinpath('swot')
        param_dict["sword_dir"] = param_dict["input_dir"].joinpath('sword')
    elif param_dict["aws"]:
        param_dict["json_path"] = pathlib.Path(param_dict["input_dir"], param_dict["json_path"])
        param_dict["sos_dir"] = param_dict["input_dir"].joinpath('sos')
        param_dict["swot_dir"] = param_dict["input_dir"].joinpath('swot')
        param_dict["sword_dir"] = param_dict["input_dir"].joinpath('sword')
    param_dict['run_type'] = get_run_type(param_dict['json_path'])
    if param_dict['run_type'] == 'seq':
        sic4dvar_seq_run(param_dict)
    elif param_dict['run_type'] == 'set':
        sic4dvar_set_run(param_dict)
    else:
        append_to_principal_log(param_dict, f'Unknown run type {param_dict['run_type']}')
        pass
if __name__ == '__main__':
    main()
