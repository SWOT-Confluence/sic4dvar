#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Callum TYLER, callum.tyler@inrae.fr 
            All functions not mentioned below.
          Hind OUBANAS, hind.oubanas@inrae.fr
            All functions not mentioned below.
          Nikki TEBALDI, ntebaldi@umass.edu
            get_input_data(),
            write_output(),
            get_reachids(),
            get_reach_dataset(),
            part of main()
        Dylan QUITTARD, dylan.quittard@inrae.fr
            All functions not mentioned above.
        Cécile Cazals, cecile.cazals@cs-soprasteria.com
            logger
            
        


Description:
    Version of algo31 and algo5 combined into algo315 (sic4dvar) which can 
    run on Confluence.
    NOTE this code is for use only for computational resource testing. 
    Numerous modification need to be made to run accurately on confluence.
    See the TODO list at the top of this file.
"""

#>----------------------------------------------------------------------------<

import argparse, configparser
import pathlib
import datetime
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)

## Local libraries
import sic4dvar_params as params
from sic4dvar_functions.sic4dvar_runs import *
from lib.lib_log import set_logger
from lib.lib_config import read_config
from sic4dvar_functions.sic4dvar_helper_functions import get_reach_dataset_all

def init_parameters(args):
    param_dict = {}
    if args["json_path"]:  
        param_dict["json_path"] = pathlib.Path(args["json_path"]).absolute()

    if args["swot_dir"]:
        param_dict["swot_dir"] = pathlib.Path(args["swot_dir"]).absolute()

    if args["output_dir"]:
        param_dict["output_dir"] = pathlib.Path(args["output_dir"]).absolute()
    
    if args["log_dir"]:
        param_dict["log_dir"] =  pathlib.Path(args["log_dir"]).absolute()

    if args["run_type"]:
        if args["run_type"] == "seq":
            param_dict["set_run"] =  False
            param_dict["seq_run"] =  True
        elif args["run_type"] == "set":
            param_dict["set_run"] =  True
            param_dict["seq_run"] =  False
        else :
            print("WARNING : run_type %s not known" %args["run_type"])
    else :
        param_dict["set_run"] = params.set_run
        param_dict["seq_run"] = params.seq_run

    if args.AWS:
        param_dict["sos_dir"] = param_dict["input_dir"].joinpath('sos')
        param_dict["swot_dir"] = param_dict["input_dir"].joinpath('swot')
        param_dict["sword_dir"] = param_dict["input_dir"].joinpath('sword')
    else:
        if args.sos_dir:
            param_dict["sos_dir"] = args.sos_dir
        else:
            param_dict["sos_dir"] = pathlib.Path(args["sos_dir"]).joinpath(param_dict["constraint"])

    if args["swot_dir"]:
        param_dict["swot_dir"] = pathlib.Path(args["swot_dir"])
    
    if args["sword_dir"]:
        param_dict["sword_dir"] = pathlib.Path(args["sword_dir"])

    if not param_dict["output_dir"].exists():
        param_dict["output_dir"].mkdir(parents=True, exist_ok=True)
    if not param_dict["log_dir"].exists():
        param_dict["log_dir"].mkdir(parents=True, exist_ok=True)
    
    for param in args.keys():
        if param.startswith("node_level_") or param.startswith("reach_level_"):
            param_dict[param] = args[param]

    return param_dict


def main():
    
    parser = argparse.ArgumentParser(description='Sic4dvar algorithm.')
    parser.add_argument('--config_file_path', help="json file name to run.",  default=pathlib.Path(__file__).parent / "sic4dvar_param.ini", type=pathlib.Path)
    parser.add_argument('--json_path', help="json file name to run.", type=pathlib.Path)
    parser.add_argument('--output_dir', help="Output directory", type=pathlib.Path)
    parser.add_argument('--log_dir', help="Log directory", type=pathlib.Path)
    parser.add_argument('--swot_dir', help="Input swot directory", type=pathlib.Path)
    parser.add_argument('--sos_dir', help="SWORD Of Science dir", type=pathlib.Path)
    parser.add_argument('--sword_dir', help="SWORD dir", type=pathlib.Path)
    parser.add_argument('--input_dir', help="Input directory with swot, sos, and sword directories inside", type=pathlib.Path)

    parser.add_argument('-v', '--verbose', help="Be verbose, log level debug", default=logging.INFO, action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()

    if params.replace_config:
        args.config_file_path = pathlib.Path(params.config_file_path)

    param_dict = read_config(args.config_file_path)

    args_without_none = {k: v for k, v in vars(args).items() if v is not None}
    param_dict.update(args_without_none)

    log_path = param_dict["log_dir"].joinpath("sic4dvar.log")
    
    set_logger(args.loglevel, log_path)
    
    logging.info("Configuration is : ")
    for param, val in param_dict.items():
        logging.info("    %s : %s" % (param, val))
    logging.info("")

    if param_dict["aws"]:
        

        param_dict["json_path"] = pathlib.Path(param_dict["input_dir"],"reaches.json")

        param_dict["sos_dir"] = param_dict["input_dir"].joinpath('sos')
        param_dict["swot_dir"] = param_dict["input_dir"].joinpath('swot')
        param_dict["sword_dir"] = param_dict["input_dir"].joinpath('sword')

        if not param_dict["json_path"].exists():
            logging.error(f"JSON file not found in")

    param_dict["run_type"], reach_dict = get_reach_dataset_all(param_dict["json_path"])

    if param_dict["run_type"] == "seq":
        sic4dvar_seq_run(param_dict)
    elif param_dict["run_type"]=="set":
        sic4dvar_set_run(param_dict, reach_dict)
    else:
        logging.error(f"Unknown run type {param_dict['run_type']}")

if __name__ == "__main__":
    main()

