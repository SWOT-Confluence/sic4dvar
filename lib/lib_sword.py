import logging
import numpy as np
from netCDF4 import Dataset
from lib.lib_netcdf import get_nc_variable_data

class Sword(object):
    def __init__(self, sword_path, sword_version="16", ):
        self.sword = Dataset(sword_path)
    
    def get_node_id_list(self, reach_id_list):
        nodes_idx = np.isin(get_nc_variable_data(self.sword, 'nodes/reach_id').filled(np.nan), reach_id_list)
        node_id_list = get_nc_variable_data(self.sword, 'nodes/node_id').filled(np.nan)[nodes_idx].astype(int)
        
        return node_id_list
    def close(self):
        self.sword.close()

def compute_sword_path_from_cont(sword_path, cont):

    for file in sword_path.glob(f"{cont.lower()}_sword_v*.nc"):
        return file
    logging.warning(f"SWORD file not found in {sword_path} for continent {cont}")
    return None
