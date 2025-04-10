import logging
import numpy as np

def get_nc_variable_data(nc_ds, var_name):
    try:
        data = nc_ds[var_name][:]
    except:
        logging.warning(f'Error loading variable {var_name}, set to empty')
        data = np.array([])
    return data