import numpy as np
import pandas as pd

def reorder_ids_with_indices(ids):
    indexed_ids = list(enumerate(ids))
    sorted_ids = sorted(indexed_ids, key=lambda x: x[1])
    sorted_ids.reverse()
    indexes = [index for index, _ in sorted_ids]
    values = [id_value for _, id_value in sorted_ids]
    return (indexes, values)

def check_na(value):
    if value is None:
        return True
    if value == '':
        return True
    if value == '--':
        return True
    if value is np.ma.masked:
        return True
    if isinstance(value, np.ma.core.MaskedConstant):
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    try:
        if np.isnan(value):
            return True
    except TypeError:
        pass
    try:
        if value.mask:
            return True
    except AttributeError:
        pass
    try:
        if value.is_empty:
            return True
    except AttributeError:
        pass
    return False

def verify_name_length(name, extension=''):
    if len(name) > 200:
        new_name = name[0:200]
        new_name = new_name + '[...]' + extension
    else:
        new_name = name
    return new_name