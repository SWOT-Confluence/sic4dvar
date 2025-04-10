import netCDF4 as nc
import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from lib.lib_variables import code_to_continent, continent_to_station
from lib.lib_dates import daynum_to_date, seconds_to_date

class SOS:

    def __init__(self, path, reach_ids_list=[], constraint='constrained', version='16'):
        if np.array(reach_ids_list).size > 0:
            continent_codes = np.unique(np.array(reach_ids_list).astype('U1')).astype('int')
            continents = np.vectorize(code_to_continent.get)(continent_codes)
        else:
            continents = np.unique(np.vectorize(code_to_continent.get)(np.arange(1, 10).astype(int)))
        self.datasets = np.unique(np.char.add(np.char.lower(continents), f'_sword_v{version}_SOS_priors.nc'))
        self.datasets = np.array([path / constraint / p for p in self.datasets])
        self.datasets = np.vectorize(nc.Dataset)(self.datasets)
        self.constraint = constraint
        self.version = version
        self.reach_ids_per_continent = {}
        self.stations_reach_ids = {}
        self.init_reach_ids_all(reach_ids_list)

    def init_reach_ids_all(self, reach_ids_list=[]):
        reach_ids_per_continent = {}
        if np.array(reach_ids_list).size > 0:
            for code in reach_ids_list:
                first_digit = str(code)[0]
                region = code_to_continent.get(int(first_digit), 'Unknown')
                if region not in reach_ids_per_continent:
                    reach_ids_per_continent[region] = []
                reach_ids_per_continent[region].append(code)
            for continent in reach_ids_per_continent:
                reach_ids_per_continent[continent] = np.unique(reach_ids_per_continent[continent]).astype(int)
            self.reach_ids_per_continent = reach_ids_per_continent
        else:
            for dataset in self.datasets:
                name = dataset.getncattr('Name')
                reach_ids_per_continent[name] = {}
                sos_rids = dataset['reaches']['reach_id'][:].data
                reach_ids_per_continent[name] = np.unique(sos_rids)
            self.reach_ids_per_continent = reach_ids_per_continent

    def get_continent_name_from_reach_id(self, reach_id):
        continent = code_to_continent.get(int(str(reach_id[0])))
        return continent

    def get_which_dataset_to_use(self, continent):
        for i, dataset in enumerate(self.datasets):
            if dataset.getncattr('Name') == continent:
                return i

    def get_sos_index_for_reach_id(self, dataset, reach_id):
        sos_rids = dataset['reaches']['reach_id'][:]
        index_sos = np.where(sos_rids == int(reach_id))
        return index_sos

    def retrieve_annual_model(self, reach_id):
        print(reach_id)
        continent = self.get_continent_name_from_reach_id(reach_id)
        index = self.get_which_dataset_to_use(continent)
        dataset = self.datasets[index]
        index_sos = self.get_sos_index_for_reach_id(dataset, reach_id)
        sos_model = dataset['model']['mean_q'][index_sos][0]
        return sos_model

    def retrieve_monthly_q_model(self, reach_id):
        continent = self.get_continent_name_from_reach_id(reach_id)
        index = self.get_which_dataset_to_use(continent)
        dataset = self.datasets[index]
        index_sos = self.get_sos_index_for_reach_id(dataset, reach_id)
        sos_model = dataset['model']['monthly_q'][:][index_sos, :][0][0]
        return sos_model

    def get_stations_reach_ids(self):
        dataset = self.datasets[0]
        stations = dataset.getncattr('Gage_Agency')
        stations = stations.split(';')
        self.stations_reach_ids[dataset.getncattr('Name')] = {}
        for station in stations:
            self.stations_reach_ids[dataset.getncattr('Name')][station] = dataset[station]['%s_reach_id' % station][:].data
        print(self.stations_reach_ids)
        print(bug)

def get_station_q_and_qt(sos_dataset, reach_id):
    out_sos_q, out_sos_date = ([], [])
    continent_code = int(reach_id // 10000000000.0)
    station_names = continent_to_station[code_to_continent[continent_code]]
    sos_dataset_namegroup = list(sos_dataset.groups.keys())
    for station_name in set(station_names).intersection(sos_dataset_namegroup):
        station_reach_id = sos_dataset[station_name]['%s_reach_id' % station_name][:]
        station_reach_idx = np.where(station_reach_id == reach_id)
        station_q = sos_dataset[station_name]['%s_q' % station_name][station_reach_idx]
        station_qt = sos_dataset[station_name]['%s_qt' % station_name][station_reach_idx]
        if station_q.size > 0:
            if station_q.mask.any() or station_qt.mask.any():
                sos_valid_idx = np.where(np.logical_and(station_q.mask == False, station_qt.mask == False))
                out_sos_q = station_q[sos_valid_idx].ravel()
                sos_qt = station_qt[sos_valid_idx].ravel()
            else:
                out_sos_q = station_q[:].ravel()
                sos_qt = station_qt[:].ravel()
            out_sos_date = daynum_to_date(sos_qt, '0001-01-01')
    return (out_sos_q, out_sos_date)

def main():
    sos_dataset = nc.Dataset('/home/ccazals/Utils/SOS/v16d/unconstrained/eu_sword_v16d_SOS_priors.nc')
    station_q, station_qt = get_station_q_and_qt(sos_dataset, 23214400031)
    path = Path('/mnt/DATA/worksync/A trier/files/sos/v16c/')
    reach_ids_list = ['74100600021', '84100600021', '24100600021', '74100600021', '21602400121', '13121000051']
    constraint = 'constrained'
    version = '16c'
    sos_constrained = SOS(path, reach_ids_list, constraint, version)
    GRADES_annual = sos_constrained.retrieve_annual_model(reach_ids_list[0])
    GRADES_monthly_mean = sos_constrained.retrieve_monthly_q_model(reach_ids_list[0])
    print('GRADES_annual, GRADES_monthly_mean:', GRADES_annual, GRADES_monthly_mean)
if __name__ == '__main__':
    main()