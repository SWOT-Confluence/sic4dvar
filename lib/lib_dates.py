import netCDF4 as nc
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from lib.lib_verif import check_na

def get_swot_dates(node_t):
    dates_swot = []
    empty_date = []
    dates_sect = []
    for n in range(0, len(node_t)):
        dates_swot = []
        empty_date = []
        for t in range(0, len(node_t[0, :])):
            if not check_na(node_t[n, t]) and (not node_t[n].mask[t]):
                dates_swot.append(seconds_to_date(node_t[n, t]))
                dates_swot[t] = convert_to_YMD(str(dates_swot[t]))
            else:
                dates_swot.append('')
        dates_sect.append(dates_swot)
    prev_filled = 0
    index_most_dates = 0
    for i in range(0, len(dates_sect)):
        dates = np.array(dates_sect[i])
        filled_dates_count = np.count_nonzero(dates != '')
        if prev_filled < filled_dates_count:
            prev_filled = filled_dates_count
            index_most_dates = i
    return dates_sect[index_most_dates]

def get_swot_dates_old(folder_swot, reach_id):
    swot_file = folder_swot + str(reach_id) + '_SWOT.nc'
    swot_ds = nc.Dataset(swot_file)
    dates_swot = []
    empty_date = []
    dates_sect = []
    for n in range(0, len(swot_ds['node']['time'])):
        dates_swot = []
        empty_date = []
        for t in range(0, len(swot_ds['node']['time'][0, :])):
            if not check_na(swot_ds['node']['time'][n, t]):
                dates_swot.append(seconds_to_date(swot_ds['node']['time'][n, t]))
                dates_swot[t] = convert_to_YMD(str(dates_swot[t]))
            else:
                dates_swot.append('')
        dates_sect.append(dates_swot)
    prev_filled = 0
    index_most_dates = 0
    for i in range(0, len(dates_sect)):
        dates = np.array(dates_sect[i])
        filled_dates_count = np.count_nonzero(dates != '')
        if prev_filled < filled_dates_count:
            prev_filled = filled_dates_count
            index_most_dates = i
    return dates_sect[index_most_dates]

def datetime2matlabdn(dt):
    ord = dt.toordinal()
    mdn = dt + datetime.timedelta(days=366)
    frac = (dt - datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac

def convert_to_YMD(string):
    return string[0:string.rfind(' ')]

def daynum_to_date(day_num, reference_date):
    day_num = day_num
    time_delta = day_num * timedelta(days=1)
    reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
    new_datetime = reference_date + time_delta
    new_datetime = [date.replace(microsecond=0, second=0, minute=0, hour=0) for date in new_datetime]
    return new_datetime

def seconds_to_date(seconds, reference_date=None):
    time_delta = seconds * timedelta(seconds=1)
    if reference_date:
        reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
    else:
        reference_date = datetime(2000, 1, 1, 0, 0, 0)
    new_datetime = reference_date + time_delta
    new_datetime = new_datetime.replace(microsecond=0, second=0, minute=0, hour=0)
    return new_datetime

def seconds_to_date_old(seconds):
    time_delta = timedelta(seconds=int(seconds))
    reference_date = datetime(2000, 1, 1, 0, 0, 0)
    new_datetime = reference_date + time_delta
    return new_datetime

def date_to_seconds(input_date_str):
    date_format = check_date_format(input_date_str)
    input_date = datetime.strptime(str(input_date_str), date_format)
    reference_date = datetime(2000, 1, 1, 0, 0, 0)
    time_difference = (input_date - reference_date).total_seconds()
    return time_difference

def read_input_date(input_date_str):
    date_format = check_date_format(input_date_str)
    if date_format != None:
        input_date = datetime.strptime(str(input_date_str), date_format)
    else:
        input_date = None
    return input_date

def check_date_format(date_str):
    formats_to_check = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%d-%m-%Y', '%Y-%m-%d']
    for date_format in formats_to_check:
        try:
            datetime.strptime(str(date_str), date_format)
            return date_format
        except ValueError:
            pass
    return None

def is_leap_year(year):
    if year % 4 != 0:
        return False
    elif year % 4 == 0:
        if year % 100 == 0 and year % 400 != 0:
            return False
        else:
            return True

def count_days_in_months(start_date, end_date, days_in_months):
    current_date = start_date
    while current_date < end_date:
        year_month = (current_date.year, current_date.month)
        next_month = current_date.replace(day=28) + timedelta(days=4)
        end_of_month = next_month - timedelta(days=next_month.day)
        period_end_date = min(end_of_month, end_date - timedelta(days=1))
        days_in_month = (period_end_date - current_date).days + 1
        if year_month in days_in_months:
            days_in_months[year_month] += days_in_month
        else:
            days_in_months[year_month] = days_in_month
        current_date = period_end_date + timedelta(days=1)
    return days_in_months

def main():
    time = 783925729.161
    print('time:', time, seconds_to_date(np.array([time]), reference_date='2000-01-01'))
    time = 733536312.889
    print('time:', time, seconds_to_date(np.array([time]), reference_date='2000-01-01'))
if __name__ == '__main__':
    main()