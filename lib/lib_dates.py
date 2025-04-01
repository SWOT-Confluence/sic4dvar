import netCDF4 as nc
from datetime import datetime, timedelta
import numpy as np

def get_swot_dates(folder_swot, reach_id):
    # 1) Read data from SWOT
    
    swot_file = folder_swot + str(reach_id) + "_SWOT.nc"
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
                # print(dates_swot)
                # print(bug)
                dates_swot[t] = convert_to_YMD(str(dates_swot[t]))  # convert to Y/M/D for use with SoS
            else:
                dates_swot.append('')
        
        dates_sect.append(dates_swot)
    
    prev_filled = 0
    index_most_dates = 0
    
    for i in range(0, len(dates_sect)):
        # Assuming 'dates' is your array of dates
        dates = np.array(dates_sect[i])
        # print(bug)
        
        # Count the number of non-empty (filled) dates
        filled_dates_count = np.count_nonzero(dates != '')
        
        if prev_filled < filled_dates_count:
            prev_filled = filled_dates_count
            index_most_dates = i
        
        # print("Number of filled dates:", filled_dates_count)
    
    # print(dates_sect[index_most_dates])
    # print(index_most_dates)
    # print(bug)
    
    return (dates_sect[index_most_dates])


def convert_to_YMD(string):
    # For SWOT
    return string[0:string.rfind(" ")]


def daynum_to_date(day_num, reference_date):
    day_num = day_num
    time_delta = day_num * timedelta(days=1)
    reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
    new_datetime = reference_date + time_delta
    new_datetime = [date.replace(microsecond=0, second=0, minute=0, hour=0) for date in new_datetime]
    return new_datetime


def seconds_to_date(seconds, reference_date=None):
    # print("seconds:", seconds)
    
    time_delta = seconds * timedelta(seconds=1)
    
    if reference_date:
        reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
    else:
        # Define the reference date '2000-01-01 00:00:00' in datetime format
        reference_date = datetime(2000, 1, 1, 0, 0, 0)  # for SWOT files
    
    # Calculate the new datetime by adding the time delta to the reference date
    
    new_datetime = reference_date + time_delta
    # print("date:", new_datetime)
    # print(time_difference)
    new_datetime = [date.replace(microsecond=0, second=0, minute=0, hour=0) for date in new_datetime]
    return new_datetime


def seconds_to_date_old(seconds):  # needed for current version of map generation
    # print("seconds:", seconds)
    
    time_delta = timedelta(seconds=int(seconds))
    
    # Define the reference date '2000-01-01 00:00:00' in datetime format
    reference_date = datetime(2000, 1, 1, 0, 0, 0)  # for SWOT files
    
    # Calculate the new datetime by adding the time delta to the reference date
    new_datetime = reference_date + time_delta
    
    # print(time_difference)
    return new_datetime


def date_to_seconds(input_date_str):
    # Define your input date in 'YYYY-MM-DD hh:mm:ss' format
    # input_date_str = '2002-07-14 12:34:56'
    
    # print(input_date_str)
    date_format = check_date_format(input_date_str)
    # print("format:", date_format)
    
    # Convert the input date string to a datetime object
    # input_date = datetime.strptime(str(input_date_str), '%Y-%m-%d %H:%M:%S')
    input_date = datetime.strptime(str(input_date_str), date_format)
    
    # print(input_date)
    # print(bug)
    
    # Define the reference date '2000-01-01 00:00:00' in datetime format
    reference_date = datetime(2000, 1, 1, 0, 0, 0)  # for SWOT files
    
    # Calculate the time difference in seconds
    time_difference = (input_date - reference_date).total_seconds()
    
    # print(time_difference)
    return time_difference


def read_input_date(input_date_str):
    # Define your input date in 'YYYY-MM-DD hh:mm:ss' format
    # input_date_str = '2002-07-14 12:34:56'
    
    # print(input_date_str)
    date_format = check_date_format(input_date_str)
    # print("format:", date_format)
    
    # Convert the input date string to a datetime object
    # input_date = datetime.strptime(str(input_date_str), '%Y-%m-%d %H:%M:%S')
    # print("date_format:", date_format, input_date_str)
    if date_format != None:
        input_date = datetime.strptime(str(input_date_str), date_format)
    
    else:
        input_date = None
    
    return input_date


def check_date_format(date_str):
    # formats_to_check = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S']
    formats_to_check = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%d-%m-%Y', '%Y-%m-%d']
    
    for date_format in formats_to_check:
        try:
            # print(date_format)
            datetime.strptime(str(date_str), date_format)
            return date_format  # Return the matching format
        except ValueError:
            pass
    
    return None  # None of the formats match

def check_na(value):
    """ check if the specified value is None, '', pd.na, np.nan, is_empty or masked """
    #By Isadora
    if value is None:
        return True

    if value == '':
        return True

    if value == '--':
        return True

    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
def main():
    time = 783925729.161
    print("time:", time, seconds_to_date(np.array([time]), reference_date="2000-01-01"))
    time = 733536312.889
    print("time:", time, seconds_to_date(np.array([time]), reference_date="2000-01-01"))


if __name__ == "__main__":
    main()

