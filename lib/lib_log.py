import logging
import fcntl
import os
log_messages = {'1': '', '2': '', '102': 'JSON file not found.', '103': 'Run on only one reach. Stopping.', '104': 'No SOS file {sos_file} found for reach of reach id: {reach_id}', '105': 'Reaches ending in something different from 1 are not processed in set mode. Skipping reach {reach_id}', '501': 'Option q_prior_from_stations used, but no station is available for {reach_id}.', '502': 'After filtering dates, no time instant left to compute mean of q prior from stations.', '503': "No time instant available on the reach to use stations' data.", '504': 'No SWORD file {sword_file} found for reach of reach id: {reach_id}', '505': 'No SWOT file {swot_file} found for reach of reach id: {reach_id}', '506': 'Mean of q prior from stations is <= 0. Exiting.', '999': 'Error message not found.'}

def call_error_message(error_code):
    try:
        message = log_messages[str(error_code)]
    except:
        message = log_messages['999']
    return message

def set_logger(loglevel, filename=None):
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)-4.4s]  %(message)s')
    logger = logging.getLogger()
    logger.setLevel(level=loglevel)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    if filename:
        if os.path.exists(filename):
            os.remove(filename)
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

def close_logger():
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

def append_to_principal_log(param_dict, message):
    log_path = param_dict['log_path']
    if not log_path.parent.exists():
        os.makedirs(log_path.parent, exist_ok=True)
    with open(log_path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        if f.read(1) == '':
            f.write(f'Configuration is : \n')
            for k, v in param_dict.items():
                f.write(f'     {k} : {v}\n')
        f.write(message + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)