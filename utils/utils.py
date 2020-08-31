from __future__ import division

import numpy as np
import os
import pandas as pd
import pytz
import re
import torch

from datetime import date, datetime
from dateutil import parser
from six import string_types

from settings import settings



#################################################################
#
#   CHECKPOINT
#
#################################################################

def load_checkpoint(resume_path, Model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        total_step_cnt = checkpoint['step']
        epoch = checkpoint['epoch']
    
        # Model.netD = copy_state_dict(checkpoint['netD'], Model.netD)
        # Model.netD_mul = copy_state_dict(checkpoint['netD_mul'], Model.netD_mul)
        # Model.model_fusion = copy_state_dict(checkpoint['model_fusion'], Model.model_fusion)
        # Model.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        # Model.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        print("=> loaded checkpoint '{}' (step {})"
              .format(resume_path, checkpoint['step']))
        return Model, total_step_cnt, epoch
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))


#################################################################
#
#   TIME CONVERSION SESSION
#
#################################################################

def datetime_to_foldername(dt):
    return dt.strftime('%m-%d-%y')


def datetime_to_filename(dt):
    return dt.strftime('%m-%d-%y_%H.csv')


def parse_timestamp_tz_aware(string):
    return parser.parse(string)


def parse_timestamp_tz_naive(string):
    """
    return the converted value of date string in format %m/%d/%y %H:%M:%S
    NOTE: doctest for specific timezone.
    >>> parse_timestamp_tz_naive('10/03/2020 01:01:24')
    datetime.datetime(2020, 10, 3, 1, 1, 24)
    >>> parse_timestamp_tz_naive('10/03/20 01:01:24')
    datetime.datetime(2020, 10, 3, 1, 1, 24)
    """
    STARTTIME_FORMAT_WO_CENTURY = '%m/%d/%y %H:%M:%S'
    STARTTIME_FORMAT_W_CENTURY = '%m/%d/%Y %H:%M:%S'
    try:
        dt = datetime.strptime(string, STARTTIME_FORMAT_WO_CENTURY)
    except:
        dt = datetime.strptime(string, STARTTIME_FORMAT_W_CENTURY)

    return dt


def datetime_to_epoch(dt):
    """
    return the epcoh value of date time object timestamp
    NOTE: doctest for specific timezone.
    >>> datetime_to_epoch(datetime(1970, 1, 1, tzinfo=pytz.utc))
    0
    """
    try:
        # for python 3
        return int(1000 * dt.timestamp())
    except:
        # for python 2, borrow from python source code: https://hg.python.org/cpython/file/3.3/Lib/datetime.py#l1428
        _EPOCH = datetime(1970, 1, 1, tzinfo=pytz.utc)
        return 1000 * ((dt - _EPOCH).total_seconds())


def datetime_str_to_unixtime(string):
    """
    return the converted value of datestring to unix timestamp
    NOTE: doctest for specific timezone.
    >>> datetime_str_to_unixtime('2017-10-03 01:01:24-05:00')
    1507010484000
    """

    return datetime_to_epoch(parse_timestamp_tz_aware(string))


def epoch_to_datetime(unix_time):
    """
    return the converted value of unix timestamp to time-zone specific date string
    """
    if len(str(abs(unix_time))) == 13:
        return datetime.utcfromtimestamp(unix_time / 1000). \
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])
    elif len(str(abs(unix_time))) == 10:
        return datetime.utcfromtimestamp(unix_time). \
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])


def check_end_with_timezone(end_time_zone_string):
    m = re.search(r'-\d+:\d+$', end_time_zone_string)
    if m:
        return True
    else:
        m = re.search(r'-\d{4}', end_time_zone_string)
    return True if m else False


def df_to_datetime_tz_aware(in_df, column_list):
    """

    Parameters
    ----------
    in_df: input dataframe for timezone-aware datetime conversion
    column_list: time column for timezone-aware datetime conversion

    Returns
    -------
    converted dataframe
        column name and format keeps the same as input dataframe

    """
    df = in_df.copy()

    for column in column_list:
        if len(df):  # if empty df, continue
            datetime_timestamp = df[column].iloc[0]
            # if type is string
            if isinstance(datetime_timestamp, string_types):  # if "import datetime" then "isinstance(x, datetime.date)"
                if check_end_with_timezone(datetime_timestamp):
                    # if datetime string end with time zone
                    df[column] = pd.to_datetime(df[column], utc=True).apply(
                        lambda x: x.tz_convert(settings['TIMEZONE']))
                    df[column] = pd.to_datetime(df[column], utc=True).apply(
                        lambda x: x.tz_convert(settings['TIMEZONE']))
                else:
                    # if no time zone contained
                    df[column] = pd.to_datetime(df[column]).apply(lambda x: x.tz_localize(settings['TIMEZONE']))

            # if type is datetime.date
            elif isinstance(datetime_timestamp, date):

                if datetime_timestamp.tzinfo is None or datetime_timestamp.tzinfo.utcoffset(datetime_timestamp) is None:
                    # if datetime is tz naive
                    df[column] = df[column].apply(lambda x: x.replace(tzinfo=pytz.UTC).astimezone(settings['TIMEZONE']))
                else:
                    # if datetime is tz aware
                    continue
            # if type is unixtime (13-digit int):
            elif isinstance(datetime_timestamp, (int, np.int64)) or isinstance(datetime_timestamp, (int, np.float)):
                # print(settings)
                df[column] = pd.to_datetime(df[column], unit='ms', utc=True) \
                    .dt.tz_convert(settings["TIMEZONE"])

            else:
                print('Cannot recognize the data type.')
        else:
            print('Empty column')

    return df


#################################################################
#
#   FILE/FOLDER OPERATION SESSION
#
#################################################################

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def list_files_in_directory(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]


def list_folder_in_directory(mypath):
    return [f for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def create_folder(f, deleteExisting=False):
    '''
    Create the folder

    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)

            deleteExising: if True then the existing folder will be deleted.

    '''
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


def get_sorted_dates_from_path(path):
    """
    get sorted dates (datetime object) from a given data folder

    Parameters
    ----------
    path: string

    Returns
    -------
    a list of datetimes

    """

    files = list_files_in_directory(path)
    files = [f for f in files if not f.startswith('.')]
    date_list = []
    for f in files:
        date_list.append(parser.parse(f[:8]))
    dates = list(set(date_list))
    dates.sort()
    dates = [mydate.astimezone(settings["TIMEZONE"]) for mydate in dates]
    return dates


def get_dates(subj):
    """
    get dates (sorted) of a subject

    Parameters
    ----------
    subj: string

    Returns
    -------
    a list of datetime

    """
    # path = os.path.join(settings["ROOT_DIR"], 'CLEAN', subj, 'NECKLACE')
    path = os.path.join(settings["ROOT_DIR"], 'CLEAN', subj, 'NECKLACE', 'UNSAMPLED')
    dates = get_sorted_dates_from_path(path)
    return dates
