#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculates oceanloading displacement.

calc_ol takes a site, a start date, an end date, and a time interval and calculates
oceanloading displacements at the site.

Usage:
    calc_ol  [-h] [-m MODEL] [-f] [-p] [-v] <site> <start_date> <end_date> <interval>

Arguments:
    <site>: name of the station.
    <start_date>: start date in ISO format: 2001-01-01T12:00:00
    <end_date>: end date in ISO format.
    <interval>: interval between points in minutes.

Optional arguments:
    -h, --help              Shows help message and exit
    -m MODEL, --model MODEL Name of the ocean model to use (default: GOT00.2)
    -f, --file              If true then *.txt file with data is created
    -p, --plot              If true then plot is displayed
    -v, --verbose           Increase output verbosity

"""
import argparse
import logging
import pyol
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style

from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import numpy as np
import dateutil.parser
import pkg_resources
from typing import List



def write_ol_to_file(site: str, date_list: List[datetime.datetime], oldata: List[float], oceanModel: str, filename: str):
    """
    Saves the ocean loading time series to a file
    Args:
        site: Name of the station
        date_list: datetimes for the oceanloading calculations
        oldata: calculated displacements
        oceanModel: Name of the ocean model to used
        file: Name of the file to which the time series are saved
    """
    with open(filename, 'w') as f:
        f.write('# Site       : %s\n' % site)
        f.write('# Ocean Model: %s\n' % oceanModel)
        f.write('#DATE\t   TIME\t\t dU [m]\t   dW [m]\t dS[m]\n')
        for i in range(len(date_list)):
                f.write ('%s %9.6f %9.6f %9.6f\n' % (date_list[i].strftime('%Y-%m-%d %H:%M:%S'), oldata[i, 0], oldata[i, 1], oldata[i, 2]))
        f.write('#END OF FILE')


def plot_ol(site: str, date_list: [datetime.datetime], oldata: [float], oceanModel: str):
    """
    Plots ocean loading displacements

    Args:
        site: Name of the station
        date_list: datetimes for the oceanloading calculations
        oldata: calculated displacements
        oceanModel: Name of the ocean model to used
    """
    style.use(['classic', 'seaborn-whitegrid', 'seaborn-talk', 'seaborn-deep'])

    fig, ax = plt.subplots()
    ax.plot(date_list, oldata[:, 0] * 1000, label='Radial [mm]', linestyle='-')
    ax.plot(date_list, oldata[:, 1] * 1000, label='West   [mm]', linestyle='-')
    ax.plot(date_list, oldata[:, 2] * 1000, label='South  [mm]', linestyle='-')
    plt.ylabel('Displacement [mm]')
    plt.title(site.upper() + ", model: " + oceanModel)
    ax.legend(loc='best')
    fig.autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    # Just some parsing to datetime formats, then calling the actual computation file.
    parser = argparse.ArgumentParser(description='Compute displacements caused by ocean tides.')
    parser.add_argument('site', type=str,      help='Name of the station')
    parser.add_argument('start_date', type=str,help='Start date in ISO format: 2017-01-01T12:00:00')
    parser.add_argument('end_date', type=str,  help='End date in ISO format  : 2017-01-01T12:00:00')
    parser.add_argument('interval', type=float,help='Interval between points in minutes')
    parser.add_argument("-m","--model", type=str, required=False, default='GOT00.2', help='Name of the ocean loading model to use (default: GOT00.2)')
    parser.add_argument("-f", "--file", required=False, action="store_true", help='If true then *.txt file with data is created')
    parser.add_argument("-p", "--plot",  required=False, action="store_true", help='If true then plot is displayed')
    parser.add_argument("-v", "--verbose", required=False,action="store_true", help="Increase output verbosity" )
    args = parser.parse_args()

    calc_start_date = dateutil.parser.parse(args.start_date)
    calc_end_date = dateutil.parser.parse(args.end_date)
    calc_interval = datetime.timedelta(seconds=60*args.interval)
    oModel=args.model
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    logging.info('Using %s ocean model...' % args.model)
    logging.info('Start date    : %s' % calc_start_date.strftime('%Y-%m-%d %H:%M:%S'))
    logging.info('End   date    : %s' % calc_end_date.strftime('%Y-%m-%d %H:%M:%S'))
    logging.info('Interval [min]: %.2f' % args.interval)

    path = pkg_resources.resource_filename(__name__, args.site + '_' + calc_start_date.strftime('%y%j')  + '_' + calc_end_date.strftime('%y%j') +'_' + str(int(args.interval)) + '.txt')
    # Compute displacements (N,E,U)
    date_list = [calc_start_date + i*calc_interval  for i in range(int((calc_end_date-calc_start_date)/calc_interval))]
    py_data = pyol.calc_displacement(date_list, args.site, oModel)
    print(args)
    
    if args.file:  # if True then store results into file
        write_ol_to_file(site=args.site, date_list=date_list, oldata=py_data, oceanModel=oModel, filename=path)
        logging.info('Data stored in %s ' % path)
    if args.plot:  # if True then display results
        plot_ol(site=args.site, date_list=date_list, oldata=py_data, oceanModel=oModel)
