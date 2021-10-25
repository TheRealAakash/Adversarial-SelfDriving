import sys
# from colors import colors
import datetime as dt
import os
import glob
import numpy as np
import math
import time


def update_print_line(data):
    sys.stdout.write("\r" + data.__str__())
    # sys.stdout.write("\r" + data.__str__() + "                                      ")
    sys.stdout.flush()
    # sys.stdout.flush()


'''def important(msg):
    print(colors.red(msg))'''


def iter_per_second(itr, start_time):
    time_diff = dt.datetime.today().timestamp() - start_time
    return round(itr / time_diff, 2)


def seconds_per_itr(itr, start_time):
    time_diff = dt.datetime.today().timestamp() - start_time
    return round(time_diff / itr, 2)


def time_left(games_left, games_per_second):
    return games_left / games_per_second


def delete_files(path):
    files = glob.glob(f'{path}/*.npy')
    for f in files:
        os.remove(f)


def calculate_dist(point1, point2):
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]

    dist = math.hypot(x2 - x1, y2 - y1)
    return dist


def get_time_data(start_time, ITRS, itr, unit):
    time_now = time.time()
    elapsed_time_seconds = time_now - start_time
    elapsed_time = dt.timedelta(seconds=int(elapsed_time_seconds))
    ep_left = ITRS - itr
    est_time_seconds = int(seconds_per_itr(itr, start_time) * ep_left)
    est_time = dt.timedelta(seconds=est_time_seconds)

    eps_per_second = iter_per_second(itr, start_time)
    seconds_per_ep = seconds_per_itr(itr, start_time)

    if est_time_seconds >= elapsed_time_seconds:
        time_data = f"[{elapsed_time} < {est_time}"
    else:
        time_data = f"[{elapsed_time} > {est_time}"

    if eps_per_second >= seconds_per_ep:
        time_data = f"{time_data},  {eps_per_second} {unit}/sec]"
    else:
        time_data = f"{time_data},  {seconds_per_ep} Secs/{unit}]"

    return time_data


def calculate_mid(line):
    mid = (int((line[0][0] + line[1][0]) // 2), int((line[0][1] + line[1][1]) // 2))
    return mid


def calculate_corner(line, side):
    if side == "left":
        cord = (int(line[0][0]), line[0][1])
    else:
        cord = (int(line[1][0]), line[1][1])

    return cord
