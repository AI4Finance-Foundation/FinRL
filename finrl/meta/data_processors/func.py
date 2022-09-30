from __future__ import annotations

import datetime
import os


# filename: str
# output: stockname
def calc_stockname_from_filename(filename):
    return filename.split("/")[-1].split(".csv")[0]


def calc_all_filenames(path):
    dir_list = os.listdir(path)
    dir_list.sort()
    paths2 = []
    for dir in dir_list:
        filename = os.path.join(os.path.abspath(path), dir)
        if ".csv" in filename and "#" not in filename and "~" not in filename:
            paths2.append(filename)
    return paths2


def calc_stocknames(path):
    filenames = calc_all_filenames(path)
    res = []
    for filename in filenames:
        stockname = calc_stockname_from_filename(filename)
        res.append(stockname)
    return res


def remove_all_files(remove, path_of_data):
    assert remove in [0, 1]
    if remove == 1:
        os.system("rm -f " + path_of_data + "/*")
    dir_list = os.listdir(path_of_data)
    for file in dir_list:
        if "~" in file:
            os.system("rm -f " + path_of_data + "/" + file)
    dir_list = os.listdir(path_of_data)

    if remove == 1:
        if len(dir_list) == 0:
            print(f"dir_list: {dir_list}. Right.")
        else:
            print(
                "dir_list: {}. Wrong. You should remove all files by hands.".format(
                    dir_list
                )
            )
        assert len(dir_list) == 0
    else:
        if len(dir_list) == 0:
            print(f"dir_list: {dir_list}. Wrong. There is not data.")
        else:
            print(f"dir_list: {dir_list}. Right.")
        assert len(dir_list) > 0


def date2str(dat):
    return datetime.date.strftime(dat, "%Y-%m-%d")


def str2date(str_dat):
    return datetime.datetime.strptime(str_dat, "%Y-%m-%d").date()
