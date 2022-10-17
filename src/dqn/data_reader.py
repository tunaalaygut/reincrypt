import os
from natsort import natsorted
import sys
sys.path.append('../')
from utility.r_utils import str_to_ndarray


class DataReader:
    def get_filelist(self, rootpath):
        pathlist = list()

        country = os.listdir(rootpath)
        for i in range(len(country)):
            country[i] = rootpath + str(country[i]) + '/'

        datelist = list()
        for i in range(len(country)):
            datelist = os.listdir(country[i])

            for j in range(len(datelist)):
                pathlist.append(country[i] + datelist[j] + '/')

        pathlist.sort()
        print('numof all data: ', len(pathlist))
        return pathlist

    def read(self, data_dirs: list, limit=None) -> tuple:
        X, y = [], []
        
        date_begin, date_end = None, None
        first_file = natsorted(os.listdir(data_dirs[0]))[0]
        last_file = natsorted(
            os.listdir(data_dirs[0]))[limit -1 if limit else -1]

        with open(os.path.join(data_dirs[0], first_file), "r+") as f:
            date_begin = f.read().split("$")[-1].strip()
        with open(os.path.join(data_dirs[0], last_file), "r+") as f:
            date_end = f.read().split("$")[-1].strip()

        for data_dir in natsorted(data_dirs):
            X_sub, y_sub = [], []
            for rimg_file in natsorted(os.listdir(data_dir)):
                filepath = os.path.join(data_dir, rimg_file)
                with open(filepath, "r+") as f:
                    X_part, y_part, _ = f.read().split("$")
                    X_sub.append(str_to_ndarray(X_part.strip()))
                    y_sub.append(float(y_part.strip()))
                if(limit and len(X_sub) == limit): break
            X.append(X_sub)
            y.append(y_sub)

        return X, y, date_begin, date_end
