import numpy as np
import gym
import settings
import random
import cv2
import pandas as pd
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


class log:

    # 绝对路径
    def __init__(self, abs_dir_path):
        self.buf = []
        self.abs_dir_path = ""
        self._mkdir(abs_dir_path)
        self.abs_dir_path = abs_dir_path

    def add(self, data):
        if isinstance(data,list):
            self.buf.extend(data)
        else:
            self.buf.append(data)

    # file_name should end with .csv
    def write(self, file_name, columns=None):
        frame = pd.DataFrame(self.buf, columns=columns)
        frame.to_csv(path_or_buf=self.abs_dir_path + "/" + file_name, index=False)

    # file_name should end with .csv
    # return pandas.DataFrame
    def read(self, file_name):
        frame = pd.read_csv(filepath_or_buffer=self.abs_dir_path + '/' + file_name)
        return frame

    def _mkdir(self, path):
        is_exist = os.path.exists(path)
        if not is_exist:
            os.mkdir(path)
            print("new folder in path: {}".format(path))


def get_env():
    return gym.make(settings.GAME)

# actual buf size is size+1
class RingBuf:
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        self.len = size + 1

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    # code style need to be improved
    def sample(self, batch_size):
        assert (self.end + 1) % self.len == self.start  # sample when buf is full
        if self.data[self.len - 1] == None:  # the last item is None when it just full
            return random.sample(self.data[self.start:self.end], batch_size)
        else:
            return random.sample(self.data, batch_size)  # sample from list that have a len of size+1

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
