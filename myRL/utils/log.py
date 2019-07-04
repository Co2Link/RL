import pandas as pd
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

class log:

    # 绝对路径
    def __init__(self,abs_dir_path):
        self.buf = []
        self.abs_dir_path = ""
        self.mkdir(abs_dir_path)
        self.abs_dir_path = abs_dir_path
    def add(self,data):
        self.buf.append(data)

    # file_name should end with .csv
    def write(self,file_name,columns=None):
        frame = pd.DataFrame(self.buf,columns=columns)
        frame.to_csv(path_or_buf=self.abs_dir_path+"/"+file_name,index=False)

    # file_name should end with .csv
    # return pandas.DataFrame
    def read(self,file_name):
        frame = pd.read_csv(filepath_or_buffer=self.abs_dir_path+'/'+file_name)
        return frame
    def mkdir(self,path):
        is_exist = os.path.exists(path)
        if not is_exist:
            os.mkdir(path)
            print("new folder in path: {}".format(path))

def DEBUG():
    log_test = log(dir_path=ROOT_DIR+"/log_test")

    my_array = np.arange(1,101).reshape(20,5)

    for i in my_array:
        log_test.add(i)
    log_test.write(file_name="file_name.csv",columns=['a','b','c','d','e'])
    print("#####")
    read_frame = log_test.read(file_name="file_name.csv")
    print(read_frame)

if __name__ == '__main__':
    DEBUG()
    # pass