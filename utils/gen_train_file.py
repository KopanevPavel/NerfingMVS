from glob import glob
import numpy as np
import os
import sys
from os.path import dirname as up

def read_data(directory):
    names = []
    for filename in sorted(glob(directory+"*")):
        print(filename)
        head, tail = os.path.split(filename)
        names += [tail]

    textfile = open(up(directory[:-1])+"/train.txt", "w")
    for element in names:
        textfile.write(element + "\n")
    textfile.close()

if __name__ == '__main__':
   read_data(sys.argv[1])
