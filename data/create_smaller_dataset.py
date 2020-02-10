import shutil
import os
import numpy as np

destination = "/home/tudorm/deep-fluids/data/t_reduced/"
source="/home/tudorm/deep-fluids/data/t/"
size_new_dataset = 100

if not os.path.exists(destination):
    os.mkdir(destination)


file_list  = os.listdir(source)
print('number of data points is original set: ' + str(len(file_list)))
print('number of datapoint in reduced data set: ' + str(len(size_new_dataset)))

choice = np.random.choice(len(file_list), size_new_dataset)

for i in choice:
    file_to_copy = file_list[i]
    shutil.copy(source+file_to_copy,destination+file_to_copy)

