import shutil
import os
destination = "/home/tudorm/deep-fluids/data/t/"
if not os.path.exists(destination):
    os.mkdir(destination)

path_npz = "/home/tudorm/deep-fluids/data/001/"        
i=0  
j=0
k=0
for folder in os.listdir(path_npz):
    #k=0
    for file in os.listdir(path_npz+folder):        # THINK ABOUT THE FIRST TIME POINT (DO YOU NEED IT?)
        shutil.copy(path_npz+folder+"/"+file,destination+"{}_{}_{}.npz".format(i,j,int(file.split(".")[0][-2:])))
        print(folder+"/"+file, " ... ", "{}_{}_{}.npz".format(i,j,int(file.split(".")[0][-2:])))
        k+=1
    j+=1
