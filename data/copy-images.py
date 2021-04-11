import os
import shutil

for root, dirs, files in os.walk('/home/pratik/Desktop/midas-task-2/data/mnistTask'):  # replace the . with your starting directory
   for file in files:
      path_file = os.path.join(root,file)
      shutil.copy2(path_file,'/home/pratik/Desktop/midas-task-2/data/mnist') # change you destination dir
