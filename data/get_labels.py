"""
Script to get filenames and store it in a csv file. 
Initially it is required to create csv file named test_labels.csv. 
Process the data gathered into labels(targets) of images. 

NOTE : Along with this Excel and other preprocessing steps within the notebooks were taken. So running this script will not reproduce the labels.csv file.

The steps were: 
> Initially a blank csv file is created. Then the file names in dataset folder is read and preprocessed as class. Finally stored in the blank csv, then loaded in notebooks for dataset class and data loaders. 

"""

import os, csv

f=open("test.csv",'r+')
w=csv.writer(f)

for path, dirs, files in os.walk("testPart1"):
    for filename in files:
        w.writerow([filename])
        

