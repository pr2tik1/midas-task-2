"""
Script to get filenames and store it in a csv file. 
Initially it is required to create csv file named test_labels.csv. 
Process the data gathered into labels(targets) of images. 
"""

import os, csv

f=open("test_labels.csv",'r+')
w=csv.writer(f)
for path, dirs, files in os.walk("data/trainPart1/"):
    for filename in files:
        w.writerow([filename])