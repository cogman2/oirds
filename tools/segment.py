#!/usr/bin python
# Get an OIRDS data subset with single-object images.
# Create the train.txt, val.txt and label.txt files.
import os
import pandas as pd

# Excel Columns
# 1, 2, 3 = "Image Path", "Image Name", "Target Number"
# 7, 8 = "Intersection Polygon", "Average Target Centroid"
# 9, 15 = "Mode of Target Type", "Average Target Orientation"

total = pd.read_csv('/data/OIRDS/datasets.csv', index_col=0)
un_img = total[total.iloc[:,2]==1] # Target Number

un_types = set(un_img.iloc[:,5])
n = len(un_types)
# thanks, Greg Hewgill, Dan Lenski and Matt Lavin
type_dict = dict(zip(un_types, range(n)))
def label(x):
    return type_dict[x]

os.chdir('/data/OIRDS/train/crop')
with open('train.txt', 'w+') as train:
    with open('val.txt', 'w+') as test:
        for i in range(len(un_img)):
            line = un_img.iloc[i,1][:-3]+'png '+str(label(un_img.iloc[i,5]))+'\n'
            if(i%20==0):
                test.write(line)
            else:
                train.write(line)

with open('labels.txt', 'w+') as f:
    for vehicle in un_types:
        f.write(vehicle[8:]+'\n')


