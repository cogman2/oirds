#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:30:16 2015

@author: robertsneddon
"""
def main():
import os
os.chdir('/Users/robertsneddon/Downloads')
import pandas as pd

maxx = [];
minx = [];
maxy = [];
miny = [];
datasets = pd.read_csv('datasets.csv');
 
for i, item_i in enumerate(datasets['Intersection Polygon']):
#    print i;
    coords = item_i.replace("[","").replace("]","").split(";");     
        
    maxx.append(0);
    minx.append(10000000000);
    maxy.append(0);
    miny.append(10000000000);   
         
    for j, item_j in enumerate(coords):
#        print j;
        
        try:
            x_coord = int(item_j.split()[0]);
            y_coord = int(item_j.split()[1]);
        except:
            break;
                 
        if x_coord < minx[i]: minx[i] = x_coord;
        if y_coord < miny[i]: miny[i] = y_coord;
        if x_coord > maxx[i]: maxx[i] = x_coord;
        if y_coord > maxy[i]: maxy[i] = y_coord;

myDatasets = pd.DataFrame(datasets)
myMaxx = pd.DataFrame(maxx, columns=["Max X"])
myMinx = pd.DataFrame(minx, columns=["Min X"])
myMaxy = pd.DataFrame(maxy, columns=["Max Y"])
myMiny = pd.DataFrame(miny, columns=["Min Y"])
newDatasets = pd.concat([myDatasets,myMinx,myMiny,myMaxx,myMaxy], axis=1)
newDatasets.to_csv('newDatasets.csv', sep=',', encoding='utf-8')

            
if __name__=="__main__":
    main()
