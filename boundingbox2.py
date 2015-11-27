#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:30:16 2015

@author: robertsneddon
"""
def main():
    import os
#
#    This is a quick hack to allow you to compute bounding boxes, if you
#   don't have the dataset with the polygon information.
#
    os.chdir('/Users/robertsneddon/Downloads') #Change to your directory as needed
    import pandas as pd
    
    total = pd.DataFrame()
    for i in range(20): 
        fname = "/data/oirds/DataSet_"+str(i+1)+"/DataSet"+str(i+1)+".xls"
        book = pd.read_excel(io=fname, sheetname=0, parse_cols=[1,2,3,7,8,9,15,46])
        total = total.append(book)
    
    total = total[['Image Path', 'Image Name', 'Target Number', 
                   'Intersection Polygon', 'Average Target Centroid',
                   'Mode of Target Type', 'Average Target Orientation',
                   'Mode of Image Size']]
    im_sizes = total.iloc[:,7].unique();
    im_sizes.sort();
    #print "Image sizes:\n"
    total.to_csv('datasets.csv')    
    
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
    
    myDatasets = pd.DataFrame(datasets);
    myMaxx = pd.DataFrame(maxx, columns=["Max X"]);
    myMinx = pd.DataFrame(minx, columns=["Min X"]);
    myMaxy = pd.DataFrame(maxy, columns=["Max Y"]);
    myMiny = pd.DataFrame(miny, columns=["Min Y"]);
#   print datasets
    newDatasets = pd.concat([myDatasets,myMinx,myMiny,myMaxx,myMaxy], axis=1)
    newDatasets.to_csv('newDatasets.csv', sep=',', encoding='utf-8')

            
if __name__=="__main__":
    main()
