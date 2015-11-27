#!/usr/bin/env python
# Get an OIRDS data subset with single-object images.
# Create the train.txt, val.txt and label.txt files.
import glob
import itertools
import os
import pandas as pd
import random
import sys
import img_manip
from PIL import Image
import numpy

imagePathColName = 'Image Path'
imageNameColName = 'Image Name'
modeColName = 'Mode of Target Type'
centroidColName = 'Average Target Centroid'
occlusionColName = 'Average Target Occlusion %'
shadowColName = 'Average Target Shadow %'
#  1 = Image Path
#  2 = Image Name
#  3 = Target Number
#  7 = Intersection Polygon
#  8 = Average Target Centroid
#  9 = Mode of Target Type
# 15 = Average Target Orientation
# 25 - Average Target Occlusion %
# 28 - Average Target Shadow %

# define thresholds
occlusionThreshold = 0.1
shadowThreshold = 0.1

cols = [1, 2, 8, 9, 25, 28]

defaultDim = (50,50)

modes = ['VEHICLE/CAR', 'VEHICLE/PICK-UP', 'VEHICLE/TRUCK', 'VEHICLE/UNKNOWN', 'VEHICLE/VAN']
mode_indices = dict( zip( modes, [str(x) for x in range( len(modes) )] ) )

files = []

def get_xls_dataframe( xlsFiles, colIndices ):
    xlsInfo = pd.DataFrame()
    for xlsFile in xlsFiles:
        xlsInfo = xlsInfo.append( pd.read_excel( io=xlsFile, parse_cols=colIndices, ignore_index=True ) )
    return xlsInfo

def get_output_filename( filename, label_idx, angle_deg ):
    output_filename = filename[:-4]+"_"+label_idx
    if angle_deg > 0:
        output_filename += output_filename + "_angle_" + angle_deg
        
    output_filename += filename[-4:]
    return output_filename

def boundary_fixed_centroid( centroid, size ):
    # check lower bounds
    new_centroid = [centroid[i] if centroid[i] > defaultDim[i]//2 else defaultDim[i]//2 for i in xrange(len(centroid))]

    # check upper bounds
    new_centroid = [new_centroid[i] if new_centroid[i] < (size[i]-defaultDim[i]//2) else (size[i]-defaultDim[i]//2) for i in xrange(len(centroid))]

    return new_centroid

def write_files( train_percent ):
    file_set = set( files )
    train_files = set([ x for x in files if random.random() < train_percent ])
    test_files = file_set - train_files
    train = open( 'train.txt', 'w+' )
    for train_file in train_files:
        train.write( train_file + '\n' )
    test = open( 'test.txt', 'w+' )
    for test_file in test_files:
        test.write( test_file + '\n' )
    labels = open( 'labels.txt', 'w+' )
    for modeKey in mode_indices.keys():
        labels.write( modeKey + " " + mode_indices[modeKey] + '\n' )

def processXls( filename, parentDataDir ):
    xlsInfo = get_xls_dataframe( [filename], cols )
    for i in xrange(len(xlsInfo)):
        filename = str(parentDataDir +
                       xlsInfo.iloc[i][imagePathColName][3:] + "/" +
                       xlsInfo.iloc[i][imageNameColName]).replace( "Dataset", "DataSet" ).replace( '.tif', '.png' )
        occlusion = float(xlsInfo.iloc[i][occlusionColName])
        if occlusion > occlusionThreshold:
            continue
        shadow = float(xlsInfo.iloc[i][shadowColName])
        if shadow > shadowThreshold:
            continue
        mode = xlsInfo.iloc[i][modeColName]
        centroid = numpy.array([float(x) for x in xlsInfo.iloc[i][centroidColName][1:-1].split( ' ' )]) # get rid of []'s, and split into two

        img = img_manip.open( filename )
        img_size = img.size

        centroid = boundary_fixed_centroid( centroid, img_size )
        cropped_img = img_manip.crop( img, centroid, defaultDim )
        mode_index = mode_indices[mode]
        output_file_name = get_output_filename( filename, mode_index, -1 )
        cropped_img.save( output_file_name )

        files.append( output_file_name + " " + mode_index )


def processAllXlsFiles( parentDataDir ):
    if parentDataDir[-1] != '/':
        parentDataDir += "/"
    dataDirs = [parentDataDir+z for z in filter( lambda x: x.startswith( "DataSet_" ), os.listdir( parentDataDir ) )]
    xlsFiles = list( itertools.chain( *[ glob.glob( x + "/*.xls" ) for x in dataDirs ] ) )
    for xlsFile in xlsFiles:
        processXls( xlsFile, parentDataDir )
    write_files( 0.8 )
