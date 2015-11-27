#!/usr/bin/env python
# Get an OIRDS data subset with single-object images.
# Create the train.txt, val.txt and label.txt files.
import glob
import itertools
import os
import pandas as pd
import random
import sys

if len(sys.argv) < 2:
    print "Usage: ", sys.argv[0], " dataDir"
    sys.exit( 0 )


parentDataDir = sys.argv[1]
if parentDataDir[-1] != '/':
    parentDataDir += "/"
dataDirs = [parentDataDir+z for z in filter( lambda x: x.startswith( "DataSet_" ), os.listdir( parentDataDir ) )]
xlsFiles = list( itertools.chain( *[ glob.glob( x + "/*.xls" ) for x in dataDirs ] ) )

imagePathIndex = 'Image Path'
imageNameIndex = 'Image Name'
modeIndex = 'Mode of Target Type'
# 1, 2, 3 = "Image Path", "Image Name", "Target Number"
# 7, 8 = "Intersection Polygon", "Average Target Centroid"
# 9, 15 = "Mode of Target Type", "Average Target Orientation"

cols = [1,2,9]
xlsInfo = pd.DataFrame()
for xlsFile in xlsFiles:
    xlsInfo = xlsInfo.append( pd.read_excel( io=xlsFile, parse_cols=cols, ignore_index=True ) )

modes = [ str(x) for x in set(xlsInfo['Mode of Target Type']) ]
modeIndices = dict( zip( modes, [str(x) for x in range( len(modes) )] ) )

files = set([ parentDataDir +
              xlsInfo.iloc[i][imagePathIndex][3:] + "/" +
              xlsInfo.iloc[i][imageNameIndex]     + " " +
              modeIndices[xlsInfo.iloc[i][modeIndex]]
              for i in range(len(xlsInfo)) ])

corrected_files = set([x.replace( "Dataset", "DataSet" ) for x in files ])
test_files = set([ x for x in corrected_files if random.random() < 0.2 ])
train_files = corrected_files-test_files

train = open( 'train.txt', 'w+' )
for train_file in train_files:
    train.write( train_file + '\n' )
test = open( 'test.txt', 'w+' )
for test_file in test_files:
    test.write( test_file + '\n' )
labels = open( 'labels.txt', 'w+' )
for modeKey in modeIndices.keys():
    labels.write( modeKey + " " + modeIndices[modeKey] + '\n' )
