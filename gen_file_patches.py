#!/usr/bin/env python
# Get an OIRDS data subset with single-object images.
# Create the train.txt, val.txt and label.txt files.
import glob
import itertools
import os
import pandas as pd
import random
import sys
import img_tools
from PIL import Image
import numpy
import shapely.geometry

imagePathColName = 'Image Path'
imageNameColName = 'Image Name'
intersectionColName = 'Intersection Polygon' 
modeColName = 'Mode of Target Type'
centroidColName = 'Average Target Centroid'
numTargetsColName = 'Average Number of Targets in Image'
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
# 43 - Average Number of Targets in Image

# define thresholds
occlusionThreshold = 0.1
shadowThreshold = 0.3

cols = [1, 2, 7, 8, 9, 25, 28, 43]

defaultDim = (50,50)

no_vehicle_mode = "NOVEHICLE"
modes = ['VEHICLE/CAR', 'VEHICLE/PICK-UP', 'VEHICLE/TRUCK', 'VEHICLE/UNKNOWN', 'VEHICLE/VAN', no_vehicle_mode]
mode_indices = dict( zip( modes, [str(x) for x in range( len(modes) )] ) )
bkgd_mode_index = mode_indices[ no_vehicle_mode ]

file_exclusion_regions = dict()

files = []

def get_xls_dataframe( xlsFiles, colIndices ):
    xlsInfo = pd.DataFrame()
    for xlsFile in xlsFiles:
        xlsInfo = xlsInfo.append( pd.read_excel( io=xlsFile, parse_cols=colIndices, ignore_index=True ) )
    return xlsInfo

def get_output_filename( filename, label_idx, angle_deg ):
    output_filename = filename[:-4]+"_Class_"+label_idx
    if angle_deg > 0:
        output_filename += "_angle_" + str(angle_deg)
        
    output_filename += filename[-4:]
    return output_filename

def boundary_fixed_centroid( centroid, size ):
    # check lower bounds
    new_centroid = [centroid[i] if centroid[i] > defaultDim[i]//2 else defaultDim[i]//2 for i in xrange(len(centroid))]

    # check upper bounds
    new_centroid = [new_centroid[i] if new_centroid[i] < (size[i]-defaultDim[i]//2) else (size[i]-defaultDim[i]//2) for i in xrange(len(centroid))]

    return new_centroid

def get_bkgd_patch( img, dims, exclusion_regions ):
    img_size = img.size
    patch_centroid = (0,0)
    done = False
    n_iter = 0
    while( not done ):
        if n_iter > 10000:
            return img
        patch_centroid = img_tools.random_patch_centroid( img, dims )
        patch_bbox = img_tools.bbox( patch_centroid, dims )
        intersects = False
        for exclusion_region in exclusion_regions:
            intersects = shapely.geometry.Polygon( patch_bbox ).intersects( shapely.geometry.Polygon( exclusion_region ) )
            if intersects:
                break
        if not intersects:
            done = True
            break
        n_iter += 1

    return img_tools.crop( img, patch_centroid, dims ) # return if it doesn't intersect

def get_bkgd_patches( img, dims, exclusion_regions ):
    img_size = img.size
    patches = []
    i_step = dims[0]
    j_step = dims[1]
    i_end = img_size[0]-i_step
    j_end = img_size[1]-j_step

    i = 0
    j = 0
    while i < i_end:
        intersects = False
        while j < j_end:
            shifted_i = i + i_step
            shifted_j = j + j_step
            patch_bbox = [ [i,j], [shifted_i, j], [shifted_i,shifted_j], [shifted_i, j] ]
            for exclusion_region in exclusion_regions:
                if shapely.geometry.Polygon( patch_bbox ).intersects( shapely.geometry.Polygon( exclusion_region ) ):
                    intersects = True
                    break
            if intersects:
                j += 1
            else:
                centroid =  ((i+shifted_i)//2,(j+shifted_j)//2)
                patches.append( img_tools.crop( img, centroid, dims ) )
                j += j_step
        if intersects:
            i += 1
        else:
            i += i_step #it's possible that this will cause us to miss some patches                                                                                                                                 
    return patches

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



        # get background patch
        intersection_list = [u.split(' ') for u in str(xlsInfo.iloc[i][intersectionColName])[1:-1].split(';')]
        try:
            intersection_region = [(int(v[0]), int(v[1])) for v in intersection_list]
            if not file_exclusion_regions.has_key( filename ):
                file_exclusion_regions[filename] = [intersection_region]
            else:
                file_exclusion_regions[filename] += [intersection_region]
        except ValueError:
            print 'Problems with intersection: ', intersection_list


        occlusion = float(xlsInfo.iloc[i][occlusionColName])
        if occlusion > occlusionThreshold:
            continue
        shadow = float(xlsInfo.iloc[i][shadowColName])
        if shadow > shadowThreshold:
            continue
        mode = xlsInfo.iloc[i][modeColName]
        centroid = numpy.array([float(x) for x in xlsInfo.iloc[i][centroidColName][1:-1].split( ' ' )]) # get rid of []'s, and split into two

        img = img_tools.open( filename )
        img_size = img.size

        mode_index = mode_indices[mode]
        output_file_name = get_output_filename( filename, mode_index, -1 )
        files.append( output_file_name + " " + mode_index )
        if not os.path.exists( output_file_name ):
            centroid = boundary_fixed_centroid( centroid, img_size )
            cropped_img = img_tools.crop( img, centroid, defaultDim )
            cropped_img.save( output_file_name )
        
        # loop through angles
        for i in xrange(1,7):
            mode_index = mode_indices[mode]
            angle = 30*i
            output_file_name = get_output_filename( filename, mode_index, angle )
            if not os.path.exists( output_file_name ):
                centroid = boundary_fixed_centroid( centroid, img_size )
                cropped_img = img_tools.crop_rotate( img, centroid, defaultDim, angle )
                cropped_img.save( output_file_name )
            files.append( output_file_name + " " + mode_index )

    # loop through backgrounds
    for fil in file_exclusion_regions.keys():
        bkgd_output_file_name = get_output_filename( fil, bkgd_mode_index, -1 )
        if os.path.exists( bkgd_output_file_name ):
            files.append( bkgd_output_file_name + " " + bkgd_mode_index )
        else:
            img = img_tools.open( fil )
            bkgd_patch = get_bkgd_patch( img, defaultDim, file_exclusion_regions[fil] )
            if bkgd_patch != img:
                bkgd_patch.save( bkgd_output_file_name )
                files.append( bkgd_output_file_name + " " + bkgd_mode_index )

def processAllXlsFiles( parentDataDir ):
    if parentDataDir[-1] != '/':
        parentDataDir += "/"
    dataDirs = [parentDataDir+z for z in filter( lambda x: x.startswith( "DataSet_" ), os.listdir( parentDataDir ) )]
    xlsFiles = list( itertools.chain( *[ glob.glob( x + "/*.xls" ) for x in dataDirs ] ) )
    for xlsFile in xlsFiles:
        processXls( xlsFile, parentDataDir )
    write_files( 0.8 )

if __name__ == "__main__":
    if len( sys.argv ) < 2:
        print "Usage: ", sys.argv[0], " OIRDS_dir"
        sys.exit(0)
    processAllXlsFiles( sys.argv[1] )

