#!/usr/bin/env python
# Used to build the training images for a labeled data set.
# each training image is white.  The polygons representing each label
# has an assigned color in the image.

#import h5py, os

import matplotlib 
matplotlib.use('Agg') 
import lmdb
from PIL import Image

label_colors = [(64,128,5),(192,0,1),(0,128,2),(0,128,3),(128,0,4)]
hacked_color = (64,49,7)
modes = ['VEHICLE/CAR','VEHICLE/PICK-UP','VEHICLE/TRUCK','VEHICLE/UNKNOWN','VEHICLE/VAN']
modeIndices = dict( zip( modes, [int(x) for x in range( len(modes) )] ) )
imageSizeCrop=128

def main():
    import os
    import pandas as pd
    import itertools
    import glob
    import sys
    import random
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    if len(sys.argv) < 2:
      print "Usage: ", sys.argv[0], " dataDir"
      sys.exit( 0 )


    if (os.path.isdir("./png_gt")):
      os.rmdir("./png_gt")
    if (os.path.isdir("./png_raw")):
      os.rmdir("./png_raw")
    if (os.path.isdir("./raw_train")):
      os.rmdir("./raw_train")
    if (os.path.isdir("./raw_ttest")):
      os.rmdir("./raw_test")
    if (os.path.isdir("./groundtruth_train")):
      os.rmdir("./groundtruth_train")
    if (os.path.isdir("./groundtruth_test")):
      os.rmdir("./groundtruth_test")

    os.mkdir("./png_gt",0755)
    os.mkdir("./png_raw",0755)

    parentDataDir = sys.argv[1]
    if parentDataDir[-1] != '/':
       parentDataDir += "/"
    dataDirs = [parentDataDir+z for z in filter( lambda x: x.startswith( "DataSet_" ), os.listdir( parentDataDir ) )]
    xlsFiles = list( itertools.chain( *[ glob.glob( x + "/*.xls" ) for x in dataDirs ] ) )

    imagePathIndex = 'Image Path'
    imageNameIndex = 'Image Name'
    modeIndex = 'Mode of Target Type'

    randUniform = random.seed(23361)

# 1, 2, 3 = "Image Path", "Image Name", "Target Number"
# 7, 9 = "Intersection Polygon", "Mode of Target Type"

    cols = [1,2,3,7,9]
    xlsInfo = pd.DataFrame()
    for xlsFile in xlsFiles:
       xlsInfo = xlsInfo.append( pd.read_excel( io=xlsFile, parse_cols=cols, ignore_index=True ) )

    xlsInfo['isTest'] = 0
    xlsInfo = xlsInfo.reset_index()
    out_db_train = lmdb.open('raw_train', map_size=int(4294967296))
    out_db_test = lmdb.open('raw_test', map_size=int(4294967296))
    label_db_train = lmdb.open('groundtruth_train', map_size=int(4294967296))
    label_db_test = lmdb.open('groundtruth_test', map_size=int(4294967296))

    setIsTest(xlsInfo,0.18)

    with out_db_train.begin(write=True) as odn_txn:
     with out_db_test.begin(write=True) as ods_txn:
      with label_db_train.begin(write=True) as ldn_txn:
       with label_db_test.begin(write=True) as lds_txn:
         writeOutImages(xlsInfo, parentDataDir, odn_txn, ods_txn, ldn_txn, lds_txn)

    out_db_train.close()
    label_db_train.close()
    out_db_test.close()
    label_db_test.close()
    sys.exit(0)


def  writeOutImages(xlsInfo, parentDataDir,odn_txn, ods_txn, ldn_txn, lds_txn):
    lastList=[]
    lastname=''
    test_idx = 0
    train_idx =0
    for i,r in xlsInfo.iterrows():
       if (lastname!= r[2] and len(lastList) > 0):
           labelImage, rawImage = convertImg(lastname, lastList, parentDataDir)
           if (rawImage.size[0] < imageSizeCrop or rawImage.size[1] < imageSizeCrop):
               continue
           if (r[6]==1):
              outGT(rawImage, ods_txn, test_idx)
              outGTLabel(labelImage, lds_txn, test_idx)
              test_idx+=1
           else:
              outGT(rawImage, odn_txn, train_idx)
              outGTLabel(labelImage, ldn_txn, train_idx)
              train_idx+=1
           labelImage.save("./png_gt/"+ lastname[0:lastname.index('.tif')] + ".png")
           rawImage.save("./png_raw/"+ lastname[0:lastname.index('.tif')] + ".png")
           # need to send to training or test set here!
           lastList=[]
       else:
           lastList.append(r)
       lastname=r[2]

def resizeImg(im):
   wpercent = (imageSizeCrop/float(im.size[0]))
   hsize = int((float(im.size[1])*float(wpercent)))
   return im.resize((imageSizeCrop ,hsize),Image.ANTIALIAS).crop((0,0,imageSizeCrop, imageSizeCrop))
   
def resize(poly, initialSize):
   from shapely.geometry import polygon
   from shapely.ops import transform
   wpercent = (imageSizeCrop/float(initialSize[0]))
   hpercent = (imageSizeCrop/float(initialSize[1]))
   return transform(lambda x, y, z=None: (x*wpercent,y*hpercent), poly)


def setIsTest(xlsInfo, percent):
    import numpy as np
    labelCounts = [0 for i in xrange(len(label_colors))]
    for i,r in xlsInfo.iterrows():  
       labelCounts[modeIndices[r[5]]] += 1
    for i in xrange(len(labelCounts)):
      labelCounts[i] = int(labelCounts[i]*percent)
    for i in np.random.permutation(len(xlsInfo)):
       if(labelCounts[modeIndices[xlsInfo.iloc[i,5]]] > 0):
         xlsInfo.loc[i,'isTest']=1
         labelCounts[modeIndices[xlsInfo.iloc[i,5]]] -= 1

def convertImg(name,xlsInfoList, dir):
  from shapely.wkt import dumps, loads
#  from shapely import dumps, loads
  from shapely.geometry import polygon
  from PIL import Image
  print name + '-----------'
  imRaw = Image.open(dir + 'png/' + name[0:name.index('.tif')] + '.png') 
  initialSize = imRaw.size
  imRaw = resizeImg(imRaw)
  imLabel = Image.new("RGB", imRaw.size)
  for r in xlsInfoList:
    poly = r[4].replace("[",'(').replace("]","").replace(";",",")
    beg = poly[1:poly.index(',')]
    poly = 'POLYGON (' + poly + ',' + beg + '))'
    polyObj = loads(poly)
    polyObj = resize(polyObj, initialSize)
    try:
        labelImage(imLabel, polyObj, hacked_color) 
# label_colors[modeIndices[r[5]]])
    except:
        continue
  return imLabel, imRaw
   
def labelImage(img, poly, color):
  from shapely.geometry import Point
  width, length = img.size
  bounds = poly.bounds
  for x in (xrange(int(bounds[0]),int(bounds[2]))):
     for y in (xrange(int(bounds[1]),int(bounds[3]))):
       if poly.contains(Point(x, y)):
          img.putpixel((x, y), color)

def outGT (im, out_txn, idx):
   import caffe
   import numpy as np
   tmp = np.array(im)
   tmp = tmp.transpose((2,0,1))
   im_dat = caffe.io.array_to_datum(tmp)
   out_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())

def outGTLabel (im, out_txn, idx):
   import caffe
   import numpy as np
   tmp = np.array(im)
   # choose the last index, as it is the class number (B channel)
   tmp = tmp[:,:,2:3:1]
   tmp = tmp.transpose((2,0,1))
   im_dat = caffe.io.array_to_datum(tmp)
   out_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())

if __name__=="__main__":
    main()

