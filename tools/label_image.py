#!/usr/bin/env python
# Used to build the training images for a labeled data set.
# each training image is white.  The polygons representing each label
# has an assigned color in the image.

#import h5py, os

import lmdb
label_colors = [(64,128,64),(192,0,128),(0,128,192),(0,128,64),(128,0,0)]
modes = ['VEHICLE/CAR','VEHICLE/PICK-UP','VEHICLE/TRUCK','VEHICLE/UNKNOWN','VEHICLE/VAN']
modeIndices = dict( zip( modes, [int(x) for x in range( len(modes) )] ) )

def main():
    import os
    from PIL import Image
    import pandas as pd
    import itertools
    import glob
    import sys
    import random
    from PIL import Image
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

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

    randUniform = random.seed(23361)

# 1, 2, 3 = "Image Path", "Image Name", "Target Number"
# 7, 9 = "Intersection Polygon", "Mode of Target Type"

    cols = [1,2,3,7,9]
    xlsInfo = pd.DataFrame()
    for xlsFile in xlsFiles:
       xlsInfo = xlsInfo.append( pd.read_excel( io=xlsFile, parse_cols=cols, ignore_index=True ) )

    xlsInfo['isTest'] = 0
    xlsInfo = xlsInfo.reset_index()
    out_db_train = lmdb.open('raw_train', map_size=int(94371840))
    out_db_test = lmdb.open('raw_test', map_size=int(94371840))
    label_db_train = lmdb.open('groundtruth_train', map_size=int(94371840))
    label_db_test = lmdb.open('groundtruth_test', map_size=int(94371840))

    lastList=[]
    lastname=''

    setIsTest(xlsInfo,0.18)
    for i,r in xlsInfo.iterrows():
       if (lastname!= r[2] and len(lastList) > 0):
           labelImage, rawImage = convertImg(lastname, lastList, parentDataDir)
           if (r[6]==1):
              outGT(rawImage, out_db_test)
              outGT(labelImage, label_db_test)
           else:
              outGT(rawImage, out_db_train)
              outGT(labelImage, label_db_train)
           #labelImage.save(parentDataDir+"/png/"+ lastname[0:lastname.index('.tif')] + ".png")
           # need to send to training or test set here!
           lastList=[]
       else:
           lastList.append(r)
       lastname=r[2]

    out_db_train.close()
    label_db_train.close()
    out_db_test.close()
    label_db_test.close()
    sys.exit(0)



#with h5py.File('train.h5','w') as H:
 #   H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
 #   H.create_dataset( 'y', data=y ) # note the name y given to the dataset!

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
  imRaw = Image.open(dir + '/png/' + xlsInfoList[0][2][0:xlsInfoList[0][2].index('.tif')] + '.png') 
  imLabel = Image.new("RGB", imRaw.size, "white")
  for r in xlsInfoList:
    poly = r[4].replace("[",'(').replace("]","").replace(";",",")
    beg = poly[1:poly.index(',')]
    poly = 'POLYGON (' + poly + ',' + beg + '))'
    polyObj = loads(poly)
    try:
        labelImage(imLabel, polyObj, label_colors[modeIndices[r[5]]])
    except:
        continue
  return imLabel, imRaw
   
def labelImage(img, poly, color):
  from shapely.geometry import Point
  width, _ = img.size
  for i, px in enumerate(img.getdata()):
     y = i / width
     x = i % width
     if poly.contains(Point(x, y)):
        img.putpixel((x, y), color)

def outGT (im, out_txn):
   import caffe
   im = np.array(im.resize(128,128))
   # convert to one dimensional ground truth labels
   tmp = np.uint8(np.zeros(im[:,:,0:1].shape))

   # - in Channel x Height x Width order (switch from H x W x C)
   tmp = tmp.transpose((2,0,1))
   im_dat = caffe.io.array_to_datum(tmp)
   out_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())

if __name__=="__main__":
    main()
