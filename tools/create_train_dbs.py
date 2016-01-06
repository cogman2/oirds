#!/usr/bin/env python
# Used to build the training images for a labeled data set.
# each training image is white.  The polygons representing each label
# has an assigned color in the image.

import gt_tool
import matplotlib 
matplotlib.use('Agg') 
import lmdb
from PIL import Image

def main():
    import os
    import shutil
    import glob
    import sys
    import random
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    if len(sys.argv) < 3:
      print "Usage: ", sys.argv[0], " dataDir multiplicationFactor [labelIndex]"
      sys.exit( 0 )

    multiplicationFactor = int(sys.argv[2])
    singleLabelIndex = -1
    if (len(sys.argv) > 2):
      singleLabelIndex = int(sys.argv[3])

    if (os.path.isdir("./png_gt")):
      shutil.rmtree("./png_gt")
    if (os.path.isdir("./png_raw")):
      shutil.rmtree("./png_raw")
    if (os.path.isdir("./raw_train")):
      shutil.rmtree("./raw_train")
    if (os.path.isdir("./raw_test")):
      shutil.rmtree("./raw_test")
    if (os.path.isdir("./groundtruth_train")):
      shutil.rmtree("./groundtruth_train")
    if (os.path.isdir("./groundtruth_test")):
      shutil.rmtree("./groundtruth_test")

    os.mkdir("./png_gt",0755)
    os.mkdir("./png_raw",0755)

    parentDataDir = sys.argv[1]
    if parentDataDir[-1] != '/':
       parentDataDir += "/"
    
    xlsInfo = gt_tool.loadXLSFiles(parentDataDir)

    randUniform = random.seed(23361)

    out_db_train = lmdb.open('raw_train', map_size=int(4294967296))
    out_db_test = lmdb.open('raw_test', map_size=int(4294967296))
    label_db_train = lmdb.open('groundtruth_train', map_size=int(4294967296))
    label_db_test = lmdb.open('groundtruth_test', map_size=int(4294967296))

    testNames = gt_tool.getTestNames(xlsInfo,0.18)

    with out_db_train.begin(write=True) as odn_txn:
     with out_db_test.begin(write=True) as ods_txn:
      with label_db_train.begin(write=True) as ldn_txn:
       with label_db_test.begin(write=True) as lds_txn:
         writeOutImages(xlsInfo, parentDataDir, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, singleLabelIndex, multiplicationFactor)

    out_db_train.close()
    label_db_train.close()
    out_db_test.close()
    label_db_test.close()
    sys.exit(0)


def  writeOutImages(xlsInfo, parentDataDir,odn_txn, ods_txn, ldn_txn, lds_txn, testNames, singleLabelIndex, multiplicationFactor):
    lastList=[]
    lastname=''
    test_idx = 0
    train_idx =0
    for i,r in xlsInfo.iterrows():
       if (lastname!= r[2] and len(lastList) > 0):
           idxUpdates = outputImages(lastname, lastList, parentDataDir,odn_txn, ods_txn, ldn_txn, lds_txn, testNames, test_idx, train_idx,singleLabelIndex,multiplicationFactor)
           test_idx += idxUpdates[1]
           train_idx += idxUpdates[0]
           lastList=[]
       else:
           lastList.append(r)
       lastname=r[2]
    if (len(lastList) > 0):
       outputImages(lastname, lastList, parentDataDir,odn_txn, ods_txn, ldn_txn, lds_txn, testNames, test_idx, train_idx, singleLabelIndex,multiplicationFactor)

def outputImages(name, imageData, parentDataDir,odn_txn, ods_txn, ldn_txn, lds_txn, testNames, test_idx, train_idx,singleLabelIndex,multiplicationFactor):
   print name + '-----------'
   for i in range(0,multiplicationFactor):
      labelImage, labelIndices, rawImage = createGTImg(name, imageData, parentDataDir,singleLabelIndex,i>0)
      if (rawImage.size[0] < gt_tool.imageCropSize or rawImage.size[1] < gt_tool.imageCropSize):
         return (0,0)
      labelImage.save("./png_gt/" + name[0:name.index('.tif')] + "_" + str(i) + ".png")
      rawImage.save("./png_raw/"  + name[0:name.index('.tif')] + "_" + str(i) + ".png")
      if (name in testNames):
         outGT(rawImage, ods_txn, test_idx + i)
         outGTLabel(labelIndices, lds_txn, test_idx+i)
      else:
         outGT(rawImage, odn_txn, train_idx+i)
         outGTLabel(labelIndices, ldn_txn, train_idx+i)
   if (name in testNames):
      return (0,multiplicationFactor)
   else:
      return (multiplicationFactor,0)
           
def createGTImg(name,xlsInfoList, dir, singleLabelIndex, augment):
  initialSize, imRaw= gt_tool.loadImage(dir + 'png/' + name[0:name.index('.tif')] + '.png')
  newImage, labelData = gt_tool.createLabelImage(xlsInfoList, initialSize, imRaw, singleLabelIndex, augment)
  return labelData[0], labelData[1], newImage
   
def outGT (im, out_txn, idx):
   import caffe
   import numpy as np
   tmp = np.array(im)
   tmp= tmp[:,:,::-1]
   tmp = tmp.transpose((2,0,1))
   im_dat = caffe.io.array_to_datum(tmp)
   out_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())

def outGTLabel (imArray, out_txn, idx):
   import caffe
   import numpy as np
   im_dat = caffe.io.array_to_datum(imArray)
   out_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())

if __name__=="__main__":
    main()

