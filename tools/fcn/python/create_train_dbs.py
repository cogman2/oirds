#!/usr/bin/env python
# Used to build the training images for a labeled data set.
# each training image is black (0,0,0).  The polygons representing each label
# has an assigned color in the image.


import matplotlib 
matplotlib.use('Agg') 
import lmdb
from PIL import Image
import gt_tool
import json_tools

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

    if len(sys.argv) < 1:
      print "Usage: ", sys.argv[0], " config"
      sys.exit( 0 )

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

    config = json_tools.loadConfig(sys.argv[1])

    multiplicationFactor = json_tools.getMultFactor(config)
    
    gtTool = gt_tool.GTTool(config)
    gtTool.load()
   
    randUniform = random.seed(23361)

    out_db_train = lmdb.open('raw_train', map_size=int(4294967296))
    out_db_test = lmdb.open('raw_test', map_size=int(4294967296))
    label_db_train = lmdb.open('groundtruth_train', map_size=int(4294967296))
    label_db_test = lmdb.open('groundtruth_test', map_size=int(4294967296))

    testNames = gtTool.getTestNames(json_tools.getPercentageForTest(config))

    with out_db_train.begin(write=True) as odn_txn:
     with out_db_test.begin(write=True) as ods_txn:
      with label_db_train.begin(write=True) as ldn_txn:
       with label_db_test.begin(write=True) as lds_txn:
         writeOutImages(gtTool, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, config, multiplicationFactor)

    out_db_train.close()
    label_db_train.close()
    out_db_test.close()
    label_db_test.close()
    sys.exit(0)


def  writeOutImages(gtTool, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, config, multiplicationFactor):
    test_idx = [0]
    train_idx = [0]
    def f(lastname, lastList):
        idxUpdates = outputImages(lastname, lastList, gtTool, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, test_idx[0], train_idx[0],config,multiplicationFactor)
        test_idx[0] = test_idx[0] +  idxUpdates[1]
        train_idx[0] = train_idx[0] +  idxUpdates[0]
    gtTool.iterate(f)

def outputImages(name, imageData, gtTool, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, test_idx, train_idx,config,multiplicationFactor):
   print name + '-----------'
   for i in range(0,multiplicationFactor):
      labelImage, labelIndices, rawImage = createGTImg(name, imageData, gtTool, config,i>0)
      imageCropSizeMin = 0
      if (json_tools.isResize(config)):
        imageCropSizeMin = json_tools.getCropSize(config)
      if (rawImage.size[0] < imageCropSizeMin or rawImage.size[1] < imageCropSizeMin):
         print 'skipping'
         return (0,0)
      labelImage.save("./png_gt/" + name[0:name.rindex('.')] + "_" + str(i) + ".png")
      rawImage.save("./png_raw/"  + name[0:name.rindex('.')] + "_" + str(i) + ".png")
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
           
def createGTImg(name, xlsInfoList, gtTool, config, augment):
  initialSize, imRaw= gtTool.loadImage(name)
  imageCropSizeMin = json_tools.getCropSize(config)
  newImage, labelData = gtTool.createLabelImage(xlsInfoList, initialSize, imRaw, json_tools.getSingleLabel(config), augment)
  labelImage = labelData[0]
  labelIndices = labelData[1]
  if (newImage.size[0] > imageCropSizeMin):
     cx = labelData[2][0][0]
     cy = labelData[2][0][1]
     cxp = cx - imageCropSizeMin/2
     cyp = cy - imageCropSizeMin/2
     cxp = 0 if (cxp < 0) else cxp
     cyp = 0 if (cyp < 0) else cyp
     newImage = newImage.crop((cxp, cyp, cxp+imageCropSizeMin, cyp+imageCropSizeMin))
     labelImage = labelImage.crop((cxp, cyp, cxp+imageCropSizeMin, cyp+imageCropSizeMin))
     labelIndices = labelIndices[0:1,cxp:cxp+imageCropSizeMin, cyp:cyp+imageCropSizeMin]
  return labelImage, labelIndices, newImage
   
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

