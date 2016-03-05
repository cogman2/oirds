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

    gtTool = gt_tool.GTTool(config)
    gtTool.load()
   
    randUniform = random.seed(23361)

    out_db_train = lmdb.open('raw_train', map_size=int(4294967296))
    out_db_test = lmdb.open('raw_test', map_size=int(4294967296))
    label_db_train = lmdb.open('groundtruth_train', map_size=int(4294967296))
    label_db_test = lmdb.open('groundtruth_test', map_size=int(4294967296))

    testSlice = json_tools.getTestSlice(config) if json_tools.hasTestSlice(config)  else None
    testNames = gtTool.getTestNames(json_tools.getPercentageForTest(config), testSlice)

    with out_db_train.begin(write=True) as odn_txn:
     with out_db_test.begin(write=True) as ods_txn:
      with label_db_train.begin(write=True) as ldn_txn:
       with label_db_test.begin(write=True) as lds_txn:
         writeOutImages(gtTool, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, config)

    out_db_train.close()
    label_db_train.close()
    out_db_test.close()
    label_db_test.close()
    sys.exit(0)


def  writeOutImages(gtTool, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, config):
    test_idx = [0]
    train_idx = [0]
    def f(lastname, lastList):
        idxUpdates = outputImages(lastname, lastList, gtTool, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, test_idx[0], train_idx[0],config)
        test_idx[0] = test_idx[0] +  idxUpdates[1]
        train_idx[0] = train_idx[0] +  idxUpdates[0]
    gtTool.iterate(f)

def echoFunction(x,y,z):
  return x,y,z

def getAugmentFunctions(config):
  return [echoFunction]

def outputImages(name, imageData, gtTool, odn_txn, ods_txn, ldn_txn, lds_txn, testNames, test_idx, train_idx,config):
   print name + '-----------'
   c = 0;
   oRawImage, olabelImage, olabelIndices =  createGTImg(name, imageData, gtTool, config)
   imageCropSize = json_tools.getCropSize(config)
   slide = json_tools.getSlide(config)
   if (imageCropSize == 0):
     imageCropSize = min(oRawImage.size[0], oRawImage.size[1])
   imageResize = imageCropSize
   if (json_tools.isResize(config)):
      imageResize = json_tools.getResize(config)
   if (oRawImage.size[0] < imageCropSize or oRawImage.size[1] < imageCropSize):
      print 'skipping'
      return (0,0)
   augmentFunctions = getAugmentFunctions(config)
   for augmentFunction in augmentFunctions:
      (aRawImage, aLabelImage, aLabelIndices) = augmentFunction(oRawImage, olabelImage, olabelIndices)
      for (labelImage, labelIndices, rawImage) in imageSetFromCroppedImage(aRawImage, aLabelImage, aLabelIndices, aRawImage.size, slide):
        if (imageResize != imageCropSize):
          rawImage = resizeImg(rawImage,imageResize)
          labelImage = resizeImg(labelImage,imageResize)
          labelIndices = resizeImg(labelIndices,imageResize)

        labelImage.save("./png_gt/" + name[0:name.rindex('.')] + "_" + str(i) + ".png")
        rawImage.save("./png_raw/"  + name[0:name.rindex('.')] + "_" + str(i) + ".png")
        c += 1
        if (name in testNames):
           outGT(rawImage, ods_txn, test_idx + i)
           outGTLabel(labelIndices, lds_txn, test_idx+i)
        else:
           outGT(rawImage, odn_txn, train_idx+i)
           outGTLabel(labelIndices, ldn_txn, train_idx+i)
   return (0,c) if (name in testNames) else (c,0)

def resizeImg(im, imageCropSize):
   wpercent = (imageCropSize/float(im.size[0]))
   hsize = int((float(im.size[1])*float(wpercent)))
   return im.resize((imageCropSize ,hsize),Image.ANTIALIAS).crop((0,0,imageCropSize, imageCropSize))

def imageSetFromCroppedImage( rawImage,labelImage, labelIndices, imageSize, imageCropSize, slide ):
  cx = imageSize[0] / slide
  cy = imageSize[1] / slide
  result = []
  for xi in xrange(int(cx)):
    for yi in xrange(int(cy)):
       result.append(cropImageAt(cx*slide, cxy*slide, imageCropSize, rawImage, labelImage, labelIndices, imageSize))
  return result

def createGTImg(name, xlsInfoList, gtTool, config):
  imRaw= gtTool.loadImage(name)
  newImage, labelData = gtTool.createLabelImage(xlsInfoList, imRaw, json_tools.getSingleLabel(config))
  labelImage = labelData[0]
  labelIndices = labelData[1]
  return newImage, labelImage, labelIndices

def cdropImageAt(cxp, cxy, imageCropSize, rawImage, labelImage, labelIndices, imageSize):
  cx = labelData[2][0][0]
  cy = labelData[2][0][1]
  cxe = max(cxp+imageCropSize, imageSize[0])
  cye = max(cxy+imageCropSize, imageSize[1])
  orawImage = newImage.crop((cxp, cyp,cxe, cye))
  olabelImage = labelImage.crop((cxp, cyp, cxe, cye))
  olabelIndices = labelIndices[0:1,cxp:cye,cyp:cye]
  return orawImage, olabelImage, olabelIndices

   
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

