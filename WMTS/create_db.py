
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
import re
from os import listdir
from os.path import isfile, join

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
      print "Usage: ", sys.argv[0], " imageDir"
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

   totalCount ,skipInterval = testSkipInterval(sys.argv[1])
   count = 0

   testMapSize = int(float(totalCount* 256*256) * 0.11)
   trainMapSize = int(float(totalCount* 256*256) * 0.91)
   out_db_train = lmdb.open('raw_train', map_size=trainMapSize)
   out_db_test = lmdb.open('raw_test', map_size=testMapSize)
   label_db_train = lmdb.open('groundtruth_train', map_size=trainMapSize)
   label_db_test = lmdb.open('groundtruth_test', map_size=testMapSize)

   pattern = re.compile(".*_gt.png")

   with out_db_train.begin(write=True) as odn_txn:
    with out_db_test.begin(write=True) as ods_txn:
     with label_db_train.begin(write=True) as ldn_txn:
      with label_db_test.begin(write=True) as lds_txn:
        for f in os.listdir(path):
          if (isfile(join(path, f)) and pattern.match(f) != None)
            writeOutImages(getImages(f), odn_txn, ods_txn, ldn_txn, lds_txn, (count % skipInterval) == 0)
            count += 1

   out_db_train.close()
   label_db_train.close()
   out_db_test.close()
   label_db_test.close()
   sys.exit(0)

def getImages(path, fname):
   indices = np.load(join(path,fname.replace("_gt.png", ".npy")))
   rawimage = Image.open(join(path, fname.replace("_gt.png","_raw.png")))
   return (indices, rawimage)

def testSkipInterval(path, percentToTest=0.1):
  c = countFilesInDir(path)
  return c, int(float(c) * percentToTest)

def countFilesInDir(path):
  pattern = re.compoile(".*_gt.png")
  list_dir = os.listdir(path)
  count = 0
  for file in list_dir:
    if (isfile(join(mypath, f)) and pattern.match(f) != None):
      count += 1
  return count

def countFilesInDir(path):
  pattern = re.compoile(".*_gt.png")
  list_dir = os.listdir(path)
  count = 0
  for file in list_dir:
    if (isfile(join(mypath, f)) and pattern.match(f) != None):
      count += 1
  return count

def  writeOutImages(imageSet, odn_txn, ods_txn, ldn_txn, lds_txn, isTest):
   if isTest:
      outRaw(imageSet[1], ods_txn, test_idx + c)
      outGTLabel(imageSet[0], lds_txn, test_idx+c)
   else:
      outRaw(imageSet[1], odn_txn, train_idx + c)
      outGTLabel(imageSet[0], ldn_txn, train_idx+c)

def outRaw (im, out_txn, idx):
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

