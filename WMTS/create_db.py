
#!/usr/bin/env python
# Used to build the training images for a labeled data set.
# each training image is black (0,0,0).  The polygons representing each label
# has an assigned color in the image.


import matplotlib 
matplotlib.use('Agg') 
import lmdb
from PIL import Image
import re
from os import listdir
from os.path import isfile, join
import os
import numpy as np

def main():
   import shutil
   import glob
   import sys
   import random
   if sys.version_info[0] < 3:
        from StringIO import StringIO
   else:
        from io import StringIO

   if len(sys.argv) < 2:
      print "Usage: ", sys.argv[0], " imageDir waterlabel amount"
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
   print "Skip interval is ", skipInterval, " out of ", totalCount

   count = 0

   testMapSize = int(float(totalCount* 256*256*20) * 0.11)
   trainMapSize = int(float(totalCount* 256*256*20) * 0.91)
   out_db_train = lmdb.open('raw_train', map_size=trainMapSize)
   out_db_test = lmdb.open('raw_test', map_size=testMapSize)
   label_db_train = lmdb.open('groundtruth_train', map_size=trainMapSize)
   label_db_test = lmdb.open('groundtruth_test', map_size=testMapSize)

   pattern = re.compile(".*_gt.png")

   water_label = int(sys.argv[2])

   path = sys.argv[1]
   testIdx = 0
   trainIdx = 0
   with out_db_train.begin(write=True) as odn_txn:
    with out_db_test.begin(write=True) as ods_txn:
     with label_db_train.begin(write=True) as ldn_txn:
      with label_db_test.begin(write=True) as lds_txn:
        for f in os.listdir(path):
          if (count>totalCount):
            break;
          if (isfile(join(path, f)) and pattern.match(f) != None):
            isTest=((count % skipInterval) == 0)
            try:
              imageSet=getImages(path,f)
              if isTest:
                outRaw(imageSet[1], ods_txn, testIdx)
                outGTLabel(imageSet[0], lds_txn, testIdx, water_label)
                testIdx+=1
              else:
                outRaw(imageSet[1], odn_txn, trainIdx)
                outGTLabel(imageSet[0], ldn_txn, trainIdx, water_label)
                trainIdx+=1
              count += 1
            except IOError:
               print 'skipping ' , f
	       print 'count=',count

   print 'wrap up'
   out_db_train.close()
   label_db_train.close()
   out_db_test.close()
   label_db_test.close()
   sys.exit(0)

def getImages(path, fname):
   indices = np.load(join(path,fname.replace("_gt.png", ".npy")))
   rawimage = Image.open(join(path, fname.replace("_gt.png","_raw.png")))
   rawimage.load()
   return (indices, rawimage)

def testSkipInterval(mpath, percentToTest=0.1):
  c = countFilesInDir(mpath)
  return c, int(float(c) * percentToTest)

def countFilesInDir(mpath):
  pattern = re.compile(".*_gt.png")
  list_dir = os.listdir(mpath)
  count = 0
  for f in list_dir:
    if (isfile(join(mpath, f)) and pattern.match(f) != None):
      count += 1
  return count

def outRaw (im, out_txn, idx):
   import caffe
   import numpy as np
   tmp = np.asarray(im.convert("RGB"),dtype=np.uint8)
   tmp= tmp[:,:,::-1]
   tmp = tmp.transpose((2,0,1))
   im_dat = caffe.io.array_to_datum(tmp)
   out_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())

def outGTLabel (imArray, out_txn, idx, water_label):
   import caffe
   import numpy as np
   imArray = imArray*water_label
   im_dat = caffe.io.array_to_datum(imArray.reshape((1,imArray.shape[0],imArray.shape[1])))
   out_txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())

if __name__=="__main__":
    main()

