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

    if len(sys.argv) < 2:
      print "Usage: ", sys.argv[0], " dataDir"
      sys.exit( 0 )


    if (os.path.isdir("./png_gt")):
      shutil.rmtree("./png_gt")
    if (os.path.isdir("./png_raw")):
      shutil.rmtree("./png_raw")
    if (os.path.isdir("./raw_train")):
      shutil.rmtree("./raw_train")
    if (os.path.isdir("./raw_ttest")):
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

    gt_tool.setIsTest(xlsInfo,0.18)

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
           if (rawImage.size[0] < gt_tool.imageCropSize or rawImage.size[1] < gt_tool.imageCropSize):
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

def convertImg(name,xlsInfoList, dir):
  print name + '-----------'
  initialSize, imRaw= gt_tool.loadImage(dir + 'png/' + name[0:name.index('.tif')] + '.png')
  imLabel = gt_tool.createLabelImage(xlsInfoList, initialSize, imRaw.size)
  return imLabel,imRaw
   
def outGT (im, out_txn, idx):
   import caffe
   import numpy as np
   tmp = np.array(im)
   tmp= tmp[:,:,::-1]
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

