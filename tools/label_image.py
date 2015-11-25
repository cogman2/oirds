#!/usr/bin/env python
# Used to build the training images for a labeled data set.
# each training image is white.  The polygons representing each label
# has an assigned color in the image.

#import h5py, os

label_colors = [(64,128,64),(192,0,128),(0,128,192),(0,128,64),(128,0,0)]

def main():
    import os
    import pandas as pd
    import itertools
    import glob
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

    cols = [1,2,3,7]
    xlsInfo = pd.DataFrame()
    for xlsFile in xlsFiles:
       xlsInfo = xlsInfo.append( pd.read_excel( io=xlsFile, parse_cols=cols, ignore_index=True ) )

    lastList=[]
    lastname=''
    for i,r in xlsInfo.iterrows():
       if (lastname!= r[1] and len(lastList) > 0):
           labelImage, rawImage = convertImg(lastname, lastList, parentDataDir)
           labelImage.save(parentDataDir+"/labels/"+ lastname[0:lastname.index('.tif')] + ".png")
           # need to send to training or test set here!
           lastList=[]
       else:
           lastList.append(r)
       lastname=r[1]

    sys.exit(0)

#    in_db_train = lmdb.open('train', map_size=int(94371840))
#    out_db_train = lmdb.open('groundtruth_train', map_size=int(94371840))

#with h5py.File('train.h5','w') as H:
 #   H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
 #   H.create_dataset( 'y', data=y ) # note the name y given to the dataset!

#    with in_db_train.begin(write=False) as in_txn_train:
 #    with out_db_train.begin(write=True) as out_txn_train:
 #       with open('train.txt', 'r') as train:
 #          for line in train:
 #             outGT(xlsInfo, line, idxNames, out_txn_train)
 #   out_db_train.close()
#   in_db_train.close()

def convertImg(name,xlsInfoList, dir):
  from shapely.wkt import dumps, loads
  from shapely.geometry import polygon
  from PIL import Image
  print name + '-----------'
  imRaw = Image.open(dir + '/png/' + xlsInfoList[0][1][0:xlsInfoList[0][1].index('.tif')] + '.png') 
  imLabel = Image.new("RGB", imRaw.size, "white")
  for r in xlsInfoList:
    poly = r[3].replace("[",'(').replace("]","").replace(";",",")
    beg = poly[1:poly.index(',')]
    poly = 'POLYGON (' + poly + ',' + beg + '))'
    print(poly)
    polyObj = loads(poly)
    labelImage(imLabel, polyObj, label_colors[r[2]])
  return imLabel, imRaw
   
def labelImage(img, poly, color):
  from shapely.geometry import Point
  width, _ = img.size
  for i, px in enumerate(img.getdata()):
     y = i / width
     x = i % width
     if poly.contains(Point(x, y)):
        img.putpixel((x, y), color)

def outGT (xlsInfo, line, namesIdx, out_txn ):
   import caffe
   imageFile = line.split()[0]
   imageLabel = int(line.split()[1])
   imRaw = Image.open(imageFile) # or load whatever ndarray you need
   im = np.array(imRaw)
   # convert to one dimensional ground truth labels
   tmp = np.uint8(np.zeros(im[:,:,0:1].shape))
   tmp[:,:,0] = tmp[:,:,0] + np.prod(np.equal(im,label_colors[imageLabel]),2)

   # - in Channel x Height x Width order (switch from H x W x C)
   tmp = tmp.transpose((2,0,1))
   im_dat = caffe.io.array_to_datum(tmp)
   out_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())

if __name__=="__main__":
    main()
