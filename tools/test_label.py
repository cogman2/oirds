#!/usr/bin/env python
# Used to build the training images for a labeled data set.
# each training image is white.  The polygons representing each label
# has an assigned color in the image.

#import h5py, os

import gt_tool
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
#%matplotlib inline
import lmdb
import caffe
from PIL import Image
import numpy as np

def main():
    from pathlib import Path
    import os
    import random
    import pandas as pd
    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    if len(sys.argv) < 3:
      print "Usage: ", sys.argv[0], " xlsDir dataDir percentInTestSet"
      sys.exit( 0 )


    xlsDir = sys.argv[1];
    if xlsDir[-1] != '/':
       xlsDir += "/"

    xlsInfos = gt_tool.loadXLSFiles(xlsDir)

    parentDataDir = sys.argv[2]
    if parentDataDir[-1] != '/':
       parentDataDir += "/"

    randUniform = random.seed(23361)
    gt_tool.setIsTest(xlsInfos,float(sys.argv[3]))

    lastList=[]
    lastname=''
    for i,r in xlsInfos.iterrows():
       if (lastname!= r[2] and len(lastList) > 0):
          if(r[6]==1):
            runName = lastname[0:lastname.index('.tif')]
            initialSize, rawImage = loadImg(runName, xlsDir)
            result = runcaffe(runName, parentDataDir,rawImage)
            gtIm = convertImage(gt_tool.createLabelImage(lastList, initialSize, (result.shape[0], result.shape[1])))
            compareImages(result, gtIm)
            lastList=[]
       else:
          lastList.append(r)
       lastname=r[2]

def loadImg(name,dir):
  print name + '-----------'
  initialSize,im = gt_tool.loadImage(dir + 'png/' + name + '.png')
  return initialSize, convertImage(im)

def runcaffe (name, dataDir,im):
   from caffe.proto import caffe_pb2
   blob = caffe_pb2.BlobProto()
   meandata = open(dataDir + 'train_mean.binaryproto', 'rb').read()
   blob.ParseFromString(meandata)
   meanarr = np.array(caffe.io.blobproto_to_array(blob))
   net = caffe.Net(dataDir + 'deploy.prototxt', dataDir + 'train_iter_8000.caffemodel', caffe.TEST)
   transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
   transformer.set_mean('data', meanarr[0,:])
   transformer.set_transpose('data', (2,0,1))
   ## RGB -> BGR ?
   #transformer.set_channel_swap('data', (2,1,0))
   transformer.set_raw_scale('data', 255.0)
   #net.blobs['data'].reshape(1,3,imageSizeCrop,imageSizeCrop)
   #out= forward_all(data=np.asarray([transformer.preprocess('data', im)]))
   net.blobs['data'].data[...] = transformer.preprocess('data', im)
   dumpNetWeights(name, net)
   return outputResult(net.forward(), transformer, net.blobs['data'].data[0], name)

def dumpNetWeights(name, net):
  for ll in net.blobs:
    try:
      filters = net.params[ll][0].data
      vis_filter(name+'_'+ll, filters)
    except:
      continue

def outputResult(out, transformer, data, name):
  classPerPixel = out['prob'][0].argmax(axis=0)
  print 'HIST ' + str(np.histogram(classPerPixel))
  ima = transformer.deprocess('data', data)
  shapeIME =  (classPerPixel.shape[0],classPerPixel.shape[1])
  # print out['prob'][0].argmax(axis=1)
  # print out['prob'][0].argmax(axis=2)
  plt.subplot(1, 2, 1)
  plt.imshow(ima)

  plt.subplot(1, 2, 2)
  imArray = toImageArray(classPerPixel);
  plt.imshow(imArray) 

 # plt.subplot(1, 3, 3)
 # plt.imshow(out('score')[0]) 
  plt.savefig(name+'_output')
  plt.close()

  return imArray

def vis_filter(name, data, padsize=1, padval=0):
  wc = data.shape[0]/64;
  for pltNum in range(0,wc):
    vis_square(name + str(pltNum), data[pltNum*data.shape[0]:(pltNum+1)*data.shape[0],:], padsize,padval)
# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)   
def vis_square(name, data, padsize=1, padval=0):
    data = data[:,0:3,:,:]
    data = data.transpose(0, 2, 3, 1)
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data);
    plt.savefig(name + 'plt')
    plt.close()

def toImageArray(classPerPixel):
  maxValue = len(gt_tool.label_colors)+1
  ima = np.zeros((classPerPixel.shape[0], classPerPixel.shape[0], 3), dtype=np.uint8)
  for i in range(0,ima.shape[0]-1):
    for j in range(0,ima.shape[1]-1):
        ima[i,j] = gt_tool.label_colors[classPerPixel[i,j]%maxValue]
  return ima
  
   

def compareImages(im, gtIm):
  print 'STAT ' + str(gt_tool.compareImages(im,gtIm))   

def convertImage (im):
   tmp= np.array(im)
   return tmp
#.transpose(2,0,1)

if __name__=="__main__":
    main()

