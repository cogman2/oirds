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

    if len(sys.argv) < 5:
      print "Usage: ", sys.argv[0], " xlsDir dataDir modelName percentInTestSet [labelIndex]"
      sys.exit( 0 )

    xlsDir = sys.argv[1];
    if xlsDir[-1] != '/':
       xlsDir += "/"

    singleLabel = -1
    if (len(sys.argv) > 5):
      singleLabel = int(sys.argv[5])


    xlsInfos = gt_tool.loadXLSFiles(xlsDir)

    networkDataDir = sys.argv[2]
    if networkDataDir[-1] != '/':
       networkDataDir += "/"

    randUniform = random.seed(23361)
    gt_tool.setIsTest(xlsInfos,float(sys.argv[4]))

    lastList=[]
    lastname=''

    net, transformer = loadNet(networkDataDir,sys.argv[3])
    dumpNetWeights(net)

    txtOut = open('stats.txt','w');
    for i,r in xlsInfos.iterrows():
       if (lastname!= r[2] and len(lastList) > 0):
          if(r[6]==1):
            runName = lastname[0:lastname.index('.tif')]
            initialSize, rawImage = loadImg(runName, xlsDir)
            result = runcaffe(runName, net, transformer, rawImage)
            gtIm, gtIndex = gt_tool.createLabelImage(lastList, initialSize, (result.shape[0], result.shape[1]), singleLabel)
            compareResults(txtOut,runName, result, gtIndex)
          lastList=[]
       else:
          lastList.append(r)
       lastname=r[2]
    txtOut.close()

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


def loadImg(name,dir):
   from PIL import Image
   print name + '-----------'
   imRaw = Image.open(dir+'png/'+name +'.png')
   initialSize = imRaw.size
#  initialSize,im = gt_tool.loadImage(dir + 'png/' + name + '.png')
   return initialSize, convertImage(gt_tool.resizeImg(imRaw))

def loadNet(dataDir,modelName):
   from caffe.proto import caffe_pb2
   blob = caffe_pb2.BlobProto()
   meandata = open(dataDir + 'train_mean.binaryproto', 'rb').read()
   blob.ParseFromString(meandata)
   meanarr = np.array(caffe.io.blobproto_to_array(blob))
   net = caffe.Net(dataDir + 'deploy.prototxt', dataDir + modelName, caffe.TEST)
   transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
   transformer.set_mean('data', meanarr[0,:])
   transformer.set_transpose('data', (2,0,1))
   ## RGB -> BGR ?
   transformer.set_channel_swap('data', (2,1,0))
   transformer.set_raw_scale('data', 255.0)

   interp_layers = [k for k in net.params.keys() if 'up' in k]
   interp_surgery(net, interp_layers)
   return net, transformer

def runcaffe (name, net, transformer, im):
   from caffe.proto import caffe_pb2
   net.blobs['data'].data[...] = transformer.preprocess('data', im)
   return outputResult(net.forward(), transformer, net.blobs['data'].data[0],im, name)

def dumpNetWeights(net):
  for ll in net.blobs:
    try:
      filters = net.params[ll][0].data
      vis_filter(ll, filters)
    except:
      continue

def outputResult(out, transformer, data, rawImage, name):
  classPerPixel = out['prob'][0].argmax(axis=0)
  print 'RANGE ' + str(np.min(out['prob'][0])) + " to " + str(np.max(out['prob'][0]))
  print 'SHAPE ' + str(out['prob'][0].shape)
  print 'HIST ' + str(np.histogram(classPerPixel))
  ima = transformer.deprocess('data', data)
  shapeIME =  (classPerPixel.shape[0],classPerPixel.shape[1])
  # print out['prob'][0].argmax(axis=1)
  # print out['prob'][0].argmax(axis=2)
  plt.subplot(1, 3, 1)
  plt.imshow(rawImage)

  plt.subplot(1, 3, 2)
  plt.imshow(ima)

  plt.subplot(1, 3, 3)
  imArray = toImageArray(classPerPixel);
  plt.imshow(imArray) 

  plt.savefig(name+'_output')
  plt.close()

  return classPerPixel

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
  maxValue = len(gt_tool.label_colors)
  ima = np.zeros((classPerPixel.shape[0], classPerPixel.shape[0], 3), dtype=np.uint8)
  for i in range(0,ima.shape[0]):
    for j in range(0,ima.shape[1]):
        if(classPerPixel[i,j]>0 and classPerPixel[i,j]<maxValue):
          ima[i,j] = gt_tool.label_colors[(classPerPixel[i,j]-1)]
  return ima
  
   
def compareResults(fo,name, result, gt):
  fo.write('STAT ' + name + ' = ' + str(gt_tool.compareResults(result, gt[0])))
  fo.write('\n')

def convertImage (im):
   tmp= np.array(im)
   return tmp
#.transpose(2,0,1)

if __name__=="__main__":
    main()

