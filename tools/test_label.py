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
import net_tool

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
    testNames = gt_tool.getTestNames(xlsInfos,float(sys.argv[4]))

    lastList=[]
    lastname=''

    net, transformer = net_tool.loadNet(networkDataDir, sys.argv[3], gt_tool.imageCropSize)
    net_tool.dumpNetWeights(net)

    txtOut = open('stats.txt','w');
    for i,r in xlsInfos.iterrows():
       if (lastname!= r[2] and len(lastList) > 0):
          if(lastname in testNames):
             runName = lastname[0:lastname.index('.tif')]
             initialSize, rawImage = loadImg(runName, xlsDir)
             result = net_tool.runcaffe(net, transformer, rawImage)
             classes = outputResult(result[0], transformer, result[1], rawImage, runName)
             gtIm, gtIndex = gt_tool.createLabelImageGivenSize(lastList, initialSize, (classes.shape[0], classes.shape[1]), singleLabel)
             compareResults(txtOut,runName, classes, gtIndex)
          lastList=[]
       else:
          lastList.append(r)
       lastname=r[2]
    txtOut.close()

def loadImg(name,dir):
   from PIL import Image
   print name + '-----------'
   imRaw = Image.open(dir+'png/'+name +'.png')
   initialSize = imRaw.size
   return initialSize, net_tool.convertImage(gt_tool.resizeImg(imRaw))

def outputResult(out, transformer, data, rawImage, name):
  layrName = 'score'
  classPerPixel = out[layrName][0].argmax(axis=0)
  print 'RANGE ' + str(np.min(out[layrName][0])) + " to " + str(np.max(out[layrName][0]))
  print 'SHAPE ' + str(out[layrName][0].shape)
  print 'HIST ' + str(np.histogram(classPerPixel))
#  print 'RANGED ' + str(np.min(data)) + " to " + str(np.max(data))
#  print 'SHAPED ' + str(data.shape)

  ima = transformer.deprocess('data', data)
  print 'DIFF IMAGE ' + str(np.min( rawImage - ima))
  shapeIME =  (classPerPixel.shape[0],classPerPixel.shape[1])
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

def toImageArray(classPerPixel):
  maxValue = len(gt_tool.label_colors)
  ima = np.zeros((classPerPixel.shape[0], classPerPixel.shape[0], 3), dtype=np.uint8)
  for i in range(0,ima.shape[0]):
    for j in range(0,ima.shape[1]):
        if(classPerPixel[i,j]>0 and classPerPixel[i,j]<maxValue):
          ima[i,j] = gt_tool.label_colors[classPerPixel[i,j]]
  return ima
   
def compareResults(fo,name, result, gt):
  fo.write('STAT ' + name + ' = ' + str(gt_tool.compareResults(result, gt[0])))
  fo.write('\n')

if __name__=="__main__":
    main()

