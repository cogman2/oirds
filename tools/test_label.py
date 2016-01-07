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
import json_tools

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

    if len(sys.argv) < 2:
      print "Usage: ", sys.argv[0], " config "
      sys.exit( 0 )

    config = json_tools.loadConfig(sys.argv[1])

    xlsInfos = gt_tool.loadXLSFiles(json_tools.getDataDir(config))

    randUniform = random.seed(23361)
    testNames = gt_tool.getTestNames(xlsInfos,json_tools.getPercentageForTest(config))

    lastList=[]
    lastname=''

    imageSize = json_tools.getResize(config);
    net = net_tool.loadNet(config)
    net_tool.dumpNetWeights(net)

    txtOut = open('stats.txt','w');
    for i,r in xlsInfos.iterrows():
       if (lastname!= r[2] and len(lastList) > 0):
          if(lastname in testNames):
             runName = lastname[0:lastname.index('.tif')]
             initialSize, rawImage = loadImg(runName, config)
             result = net_tool.runcaffe(net, rawImage, config)
             classes = outputResult(result[0], result[2], result[1], rawImage, runName, json_tools.getNetworkOutputName(config))
             gtIm, gtIndex = gt_tool.createLabelImageGivenSize(lastList, initialSize, (classes.shape[0], classes.shape[1]), json_tools.getSingleLabel(config))
             compareResults(txtOut,runName, classes, gtIndex)
          lastList=[]
       else:
          lastList.append(r)
       lastname=r[2]
    txtOut.close()

def loadImg(name,config):
   from PIL import Image
   print name + '-----------'
   initialSize, imRaw = gt_tool.loadImage(json_tools.getDataDir(config)+'png/'+name +'.png', config)
   return initialSize, net_tool.convertImage(imRaw)

def outputResult(out, transformer, data, rawImage, name, layerName):
  classPerPixel = out[layerName][0].argmax(axis=0)
  print 'RANGE ' + str(np.min(out[layerName][0])) + " to " + str(np.max(out[layerName][0]))
  print 'SHAPE ' + str(out[layerName][0].shape)
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

