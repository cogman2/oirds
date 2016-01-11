#!/usr/bin/env python
# Used to build the training images for a labeled data set.
# each training image is white.  The polygons representing each label
# has an assigned color in the image.

#import h5py, os


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
import gt_tool

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

    gtTool = gt_tool.GTTool(config)
    gtTool.load()


    randUniform = random.seed(23361)
    if (json_tools.hasImageName(config)):
      testNames = set()
      testNames.add(json_tools.getImageName(config))
    else:
      testNames = gtTool.getTestNames(json_tools.getPercentageForTest(config))

    lastList=[]
    lastname=''

    net = net_tool.loadNet(config)
    #net_tool.dumpNetWeights(net)

    txtOut = open('stats.txt','w');

    def f(lastname, lastList):
          if(lastname in testNames):
             runName = lastname[0:lastname.rindex('.')]
             print runName + '-----------'
             initialSize, rawImage = gtTool.loadImage(lastname)
             rawImage = net_tool.convertImage(rawImage,config)
             result = net_tool.runcaffe(net, rawImage, config)
             if (json_tools.dumpBlobs(config)):
               net_tool.dumpNetFilters(net, runName)
             classes = outputResult(gtTool, result[0], result[2], result[1], rawImage, runName, json_tools.getNetworkOutputName(config))
             gtIm, gtIndex = gtTool.createLabelImageGivenSize(lastList, initialSize, (classes.shape[0], classes.shape[1]), json_tools.getSingleLabel(config))
             compareResults(txtOut,runName, classes, gtIndex)

    gtTool.iterate(f)
    txtOut.close()

def outputResult(gtTool, out, transformer, data, rawImage, name, layerName):
  classPerPixel = out[layerName][0].argmax(axis=0)
  print 'RANGE ' + str(np.min(out[layerName][0])) + " to " + str(np.max(out[layerName][0]))
  print 'SHAPE ' + str(out[layerName][0].shape)
  print 'HIST ' + str(np.histogram(classPerPixel))

  ima = transformer.deprocess('data', data)
  print 'DIFF IMAGE ' + str(np.min( rawImage - ima))
  shapeIME =  (classPerPixel.shape[0],classPerPixel.shape[1])
  plt.subplot(1, 2, 1)
  plt.imshow(rawImage)

  plt.subplot(1, 2, 2)
#  imArray = toImageArray(classPerPixel);
  imArray = overlayImageArray(gtTool, np.copy(rawImage), classPerPixel);
  plt.imshow(imArray) 

  plt.savefig(name+'_output')
  plt.close()

  return classPerPixel

def toImageArray(gtTool,classPerPixel):
  ima = np.zeros((classPerPixel.shape[0], classPerPixel.shape[0], 3), dtype=np.uint8)
  return overlayImageArray(gtTool, ima, classPerPixel)

def overlayImageArray(gtTool, ima, classPerPixel):
  for i in range(0,ima.shape[0]):
    for j in range(0,ima.shape[1]):
        ima[i,j] = gtTool.get_label_color(classPerPixel[i,j])
  return ima
   
def compareResults(fo,name, result, gt):
  stats =  str(gt_tool.compareResults(result, gt[0]))[1:-1]
  fo.write(name + ',' + stats)
  fo.write('\n')

if __name__=="__main__":
    main()

