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
import re
from os import listdir
from os.path import isfile, join
import os


import compare_tool as ct

def main():
    import os
    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    if len(sys.argv) < 2:
      print "Usage: ", sys.argv[0], " config "
      sys.exit( 0 )
    runTest(sys.argv[1])

def runTest(configFileName):
    import os
    import random
    import sys

    config = json_tools.loadConfig(configFileName)

    randUniform = random.seed(23361)

    dataDir = json_tools.getDataDir(config)

    lastList=[]
    lastname=''

    compareTool = ct.CompareTool()

    net = net_tool.loadNet(config)
    #net_tool.dumpNetWeights(net)

    try:
      os.remove(json_tools.getStatsFileName(config))
    except OSError:
      print 'creating: ' + json_tools.getStatsFileName(config)
      
    txtOut = open(json_tools.getStatsFileName(config),'w');

    pattern = re.compile(".*_gt.png")
    for f in os.listdir(dataDir):
      if (isfile(join(dataDir, f)) and pattern.match(f) != None):
         try:
           imageSet=getImages(dataDir,f, json_tools.getSingleLabel(config))
           rawImage = np.asarray(imageSet[1].convert("RGB"),dtype=np.uint8)
           result = net_tool.runcaffe(net, rawImage, config)
           if (json_tools.dumpBlobs(config)):
              net_tool.dumpNetFilters(net, f)
           classes = outputResult(result[0], result[2], result[1], imageSet[1], f, config)
           compareResults(compareTool,txtOut,f,classes, imageSet[0])
         except IOError:
           print 'skipping ' , f

    dumpTotals(compareTool, txtOut)

    txtOut.close()

def getImages(path, fname, labelId):
   indices = np.load(join(path,fname.replace("_gt.png", ".npy")))
   rawimage = Image.open(join(path, fname.replace("_gt.png","_raw.png")))
   rawimage.load()
   indices = indices * labelId
   return (indices, rawimage)

def outputResult(out, transformer, data, rawImage, name, config):
  layerName = json_tools.getNetworkOutputName(config)
  classPerPixel = out[layerName][0].argmax(axis=0)

  print 'RANGE ' + str(np.min(out[layerName][0])) + " to " + str(np.max(out[layerName][0]))
  print 'SHAPE ' + str(out[layerName][0].shape)
  print 'HIST ' + str(np.histogram(classPerPixel))

  if (json_tools.isOutputImage(config)):
    ima = transformer.deprocess('data', data)
    print 'DIFF IMAGE ' + str(np.min( rawImage - ima))
    shapeIME =  (classPerPixel.shape[0],classPerPixel.shape[1])
    plt.subplot(1, 2, 1)
    plt.imshow(rawImage)

    plt.subplot(1, 2, 2)
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
   
def compareResults(compareTool,fo,name, result, gt):
  stats =  str(compareTool.compareResults(result, gt))[1:-1]
  fo.write(name + ',' + stats)
  fo.write('\n')

def dumpTotals(compareTool,fo):
  stats =  str(compareTool.getTotals())[1:-1]
  fo.write('Totals,' + stats)
  fo.write('\n')

if __name__=="__main__":
    main()

