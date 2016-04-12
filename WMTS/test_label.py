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

    if json_tools.hasImageName(config):
       processImage(dataDir,json_tools.getImageName(config),config,txtOut,net,compareTool)
    else:
      pattern = re.compile(".*_gt.png")
      for f in os.listdir(dataDir):
        if (isfile(join(dataDir, f)) and pattern.match(f) != None):
          processImage(dataDir,f,config,txtOut,net,compareTool)

    dumpTotals(compareTool, txtOut)

    txtOut.close()

def processImage(dataDir,f,config,txtOut,net,compareTool):
    try:
      imageSet=getImages(dataDir,f, json_tools.getSingleLabel(config))
      rawImage = np.asarray(imageSet[1].convert("RGB"),dtype=np.uint8)
      result = net_tool.runcaffe(net, rawImage, config)
      if (json_tools.dumpBlobs(config)):
           net_tool.dumpNetFilters(net, f)
      classes = outputResult(result[0], result[2], result[1], rawImage, f, config)
      compareResults(compareTool,txtOut,f,classes, imageSet[0])
    except IOError:
       print 'skipping ' , f

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
    #print 'DIFF IMAGE ' + str(np.min( rawImage - ima))
    shapeIME =  (classPerPixel.shape[0],classPerPixel.shape[1])
    plt.subplot(1, 2, 1)
    plt.imshow(rawImage)

    plt.subplot(1, 2, 2)
    imArray = overlayImageArray(np.copy(rawImage), classPerPixel);
    plt.imshow(imArray) 

    plt.savefig(name+'_output.png')
    plt.close()

  return classPerPixel

def toImageArray(classPerPixel):
  ima = np.zeros((classPerPixel.shape[0], classPerPixel.shape[0], 3), dtype=np.uint8)
  return overlayImageArray( ima, classPerPixel)

def get_label_color(classPerPixel):
   return (216,13,54)  if classPerPixel == 57 else (15,25,175)
   

def overlayImageArray(ima, classPerPixel):
  for i in range(0,ima.shape[0]):
    for j in range(0,ima.shape[1]):
        if (classPerPixel[i,j] > 0):
          v = convertToShape(get_label_color(classPerPixel[i,j]),ima.shape[2])
          ima[i,j] = v
  return ima

def convertToShape(color, dims):
   if len(color) > dims:
      colorOut = np.copy(color)
      for i in xrange(dims-len(color)):
         colorOut = np.append(colorOut,0)
      return colorOut
   elif len(color) < dims:
      colorOut =color[0:len(color)]
      return colorOut
   return color
   
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

