#!/usr/bin/env python
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

    if len(sys.argv) < 3:
      print "Usage: ", sys.argv[0], "image dataDir modelName "
      sys.exit( 0 )

    networkDataDir = sys.argv[2]
    if networkDataDir[-1] != '/':
       networkDataDir += "/"

    net, transformer = net_tool.loadNet(networkDataDir,sys.argv[3], 128)

    im = loadImg(sys.argv[1]);
    result = net_tool.runcaffe(net, transformer, im)
    outputResult(result[0], transformer, result[1], im)

def loadImg(name):
   from PIL import Image
   print name + '-----------'
   imRaw = Image.open(name)
   return net_tool.convertImage(gt_tool.resizeImg(imRaw))

def outputResult(out, transformer, data, rawImage):
  layrName = 'score'
  classPerPixel = out[layrName][0].argmax(axis=0)
  print 'RANGE ' + str(np.min(out[layrName][0])) + " to " + str(np.max(out[layrName][0]))
  print 'SHAPE ' + str(out[layrName][0].shape)
  print 'HIST ' + str(np.histogram(classPerPixel))
  print 'UNIQUE ' + str(np.unique(np.array(classPerPixel).flatten()))
  plt.subplot(1, 2, 1)
  plt.imshow(rawImage)

  plt.subplot(1, 2, 2)
  imArray = toImageArray(classPerPixel);
  plt.imshow(imArray) 

  plt.savefig('im_output')
  plt.close()

  return classPerPixel

def toImageArray(classPerPixel):
  maxValue = len(gt_tool.label_colors)
  colors = dict()
  ima = np.zeros((classPerPixel.shape[0], classPerPixel.shape[0], 3), dtype=np.uint8)
  for i in range(0,ima.shape[0]):
    for j in range(0,ima.shape[1]):
      color = (0,0,0)
      if colors.has_key(classPerPixel[i,j]):
        color = colors[classPerPixel[i,j]]
      else:
        color  = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        colors[classPerPixel[i,j]] = color
      ima[i,j] = color
  return ima


if __name__=="__main__":
    main()

