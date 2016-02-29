#!/usr/bin/env python
#import h5py, os


import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
#%matplotlib inline
import lmdb
import gt_tool
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
      print "Usage: ", sys.argv[0], "image config"
      sys.exit( 0 )

    config = json_tools.loadConfig(sys.argv[2])

    gtTool = gt_tool.GTTool(config)

    net = net_tool.loadNet(config)

    im = loadImgArray(sys.argv[1], gtTool,config);

    print im.shape

    result = net_tool.runcaffe(net, im, config)
    if (json_tools.dumpBlobs(config)):
       net_tool.dumpNetFilters(net, sys.argv[1][0:sys.argv[1].find('.')])

    outputResult(result[0], result[2], result[1], im, config)

def loadImgArray(name, gtTool,config):
   from PIL import Image
   print name + '-----------'
   initialiSize, imRaw = gtTool.loadImage(name)
   return net_tool.convertImage(imRaw,config)

def outputResult(out, transformer, data, rawImage, config):
  layrName = json_tools.getNetworkOutputName(config)
  classPerPixel = out[layrName][0].argmax(axis=0)
  print 'RANGE ' + str(np.min(out[layrName][0])) + " to " + str(np.max(out[layrName][0]))
  print 'SHAPE ' + str(out[layrName][0].shape)
  print 'HIST ' + str(np.histogram(classPerPixel))
  print 'UNIQUE ' + str(np.unique(np.array(classPerPixel).flatten()))
  plt.subplot(1, 2, 1)
  plt.imshow(rawImage)

  plt.subplot(1, 2, 2)
#  imArray = toImageArray(classPerPixel);
#  plt.imshow(imArray)  
  plt.imshow(classPerPixel)
  plt.colorbar()

  plt.savefig('im_output')
  plt.close()

  return classPerPixel

def toImageArray(classPerPixel):
  ima = np.zeros((classPerPixel.shape[0], classPerPixel.shape[1], 3), dtype=np.uint8)
  for i in range(0,ima.shape[0]):
    for j in range(0,ima.shape[1]):
      color = gt_tool.get_label_color(classPerPixel[i,j])
      ima[i,j] = color
  return ima


if __name__=="__main__":
    main()

