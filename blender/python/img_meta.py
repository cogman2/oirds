#!/usr/bin/python

import numpy
import sys
from shapely.geometry import polygon


class ImgMetaData:
   img_dims = (256,256)
   start_rotation = 0
   end_rotation = 0
   azimuth = 0
   offnadir = 0 
   elevation = 0
   zoom_factor = 0
   name = ''
   image_poly = None

   def __init__(self, fname):
    self.name = fname

   def setPoly(self, descriptor):
      from shapely.wkt import dumps, loads
      poly = descriptor.replace("[",'(').replace("]","").replace(")","").replace(";",",")
      beg = poly[1:poly.index(',')]
      poly = 'POLYGON (' + poly + ',' + beg + '))'
      self.image_poly = loads(poly)

# determined by zoom factor
   def getCarScale(self):
      return 1.0

   def getImagePlace(self):
    import random
    from shapely.geometry.point import Point
    b = self.image_poly.bounds
    px = 0
    py = 0
    while(True):
      px = int(b[0] + ((b[2] - b[0])*random.random()))
      py = int(b[1] + ((b[3] - b[1])*random.random()))
      if (self.image_poly.contains(Point(px,py))):
        break;
# Readjust since the blender grid is centered in the middle of the image
    px = px - self.img_dims[0]/2
    py = py - self.img_dims[1]/2
    return (self.start_rotation + ((self.end_rotation - self.start_rotation)*random.random()),(px,py))

class ImgMetaProcess:

  images = {}

  def openFile(self,fileName):
    f = open( fileName )
    lines = f.readlines()
    tokenized_lines = numpy.array([ x.split( "," ) for x in lines ])

    lastfile = ''
    for line in tokenized_lines:
       imgmeta = ImgMetaData(line[0])
       imgmeta.img_dims = (int(line[1]),int(line[2]))
       imgmeta.start_rotation = int(line[3])
       imgmeta.end_rotation = int(line[4])
       imgmeta.setPoly(line[5])
       imgmeta.offnadir = float(line[6])
       imgmeta.azimuth = float(line[7])
       imgmeta.elevation = float(line[8])
       imgmeta.zoom_factor = float(line[9])
       if line[0] != lastfile:
          self.images[line[0]] = [imgmeta]
       else:
          self.images[line[0]].append(imgmeta)
       lastfile = line[0]
