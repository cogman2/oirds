#!/usr/bin/python

import numpy
import sys
from shapely.geometry import polygon


class ImgMetaData:
   start_rotation = 0
   end_rotation = 0
   azimuth = 0
   offnadir = 0 
   elevation = 0
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
    return (self.start_rotation + ((self.end_rotation - self.start_rotation)*random.random()),(px,py))

class ImgMetaProcess:

  images = {}

  def openFile(self,fileName):
    f = open( fileName )
    lines = f.readlines()
    tokenized_lines = numpy.array([ x.split( "," ) for x in lines ])

    lastfile = ''
    mask = numpy.zeros([256,256],dtype=numpy.uint8)
    for line in tokenized_lines:
       imgmeta = ImgMetaData(line[0])
       imgmeta.start_rotation = int(line[1])
       imgmeta.end_rotation = int(line[2])
       imgmeta.setPoly(line[3])
       imgmeta.offnadir = float(line[4])
       imgmeta.azimuth = float(line[5])
       imgmeta.elevation = float(line[6])
       if line[0] != lastfile:
          self.images[line[0]] = [imgmeta]
       else:
          self.images[line[0]].append(imgmeta)
       lastfile = line[0]
