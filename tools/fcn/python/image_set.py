import json_tools
import numpy as np
import pandas as pd
from PIL import Image

def placePolyInImage(img, poly, indices, colorIndex, color):
    from shapely.geometry import Point
    width, length = img.size
    bounds = poly.bounds
    for x in (xrange(int(bounds[0]),int(bounds[2]))):
       for y in (xrange(int(bounds[1]),int(bounds[3]))):
         if poly.contains(Point(x, y)):
           img.putpixel((x, y), color)
           indices[0,x,y]=colorIndex

def resizePoly(poly, initialSize, finalSize):
   from shapely.geometry import polygon
   from shapely.ops import transform
   wpercent = (finalSize[0]/float(initialSize[0]))
   hpercent = (finalSize[1]/float(initialSize[1]))
   return transform(lambda x, y, z=None: (x*wpercent,y*hpercent), poly)

def centers(polyList):
   return [((polyObjTuple[0].bounds[2] - polyObjTuple[0].bounds[0])/2, (polyObjTuple[0].bounds[3] - polyObjTuple[0].bounds[1])/2) for polyObjTuple in polyList]


def cropPoly(polyObjTuple,bbox):
     from shapely import affinity
     translatedPoly = affinity.translate(polyObjTuple[0],deltaX, deltaY)
     for newPolyTuple in newPolyList:
       if (newPolyTuple[0].intersects(translatedPoly)):
         return False
     newPolyList.append((translatedPoly, polyObjTuple[1]))
     return True


def movePoly(polyObjTuple, deltaX, deltaY, newPolyList):
     from shapely import affinity
     translatedPoly = affinity.translate(polyObjTuple[0],deltaX, deltaY)
     for newPolyTuple in newPolyList:
       if (newPolyTuple[0].intersects(translatedPoly)):
         return False
     newPolyList.append((translatedPoly, polyObjTuple[1]))
     return True


def rotatePoly(polyList, angle, centroid):
     from shapely import affinity
     newPolyList = list()
     for polyTuple in polyList: 
        newPolyList.append((affinity.rotate(polyTuple[0], angle, origin=centroid), polyTuple[1]))
     return newPolyList

def rollPoly(polyList, sizes, delta):
      from shapely import affinity
      xsize, ysize = sizes
      xsize = int(delta * xsize)
      newPolyList = list()
      for polyTuple in polyList: 
        bounds = polyTuple[0].bounds
        if (bounds[2] < xsize):
           newPolyList.append((affinity.translate(polyTuple[0],xoff=xsize, yoff=0.0, zoff=0.0), polyTuple[1]))
        elif (bounds[0] > xsize):
          newPolyList.append((affinity.translate(polyTuple[0],xoff=-xsize, yoff=0.0, zoff=0.0),polyTuple[1]))
        else:
          return list()
      return newPolyList

def rollImage(image, delta):
     "Roll an image sideways"
     xsize, ysize = image.size
     splitX = int(xsize * delta)
     partL = image.crop((0, 0, splitX, ysize))
     partR = image.crop((splitX, 0, xsize, ysize))
     image.paste(partR, (0, 0, xsize-splitX, ysize))
     image.paste(partL, (xsize-splitX, 0, xsize, ysize))
     return image


class IMSet:
  """A Class to manage creation of ground truth and image set databases"""
  
  rawImage = null
  polyList = []

  def __init__(self, image, polys):
   rawImage = img
   polyList = polys

  def placePolysInImage(self, finalSize):
    import numpy as np
    imLabel = Image.new("RGB", finalSize, color=(0,0,0))
    indices = np.zeros((1,finalSize[0],finalSize[1]),dtype=np.uint8)
    for polyObjTuple in polyList:
      try:
          placePolyInImage(imLabel, polyObjTuple[0], indices, polyObjTuple[1], self.get_label_color(polyObjTuple[1]))
      except:
          continue
    return imLabel, indices, centers(polyList)

  def imageSetFromCroppedImage(self,imageCropSize, slide ):
    cx = rawImage.size[0] / slide
    cy = rawImage.size[1] / slide
    result = []
    for xi in xrange(int(cx)):
      for yi in xrange(int(cy)):
       result.append(cropImageAt(cx*slide, cxy*slide, imageCropSize))
    return result
  
  def cdropImageAt(self, cxp, cxy, imageCropSize):
    cxe = max(cxp+imageCropSize, imageSize[0])
    cye = max(cxy+imageCropSize, imageSize[1])
    return new IMSet(rawImage.crop((cxp, cyp,cxe, cye)), cropPolys((cxp, cyp,cxe, cye))

  def cropPolys(self, bbox):
    return [cropPoly(poly,bbox) for poly in polyList]

