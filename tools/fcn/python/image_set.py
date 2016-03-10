import json_tools
import numpy as np
import pandas as pd
from PIL import Image


def augmentImage(inputImg, polyList):
    import numpy as np
    newImg = inputImg
    newPolyList = list()
    for polyObjTuple in polyList:
      while True:
        xwidth = polyObjTuple[0].bounds[2] - polyObjTuple[0].bounds[0]
        ywidth = polyObjTuple[0].bounds[3] - polyObjTuple[0].bounds[1]
        randXPixel = np.random.randint(0,inputImg.size[0])+1
        randYPixel = np.random.randint(0,inputImg.size[1])+1
        deltaX = int(randXPixel - polyObjTuple[0].bounds[0])
        deltaY = int(randYPixel - polyObjTuple[0].bounds[1])
        if (deltaX < 0): 
          deltaX = min(deltaX,-xwidth-1)
        else:
          deltaX = max(deltaX,xwidth+1)
   
        if (deltaY < 0): 
          deltaY = min(deltaY,-ywidth-1)
        else:
          deltaY = max(deltaY,ywidth+1)

        if (deltaX + polyObjTuple[0].bounds[0] < 1):
          deltaX = -polyObjTuple[0].bounds[0] + 1;
        if (deltaX + polyObjTuple[0].bounds[0] > inputImg.size[0]-1):
          deltaX = inputImg.size[0]-polyObjTuple[0].bounds[0]-1;

        if (deltaY + polyObjTuple[0].bounds[1] < 1):
          deltaY = -polyObjTuple[0].bounds[1] + 1;
        if (deltaY + polyObjTuple[0].bounds[1] > inputImg.size[1]-1):
          deltaY = inputImg.size[1]-polyObjTuple[0].bounds[1]-1;

        if (movePoly(polyObjTuple, deltaX, deltaY, newPolyList)):
           newImg = moveImageBlock(newImg, polyObjTuple[0], deltaX, deltaY)
           break
    return newImg, newPolyList

def moveImageBlock(image, poly, deltaX, deltaY):
     import numpy as np
     newImage = image.copy()
     newBounds = (int(np.floor(poly.bounds[0])+deltaX), \
                  int(np.floor(poly.bounds[1])+deltaY), \
                  int(np.ceil(poly.bounds[2])+deltaX), \
                  int(np.ceil(poly.bounds[3])+deltaY))
     oldBounds = (int(np.floor(poly.bounds[0])),int(np.floor(poly.bounds[1])),int(np.ceil(poly.bounds[2])),int(np.ceil(poly.bounds[3])))
     partL = image.crop(oldBounds)
     partR = image.crop(newBounds)
     newImage.paste(partR, oldBounds)
     newImage.paste(partL, newBounds)
     return newImage

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


#def cropPoly(polyObjTuple,bbox):
#     from shapely import affinity
#     translatedPoly = affinity.translate(polyObjTuple[0],deltaX, deltaY)
#     for newPolyTuple in newPolyList:
#       if (newPolyTuple[0].intersects(translatedPoly)):
#         return False
#     newPolyList.append((translatedPoly, polyObjTuple[1]))
#     return True


def movePoly(polyObjTuple, deltaX, deltaY, newPolyList):
     from shapely import affinity
     translatedPoly = affinity.translate(polyObjTuple[0],deltaX, deltaY)
     for newPolyTuple in newPolyList:
       if (newPolyTuple[0].intersects(translatedPoly)):
         return False
     newPolyList.append((translatedPoly, polyObjTuple[1], polyObjectTuple[2]))
     return True


def rotatePoly(polyList, angle, centroid):
     from shapely import affinity
     newPolyList = list()
     for polyTuple in polyList: 
        newPolyList.append((affinity.rotate(polyTuple[0], angle, origin=centroid), polyTuple[1],polyTuple[2]))
     return newPolyList

def resizePoly(poly, factor):
  from shapely import affinity
  newPolyList = list()
  for polyTuple in polyList: 
       newPolyList.append((affinity.scale(polyTyple[0],xfact=factor,yfact=factor), polyTuple[1],polyTuple[2]))
  return newPolyList

#def resizePoly(poly, factor):
#   from shapely.geometry import polygon
#   from shapely.ops import transform
#   return transform(lambda x, y, z=None: (x*factor,y*factor), poly)


def rollPoly(polyList, sizes, delta):
      from shapely import affinity
      xsize, ysize = sizes
      xsize = int(delta * xsize)
      newPolyList = list()
      for polyTuple in polyList: 
        bounds = polyTuple[0].bounds
        if (bounds[2] < xsize):
           newPolyList.append((affinity.translate(polyTuple[0],xoff=xsize, yoff=0.0, zoff=0.0), polyTuple[1],polyTuple[2]))
        elif (bounds[0] > xsize):
          newPolyList.append((affinity.translate(polyTuple[0],xoff=-xsize, yoff=0.0, zoff=0.0),polyTuple[1], polyTuple[2]))
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

def resizeImg(im, imageCropSize):
   wpercent = (imageCropSize/float(im.size[0]))
   hsize = int((float(im.size[1])*float(wpercent)))
   return im.resize((imageCropSize ,hsize),Image.ANTIALIAS).crop((0,0,imageCropSize, imageCropSize))


class IMSet:
  """A Class to manage creation of ground truth and image set databases"""
  
  rawImage = None
  polyList = []

  def __init__(self, image, polys):
   self.rawImage = image
   self.polyList = polys

  def getImgShape(self):
    return self.rawImage.size

  def resize(self,imageResize):
    return IMSet(resizeImage(self.rawImage, imageResize),resizePolys(imageResize))

  def placePolysInImage(self):
    import numpy as np
    finalSize= self.rawImage.size
    imLabel = Image.new("RGB", finalSize, color=(0,0,0))
    indices = np.zeros((1,finalSize[0],finalSize[1]),dtype=np.uint8)
    for polyObjTuple in self.polyList:
      try:
          placePolyInImage(imLabel, polyObjTuple[0], indices, polyObjTuple[1], polyObjTuple[2])
      except:
          continue
    return imLabel, indices, centers(self.polyList)

  def imageSetFromCroppedImage(self, imageCropSize, slide ):
    cx = self.rawImage.size[0] / slide
    cy = self.rawImage.size[1] / slide
    result = []
    for xi in xrange(int(cx)):
      for yi in xrange(int(cy)):
       result.append(self.cropImageAt(xi*slide, yi*slide, imageCropSize))
    return result
  
  def cropImageAt(self, cxp, cyp, imageCropSize):
    imageSize = self.rawImage.size
    cxe = max(cxp+imageCropSize, imageSize[0])
    cye = max(cyp+imageCropSize, imageSize[1])
    return IMSet(self.rawImage.crop((cxp, cyp,cxe, cye)), self.cropPolys((cxp, cyp,cxe, cye)))

  def cropPolys(self,bbox):
    from shapely.geometry import polygon
    from shapely.geometry import box
    from shapely.geometry import Point
    b = box(bbox[0],bbox[1],bbox[2],bbox[3])
    newPolyList = list()
    for polyTuple in self.polyList: 
      if (polyTuple[0].within(b)):
        newPolyList.append(polyTuple)
      x = b.intersection(polyTuple[0])
      if (x.area >= (0.75 * polyTuple[0].area)):
        newPolyList.append((x,polyTuple[1],polyTuple[2]))
    return newPolyList


  def resizePolys(self, imgSize):
    return [resizePoly(poly, self.rawImage.size, imgSize) for poly in self.polyList]
