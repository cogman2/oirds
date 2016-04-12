import json_tools
import numpy as np
import pandas as pd
from PIL import Image
import image_set

def placeCoordPolyInImage(img, dimensions, poly, bbox, indices, colorIndex,color):
   xv = (bbox[2] - bbox[0])/dimensions[0]
   yv = (bbox[3] - bbox[1])/dimensions[1]
   for x in xrange(dimensions[0]):
     for y in xrange(dimensions[1]):
       if poly.contains(Point(bbox[0]+x*xv, bbox[1]+y*yv)):
           img.putpixel((x, y), color)
           indices[0,x,y]=colorIndex

def convertPoly(dimensions, poly, bbox):
   xv = (bbox.bounds[2] - bbox.bounds[0])/dimensions[0]
   yv = (bbox.bounds[3] - bbox.bounds[1])/dimensions[1]
   r = list()
   for p in poly.exterior.coords:
     if (bbox.covers(Point(p[0],p[1]))):
       xd = (x[0] - bbox.bounds[0])/xv
       yd = (y[0] - bbox.bounds[1])/yv
       r.append([xd,yd])
   return r
   
def placePolyInImage(img, poly, indices, colorIndex, color):
    from shapely.geometry import Point
    width, length = img.size
    bounds = poly.bounds
    for x in (xrange(int(bounds[0]),int(bounds[2]))):
       for y in (xrange(int(bounds[1]),int(bounds[3]))):
         if poly.contains(Point(x, y)):
           img.putpixel((x, y), color)
           indices[0,x,y]=colorIndex

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
  
class GTTool:
  """A Class to manage creation of ground truth and image set databases"""

  label_colors = [(0,0,0),(228,100,27),(182,228,27),(27,228,140),(27,73,228),(216,13,54),(56,100,6),(155,12,216)]
  modes =  []
  modeIndices = dict()

  modeIndex=3
  polyIndex=2
  nameIndex=1

  config=dict()
  parentDataDir='.'
  xlsInfo = pd.DataFrame()


  def __init__(self, config):
     self.config = config
     self.modes = json_tools.getModes(config)
     self.modeIndices = dict( zip( self.modes, [int(x) for x in range( len(self.modes) )] ) )

     self.parentDataDir = json_tools.getDataDir(config)
     if self.parentDataDir[-1] != '/':
        self.parentDataDir += "/"
 
  def load(self):
     import json_tools
     import itertools
     import os
     import glob

     xlsInfoColumns = json_tools.getXLSColumns(self.config)

     dataDirs = [self.parentDataDir+z for z in filter( lambda x: x.startswith( "DataSet_" ), os.listdir( self.parentDataDir ) )]
     xlsFiles = list( itertools.chain( *[ glob.glob( x + "/*.xls" ) for x in dataDirs ] ) )

     sxlsInfoColumns = np.copy(xlsInfoColumns)
     sxlsInfoColumns.sort()
     cl = sxlsInfoColumns.tolist();
     self.nameIndex = cl.index( xlsInfoColumns[0])+1
     self.polyIndex = cl.index( xlsInfoColumns[1])+1
     self.modeIndex = cl.index( xlsInfoColumns[2])+1

     for xlsFile in xlsFiles:
        self.xlsInfo = self.xlsInfo.append( pd.read_excel( io=xlsFile, parse_cols=xlsInfoColumns, ignore_index=True ) )

     self.xlsInfo = self.xlsInfo.reset_index()

  def getTestNames(self, percent, testSlice):
     testNames= set()
     labelCounts = [0 for i in xrange(len(self.label_colors))]
     startCounts = [0 for i in xrange(len(self.label_colors))]
     for i,r in self.xlsInfo.iterrows():  
        labelCounts[self.modeIndices[r[self.modeIndex]]+1] += 1
     for i in xrange(len(labelCounts)):
        labelCounts[i] = int(labelCounts[i]*percent)
        if (testSlice != None):
           startCounts[i] = labelCounts[i]*(testSlice-1)
     order = np.random.permutation(len(self.xlsInfo)) if testSlice == None else range(0,len(self.xlsInfo))
     for i in order:
        if(labelCounts[self.modeIndices[self.xlsInfo.iloc[i,self.modeIndex]]+1] > 0):
          if (startCounts[self.modeIndices[self.xlsInfo.iloc[i,self.modeIndex]]+1] >0):
              startCounts[self.modeIndices[self.xlsInfo.iloc[i,self.modeIndex]]+1] -= 1
          else:
             testNames.add(self.xlsInfo.iloc[i,self.nameIndex])
             labelCounts[self.modeIndices[self.xlsInfo.iloc[i,self.modeIndex]]+1] -= 1
     return testNames

  def iterate(self,f):
    sets = dict()
    for i,r in self.xlsInfo.iterrows():  
       name = r[self.nameIndex]
       recs = sets[name] if (sets.has_key(name)) else []
       recs.append(r)
       sets[name] = recs
    for name in np.random.permutation(sets.keys()):
       f(name, sets[name])

  def loadImage(self, name):
    from PIL import Image
    fileName = self.parentDataDir + json_tools.getImageDirectory(self.config) + "/" + name[0:name.rindex('.')] + '.png'
    imRaw = Image.open(fileName)
    if (json_tools.isGray(self.config)):
      imRaw = rgb2gray(imRaw)
    return imRaw

  def getPolysForImage(self, xlsInfoList, finalSize, singleLabelIndex): 
    from shapely.wkt import dumps, loads
    from shapely.geometry import polygon
    polyList=list()
    for r in xlsInfoList:
      poly = r[self.polyIndex].replace("[",'(').replace("]","").replace(";",",")
      try:
        beg = poly[1:poly.index(',')]
        poly = 'POLYGON (' + poly + ',' + beg + '))'
        polyObj = loads(poly)
        colorIndex = self.modeIndices[r[self.modeIndex]]+1
        if (singleLabelIndex>=0):
           colorIndex= singleLabelIndex
        polyList.append((polyObj,colorIndex, self.get_label_color(colorIndex)))
      except ValueError:
        continue 
    return polyList

  def createImageSet(self, xlsInfoList, inputImg, singleLabelIndex):
    polyList  = self.getPolysForImage(xlsInfoList, inputImg.size, singleLabelIndex)
    return image_set.IMSet(inputImg, polyList)

  def get_label_color(self, colorIndex):
     while (len(self.label_colors) <= colorIndex):
        self.label_colors.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))
     return self.label_colors[colorIndex]

