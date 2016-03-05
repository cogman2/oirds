import json_tools
import numpy as np
import pandas as pd
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



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
#        polyObj = resizePoly(polyObj, finalSize, finalSize)
        colorIndex = self.modeIndices[r[self.modeIndex]]+1
        if (singleLabelIndex>=0):
           colorIndex= singleLabelIndex
        polyList.append((polyObj,colorIndex, get_label_colors(colorIndex)))
      except ValueError:
        continue 
    return polyList

  def createLabelImage(self, xlsInfoList, inputImg, singleLabelIndex):
    polyList  = self.getPolysForImage(xlsInfoList, inputImg.size, singleLabelIndex)
    return new IMSet(inputImg, polyList)

  def get_label_color(self, colorIndex):
     import numpy as np
     while (len(self.label_colors) <= colorIndex):
        self.label_colors.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))
     return self.label_colors[colorIndex]

