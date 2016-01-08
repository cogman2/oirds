
from PIL import Image

label_colors = [(0,0,0),(228,100,27),(182,228,27),(27,228,140),(27,73,228),(216,13,54),(56,100,6),(155,12,216)]
modes = ['BACKGROUND','VEHICLE/CAR','VEHICLE/PICK-UP','VEHICLE/TRUCK','VEHICLE/UNKNOWN','VEHICLE/VAN','NA','VEHICAL/ANY']
modeIndices = dict( zip( modes, [int(x) for x in range( len(modes) )] ) )
colorIndices = dict( zip( label_colors, [int(x) for x in range( len(label_colors) )] ) )


# 1, 2, 3 = "Image Path", "Image Name", "Target Number"
# 7, 9 = "Intersection Polygon", "Mode of Target Type"

modeIndex=5
polyIndex=4
nameIndex=2
xlsInfoColumns = [1,2,3,7,9]

def loadXLSFiles(parentDataDir):
    import pandas as pd
    import itertools
    import os
    import glob

    if parentDataDir[-1] != '/':
       parentDataDir += "/"

    dataDirs = [parentDataDir+z for z in filter( lambda x: x.startswith( "DataSet_" ), os.listdir( parentDataDir ) )]
    xlsFiles = list( itertools.chain( *[ glob.glob( x + "/*.xls" ) for x in dataDirs ] ) )

    xlsInfo = pd.DataFrame()
    for xlsFile in xlsFiles:
       xlsInfo = xlsInfo.append( pd.read_excel( io=xlsFile, parse_cols=xlsInfoColumns, ignore_index=True ) )

    xlsInfo['isTest'] = 0
    xlsInfo = xlsInfo.reset_index()

    return xlsInfo

def getTestNames(xlsInfo, percent):
    import numpy as np
    testNames= set()
    labelCounts = [0 for i in xrange(len(label_colors))]
    for i,r in xlsInfo.iterrows():  
       labelCounts[modeIndices[r[modeIndex]]] += 1
    for i in xrange(len(labelCounts)):
      labelCounts[i] = int(labelCounts[i]*percent)
    for i in np.random.permutation(len(xlsInfo)):
       if(labelCounts[modeIndices[xlsInfo.iloc[i,modeIndex]]] > 0):
         testNames.add(xlsInfo.iloc[i,nameIndex])
         labelCounts[modeIndices[xlsInfo.iloc[i,modeIndex]]] -= 1
    return testNames

def loadImage(name, config):
   from PIL import Image
   import json_tools
   imRaw = Image.open(name)
   initialSize = imRaw.size
   if (json_tools.isResize(config)):
     imRaw = resizeImg(imRaw, json_tools.getResize(config))
   return initialSize, imRaw

def createLabelImageFor(name, xlsInfo, initialSize, finalSize,singleLabelIndex):
    lastList=[]
    for i,r in xlsInfo.iterrows():
       if (name == r[nameIndex]):
         lastList.append(r)
    return createLabelImage(lastList, initialSize, finalSize, singleLabelIndex)

def resizeImg(im, imageCropSize):
   wpercent = (imageCropSize/float(im.size[0]))
   hsize = int((float(im.size[1])*float(wpercent)))
   return im.resize((imageCropSize ,hsize),Image.ANTIALIAS).crop((0,0,imageCropSize, imageCropSize))

def resizePoly(poly, initialSize, finalSize):
   from shapely.geometry import polygon
   from shapely.ops import transform
   wpercent = (finalSize[0]/float(initialSize[0]))
   hpercent = (finalSize[1]/float(initialSize[1]))
   return transform(lambda x, y, z=None: (x*wpercent,y*hpercent), poly)

def getPolysForImage(xlsInfoList, initialSize, finalSize, singleLabelIndex): 
   from shapely.wkt import dumps, loads
   from shapely.geometry import polygon
   polyList=list()
   for r in xlsInfoList:
     poly = r[polyIndex].replace("[",'(').replace("]","").replace(";",",")
     beg = poly[1:poly.index(',')]
     poly = 'POLYGON (' + poly + ',' + beg + '))'
     polyObj = loads(poly)
     polyObj = resizePoly(polyObj, initialSize, finalSize)
     colorIndex = modeIndices[r[modeIndex]]
     if (singleLabelIndex>=0):
        colorIndex= singleLabelIndex
     polyList.append((polyObj,colorIndex))
   return polyList

def createLabelImageGivenSize(xlsInfoList, initialSize, finalSize, singleLabelIndex):
   polyList  = getPolysForImage(xlsInfoList, initialSize,finalSize, singleLabelIndex)
   return placePolysInImage(polyList,finalSize)

def createLabelImage(xlsInfoList, initialSize, inputImg, singleLabelIndex, augment):
   polyList  = getPolysForImage(xlsInfoList, initialSize, inputImg.size, singleLabelIndex)
   if (augment):
     newPolyList = list()
     newImage = inputImg
     while(len(newPolyList)==0):
        newImage, newPolyList = augmentImage(inputImg,polyList)
     return newImage, placePolysInImage(newPolyList, inputImg.size)
   return inputImg, placePolysInImage(polyList, inputImg.size)

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
  
def movePoly(polyObjTuple, deltaX, deltaY, newPolyList):
    from shapely import affinity
    translatedPoly = affinity.translate(polyObjTuple[0],deltaX, deltaY)
    for newPolyTuple in newPolyList:
      if (newPolyTuple[0].intersects(translatedPoly)):
        return False
    newPolyList.append((translatedPoly, polyObjTuple[1]))
    return True

def placePolysInImage(polyList, finalSize):
  import numpy as np
  imLabel = Image.new("RGB", finalSize, color=(0,0,0))
  indices = np.zeros((1,finalSize[0],finalSize[1]),dtype=np.uint8)
  for polyObjTuple in polyList:
    try:
        placePolyInImage(imLabel, polyObjTuple[0], indices, polyObjTuple[1])
    except:
        continue
  return imLabel, indices

def placePolyInImage(img, poly, indices, colorIndex):
  from shapely.geometry import Point
  width, length = img.size
  bounds = poly.bounds
  for x in (xrange(int(bounds[0]),int(bounds[2]))):
     for y in (xrange(int(bounds[1]),int(bounds[3]))):
       if poly.contains(Point(x, y)):
          img.putpixel((x, y), label_colors[colorIndex])
          indices[0,x,y]=colorIndex

def rotatePoly(polyList, angle, centroid):
    from shapely import affinity
    newPolyList = list()
    for polyTuple in polyList: 
       newPolyList.append((affinity.rotate(polyTuple[0], angle, origin=centroid), polyTuple[1]))
    return polyList

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

def compareResults(result, gt):
  import numpy as np
  fp=0.0
  fn=0.0
  tp=0.0
  tn=0.0
  wrongLabel=0.0 
  gt = np.transpose(gt)
# tn = sum(sum((gt==0)==(result==0)))
# tp = 
  for i in range(0,result.shape[0]):
    for j in range(0,result.shape[1]):
      tn += float(result[i,j] == gt[i,j] and gt[i,j] == 0)
      tp += float(result[i,j] == gt[i,j] and gt[i,j] != 0)
      fp += float(gt[i,j] == 0 and result[i,j] != 0)
      fn += float(gt[i,j] != 0 and result[i,j] == 0)
      wrongLabel += float(result[i,j] != 0 and result[i,j] != gt[i,j] and gt[i,j] != 0)
  if (tp < 0.5):
    precision = 0.0
    recall = 0.0
    f1 = 0.0
  else:
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2.0 * (precision*recall / (precision+recall))
  return (result.shape[0], result.shape[1], fp, fn, tp, tn, wrongLabel, precision, recall,(tp+tn)/(tp+tn+fp+fn),f1)
