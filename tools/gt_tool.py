
from PIL import Image

label_colors = [(0,0,0),(228,100,27),(182,228,27),(27,228,140),(27,73,228),(216,13,54),(56,100,6),(155,12,216)]
modes = ['BACKGROUND','VEHICLE/CAR','VEHICLE/PICK-UP','VEHICLE/TRUCK','VEHICLE/UNKNOWN','VEHICLE/VAN','NA','VEHICAL/ANY']
modeIndices = dict( zip( modes, [int(x) for x in range( len(modes) )] ) )
colorIndices = dict( zip( label_colors, [int(x) for x in range( len(label_colors) )] ) )
imageCropSize=128


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

def setIsTest(xlsInfo, percent):
    import numpy as np
    labelCounts = [0 for i in xrange(len(label_colors))]
    for i,r in xlsInfo.iterrows():  
       labelCounts[modeIndices[r[modeIndex]]] += 1
    for i in xrange(len(labelCounts)):
      labelCounts[i] = int(labelCounts[i]*percent)
    for i in np.random.permutation(len(xlsInfo)):
       if(labelCounts[modeIndices[xlsInfo.iloc[i,modeIndex]]] > 0):
         xlsInfo.loc[i,'isTest']=1
         labelCounts[modeIndices[xlsInfo.iloc[i,modeIndex]]] -= 1

def loadImage(name):
   from PIL import Image
   imRaw = Image.open(name)
   initialSize = imRaw.size
   imRaw = resizeImg(imRaw)
   return initialSize, imRaw

def createLabelImageFor(name, xlsInfo, initialSize, finalSize,singleLabelIndex):
    lastList=[]
    for i,r in xlsInfo.iterrows():
       if (name == r[nameIndex]):
         lastList.append(r)
    return createLabelImage(lastList, initialSize, finalSize, singleLabelIndex)

def resizeImg(im):
   wpercent = (imageCropSize/float(im.size[0]))
   hsize = int((float(im.size[1])*float(wpercent)))
   return im.resize((imageCropSize ,hsize),Image.ANTIALIAS).crop((0,0,imageCropSize, imageCropSize))

def resizePoly(poly, initialSize, finalSize):
   from shapely.geometry import polygon
   from shapely.ops import transform
   wpercent = (finalSize[0]/float(initialSize[0]))
   hpercent = (finalSize[1]/float(initialSize[1]))
   return transform(lambda x, y, z=None: (x*wpercent,y*hpercent), poly)

def createLabelImage(xlsInfoList, initialSize, finalSize, singleLabelIndex):
  from shapely.wkt import dumps, loads
  from shapely.geometry import polygon
  from PIL import Image
  import numpy as np
  imLabel = Image.new("RGB", finalSize, color=(0,0,0))
  indices = np.zeros((1,finalSize[0],finalSize[1]),dtype=np.uint8)
  for r in xlsInfoList:
    poly = r[polyIndex].replace("[",'(').replace("]","").replace(";",",")
    beg = poly[1:poly.index(',')]
    poly = 'POLYGON (' + poly + ',' + beg + '))'
    polyObj = loads(poly)
    polyObj = resizePoly(polyObj, initialSize, finalSize)
    conlorIndex = modeIndices[r[modeIndex]]
    if (singleLabelIndex>=0):
      colorIndex= singleLabelIndex
    try:
        placePolyInImage(imLabel, polyObj, indices, colorIndex)
#hacked_color) 
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

def compareResults(result, gt):
  fp=0.0
  fn=0.0
  tp=0.0
  tn=0.0
  wrongLabel=0.0
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
