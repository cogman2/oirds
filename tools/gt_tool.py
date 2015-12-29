
from PIL import Image

label_colors = [(64,128,5),(192,0,1),(0,128,2),(0,128,3),(128,0,4),(56,100,6),(64.49,7)]
hacked_color = (64,49,7)

modes = ['VEHICLE/CAR','VEHICLE/PICK-UP','VEHICLE/TRUCK','VEHICLE/UNKNOWN','VEHICLE/VAN']
modeIndices = dict( zip( modes, [int(x) for x in range( len(modes) )] ) )
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

def createLabelImageFor(name, xlsInfo, initialSize, finalSize):
    lastList=[]
    for i,r in xlsInfo.iterrows():
       if (name == r[nameIndex]):
         lastList.append(r)
    return createLabelImage(lastList, initialSize, finalSize)

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

def createLabelImage(xlsInfoList, initialSize, finalSize):
  from shapely.wkt import dumps, loads
  from shapely.geometry import polygon
  from PIL import Image
  imLabel = Image.new("RGB", finalSize, color=(0,0,0))
  for r in xlsInfoList:
    poly = r[polyIndex].replace("[",'(').replace("]","").replace(";",",")
    beg = poly[1:poly.index(',')]
    poly = 'POLYGON (' + poly + ',' + beg + '))'
    polyObj = loads(poly)
    polyObj = resizePoly(polyObj, initialSize, finalSize)
    try:
        placePolyInImage(imLabel, polyObj, label_colors[modeIndices[r[modeIndex]]])
#hacked_color) 
    except:
        continue
  return imLabel

def placePolyInImage(img, poly, color):
  from shapely.geometry import Point
  width, length = img.size
  bounds = poly.bounds
  for x in (xrange(int(bounds[0]),int(bounds[2]))):
     for y in (xrange(int(bounds[1]),int(bounds[3]))):
       if poly.contains(Point(x, y)):
          img.putpixel((x, y), color)


def compareImages(im, gtIm):
  fp=0.0
  fn=0.0
  tp=0.0
  tn=0.0
  wrongLabel=0.0
  for i in range(0,im.shape[0]):
    for j in range(0,im.shape[1]):
      tn += float(all(im[i,j] == gtIm[i,j]) and all(gtIm[i,j] == [0,0,0]))
      tp += float(all(im[i,j] == gtIm[i,j]) and any(gtIm[i,j] != [0,0,0]))
      fp += float(all(gtIm[i,j] == [0,0,0]) and any(im[i,j] != [0,0,0]))
      fn += float(any(gtIm[i,j] != [0,0,0]) and all(im[i,j] == [0,0,0]))
      wrongLabel += float(any(im[i,j] != [0,0,0]) and any(im[i,j] != gtIm[i,j]) and any(gtIm[i,j] != [0,0,0]))
  if (tp < 0.5):
    precision = 0.0
    recall = 0.0
    f1 = 0.0
  else:
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2.0 * (precision*recall / (precision+recall))
  return (im.shape[0], im.shape[1], fp, fn, tp, tn, wrongLabel, precision, recall,(tp+tn)/(tp+tn+fp+fn),f1)
