import sys
import json_tools
import numpy as np
import pandas as pd
from PIL import Image
import image_set
import pudb

#COCO_ROOT='/data2/MS_COCO/'
COCO_ROOT='/opt/py-faster-rcnn/data/coco/'
#OIRDS_ROOT='/home/robertsneddon/oirds/' 
OIRDS_ROOT='/opt/py-faster-rcnn/oirds/' 
TOOLS_ROOT=OIRDS_ROOT+'tools/fcn/python/'                    # Path to python tools for Caffe deep learning
#sys.path.append(COCO_ROOT+'coco/PythonAPI/')
#sys.path.append(COCO_ROOT+'coco/PythonAPI/pycocotools/')
sys.path.append('/opt/py-faster-rcnn/data/coco-master/PythonAPI')
sys.path.append('/opt/py-faster-rcnn/data/coco-master/PythonAPI/pycocotools/')
sys.path.append(TOOLS_ROOT)
A_ROOT=COCO_ROOT+'annotations/'
R_ROOT=COCO_ROOT+'results/'
I_ROOT=COCO_ROOT+'images/'
#A_FILE='instances_val2014.json'
A_FILE='instances_train2014.json'
annType = ['segm','bbox']
annType = annType[1]
dataDir=COCO_ROOT+'images/'
dataType='val2014'
resFile='%s/results/instances_%s_fake%s100_results.json'
resFile=resFile%(dataDir, dataType, annType)
configFile=OIRDS_ROOT+'example_config/create_db.json'
cocoConfigFile=OIRDS_ROOT+'example_config/create_coco_db.json'

sys.path.append(COCO_ROOT+'coco/PythonAPI/')
sys.path.append(COCO_ROOT+'coco/PythonAPI/pycocotools/')
sys.path.append(TOOLS_ROOT)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        my_info = self.xlsInfo
        return my_info


    def loadCoco(self):
        import json_tools
        import itertools
        import os
        import glob
        from pycocotools.coco import COCO
        import pandas as pd
#       self.coco = COCO(A_ROOT + A_FILE)

        self.coco = COCO(A_ROOT + 'instances_train2014.json')
        catIds = self.coco.getCatIds(catNms=self.modes)
        imgIds = self.coco.getImgIds(catIds=catIds)
        img = self.coco.loadImgs(imgIds)
        annIds = self.coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        files = [im['file_name'] for im in img]
        polys = [ann['segmentation'] for ann in anns]
        bboxs = [ann['bbox'] for ann in anns]
        target_ids = [ann['category_id'] for ann in anns]
        index_num = [i for i in xrange(0,len(files))]
# self.modes is a list of desired categories
        target_type = [self.modes for im in img]
        poly_temp = [" " for im in img]

        self.xlsInfo = pd.DataFrame(zip(*[files,polys,target_type,target_ids,bboxs, poly_temp]),columns=[u'Image Name',u'Intersection Polygon',u'Mode of Target Type',u'Targets IDs',u'Bounding Box',u'Poly Temp'])
        self.xlsInfo = self.polyString(self.xlsInfo, 'Bounding Box')
        self.xlsInfo = self.polyString(self.xlsInfo, 'Intersection Polygon')
        self.xlsInfo = self.xlsInfo.reset_index()
        df_info = self.xlsInfo
        return df_info

    
    def polyString(self, df, label):
        if label == 'Bounding Box':
            print "Inside Bounding Box"
            print "length of bbx data is: ", len(df[label])
            for i, bbx in enumerate(df[label]):
                x1 = str(bbx[0])+" "
                y1 = str(bbx[1])
                x2 = str(bbx[0]+bbx[2])+" "
                y2 = str(bbx[1]+bbx[3])
                df.loc[i,(label)] = u'['+x1+y1+'; '+x1+y2+'; '+x2+y2+'; '+x1+y2+']'
            return df

        elif label == 'Intersection Polygon':
            for i, poly in enumerate(df[label]):
                if i >= 30: break
                if type(poly) == dict:
                    try:
                        poly = poly['counts']
                        if type(poly) == list:
                            del poly[0]
                            del poly[len(poly)-1:]
                            poly = [poly]
                    except:
                        continue
                j = 0
                polyStr = ''
                while True:
                    try: 
                        isinstance(poly[j], list)
                        temp_str = str(poly[j])
                        z = iter(temp_str.split(','))
                        tuple_list = [a+' '+b for a,b in zip(*([z]*2))]
                        polyStr = polyStr + ';'.join(tuple_list)
                        j += 1
                    except:
                        break
                polyStr = polyStr.replace('][','],[')
                df.loc[i, ('Poly Temp')] = polyStr 
        return df

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
  
