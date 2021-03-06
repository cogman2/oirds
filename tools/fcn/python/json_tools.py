import json
import caffe

def getAttribute(data, name):
   return data[name]

def getStatsFileName(data):
   return data['statsFileName'] if data.has_key('statsFileName') else 'stats.txt'

def getGpuID(data):
   return data['gpuID'] if data.has_key('gpuID') else 0

def hasTestSlice(data):
   return data.has_key('testSlice')

def getTestSlice(data):
   return data['testSlice']

def getXLSColumns(data):
   return data['xlsInfoColumns']

def getModes(data):
   return data['modes'];

def isGray(data):
   return data.has_key('isGray') and data['isGray']

def getImageDirectory(data):
   return data['imageDirectory'] if data.has_key('imageDirectory') else 'png'

def useTransformer(data):
   return data.has_key('useTransformer') and data['useTransformer']

def useCaffeImage(data):
   return data.has_key('useCaffeImage') and data['useCaffeImage']

def getNetworkOutputName(data):
  return data['networkOutputName']

def loadConfig(fileName):
  with open(fileName) as data_file:    
    data = json.load(data_file)
  return data

def getModelFileName(data):
   return str(data['modelFileName'])


def getDataDir(data):
   networkDataDir= data['dataDirectory']
   if networkDataDir[-1] != '/':
      networkDataDir += "/"
   return networkDataDir

def isOutputImage(data):
   return data.has_key('outputImage') and data['outputImage']

def dumpBlobs(data):
   return data.has_key('dumpBlobs') and data['dumpBlobs']

def getProtoTxt(data):
   return str(data['prototxt'])

def isNetSurgery(data):
   return data.has_key('netsurgery') and data['netsurgery']

def isGPU(data):
   return data.has_key('gpu') and data['gpu']

def getPercentageForTest(data):
   return data['percentageToTest']

def isResize(data):
   return data.has_key('resize')

def rotate(data):
   return data.has_key('rotate') and data['rotate']

def colorAugment(data):
   return data.has_key('colorAugment') and data['colorAugment']

def getResize(data):
   return int(data['resize'])

def isCrop(data):
   return data.has_key('cropSize')

def getSlide(data):
   return data['slide'] if data.has_key('slide') else getCropSize(data)

def getCropSize(data):
   return data['cropSize'] if data.has_key('cropSize') else 0

def isSingleLabel(data):
   return data.has_key('labelIndex')

def getSingleLabel(data):
   if(isSingleLabel(data)):
     return int(data['labelIndex'])
   return -1;

def hasImageName(data):
    return data.has_key('imageName')

def getImageName(data):
   return data['imageName']

def getMeanArray(data):
   from caffe.proto import caffe_pb2
   import numpy as np
   if (data.has_key('meanarray')):
     return np.array(data['meanarray'])
   elif (data.has_key('meanproto')):
     blob = caffe_pb2.BlobProto()
     meandata = open( data['meanproto'], 'rb').read()
     blob.ParseFromString(meandata)
     return np.array(caffe.io.blobproto_to_array(blob))[0,:]
   else:
     return [0.0, 0.0,0.0]
