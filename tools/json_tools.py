import json
import caffe

def getAttribute(data, name):
   return data[name]

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

def getMultFactor(data):
   return data['multiplicationFactor']

def isResize(data):
   return data.has_key('resize')

def getResize(data):
   import numpy as np
   return int(data['resize'])

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
