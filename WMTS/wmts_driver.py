from owslib.wms import WebMapService
import io
from io import BytesIO
from urllib import urlencode
from urlparse import urlparse, urlunparse, parse_qs, ParseResult
import numpy as np
from PIL import Image
from functools import partial


def connectToGT(config):
  return WebMapService('http://129.206.228.72/cached/osm', version='1.1.1')

def connectToDG(config):
  return WebMapService("https://evwhs.digitalglobe.com/mapservice/wmsaccess?" + config['connectid'], username=config['uname'], password=config['passwd'], version='1.1.1')

# pullGTImage(wmsConn, (-48.754156,-28.497557,-48.7509017,-28.494662), (256,256))
def pullGTImage(wmsConn,dims, bbx):
  u = wmsConn.getmap(layers=['osm_auto:all'],  srs='EPSG:4326',  bbox=bbx, size=dims,  format='image/png')
  b = u.read()
  stream = BytesIO(b)
  img = Image.open(stream)
  nio = np.asarray(img.convert("RGB"))
  bchannel = np.copy(nio.transpose()[2])
  bchannel[bchannel<251] = 0
  #set label to 1
  bchannel[bchannel>251] = 1  
  bchannel = bchannel.reshape(dims)
  return bchannel, img

def checkGTImage(img_array, dims):
  return float(np.histogram(bchannel,2)[0][1])/float((dims[0]*dims[1])) >= 0.25

# pullRawImage(wmsConn,'Aerial_CIR_Profile', (-48.754156,-28.497557,-48.7509017,-28.494662), (256,256))
def pullRawImage(wmsConn,profile, bbx,dims):
  return wms.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:4326',  bbox=bbx, size=dims,  format='image/png', transparent=True, BGCOLOR='0xFFFFFF', FEATUREPROFILE=profile)

def pullSaveGT(wmsConn, dims, bbx):
   indices, img = pullGTImage(wmsConn, dims, bbx)
   if checkGTImage(indices,dims):
      fname = re.sub(r'[\[\]\(\) ]','',str(bbx).replace(',','_').replace('.','p').replace('-','n'))
      img.save(fname + '_gt.png')
      np.save(fname+'.npy',indices)
      return true
   return false

def pullSaveRaw(wmsConn, profile, dims, bbx):
   img = pullRawImage(wmsConn, profile, dims, bbx)
   fname = re.sub(r'[\[\]\(\) ]','',str(bbx).replace(',','_').replace('.','p').replace('-','n'))
   img.save(fname + '_raw.png')

# eventually we will use this as the curried callback function
def pullAndDoBoth(wmsConnGT, wmsConnRaw, profile, dims, bbx):
  if (pullSaveGT(wmsConnGT, dims, bbx)):
     pullSaveRaw(wmsConnRaw, profile, dims, bbx)

def stupidWalk(polys, func):
  for poly in polys:
     for point in poly.points:
        func((point[0]-0.03, point[1]-0.03, point[0]+0.03, point[1]+0.03))
 
def doit(polys):
  wmsConn=connectToGT(dict())
  dims=(256,256)
  #currying ...for now, just use the GT, eventually we will use pullAndDoBoth
  mycallback = partial(pullSaveGT,wmsConn, dims)
  # to be replaced by smart walk...which will also need zoom level, starting point(possibily), initial direction and distance to move 
  # in pixels..probably put all this in a 'config' object which is a dictionary
  stupidwalk(polys, mycallback)
  
