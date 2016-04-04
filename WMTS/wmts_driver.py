from owslib.wms import WebMapService
import io
from io import BytesIO
from urllib import urlencode
from urlparse import urlparse, urlunparse, parse_qs, ParseResult
import numpy as np
from PIL import Image
from functools import partial
import shape_walker
import re
from owslib.wmts import WebMapTileService

def connectToGT(config):
  return WebMapService('http://129.206.228.72/cached/osm', version='1.1.1')

def connectToDG(config):
  return WebMapService("https://evwhs.digitalglobe.com/mapservice/wmsaccess?connectid=" + config['connectid'], username=config['uname'], password=config['passwd'], version='1.1.1')

def connectToGTTile(config):
  return WebMapTileService('http://129.206.228.72/cached/osm', version='1.1.1')

def connectToDGTile(config):
  return WebMapTileService("https://evwhs.digitalglobe.com/earthservice/wmtsaccess?connectid=" + config['connectid'], username=config['uname'], password=config['passwd'], version='1.1.1')

def toImageFile(b):
  stream = BytesIO(b)
  img = Image.open(stream)
  nio = np.asarray(img.convert("RGB"))
  bchannel = np.copy(nio.transpose()[2])
  bchannel[bchannel<251] = 0
  #set label to 1
  bchannel[bchannel>251] = 1  
  bchannel = bchannel.reshape(dims)
  return bchannel, img

# pullGTImage(wmsConn, (-48.754156,-28.497557,-48.7509017,-28.494662), (256,256))
def pullGTImage(wmsConn,dims, bbx):
  u = wmsConn.getmap(layers=['osm_auto:all'],  srs='EPSG:4326',  bbox=bbx, size=dims,  format='image/png')
  return toImageFile(u.read())

def pullGTImageTile(wmsConn,dims, tile):
  u = wmsConn.gettile(layer='DigitalGlobe:ImageryTileService', tilematrixset='EPSG:4326', tilematrix='EPSG:4326:17', row=47785, column=47785, format="image/png")
  return toImageFile(u.read())

def checkGTImage(img_array, dims):
  return float(np.histogram(img_array,2)[0][1])/float((dims[0]*dims[1])) >= 0.25

# pullRawImage(wmsConn,'Aerial_CIR_Profile', (-48.754156,-28.497557,-48.7509017,-28.494662), (256,256))
def pullRawImage(wmsConn,profile,dims, bbx):
  print bbx
  return wmsConn.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:4326',  bbox=bbx, size=dims,  format='image/png', transparent=True, BGCOLOR='0xFFFFFF', FEATUREPROFILE=profile)

def pullRawImageTile(wmsConn,profile,dims, tile):
  print tile
  return wmsConn.gettile(layer='DigitalGlobe:ImageryTileService', tilematrixset='EPSG:4326', tilematrix='EPSG:4326:17', row=tile[1], column=tile[0], format="image/png")

def pullSaveGT(wmsConn, dims, bbx):
   indices, img = pullGTImage(wmsConn, dims, bbx)
   if checkGTImage(indices,dims):
      fname = re.sub(r'[\[\]\(\) ]','',str(bbx).replace(',','_').replace('.','p').replace('-','n'))
      img.save(fname + '_gt.png')
      np.save(fname+'.npy',indices)
      return True
   return False

def pullSaveGTTile(wmsConn, dims, tile):
   indices, img = pullGTImageTile(wmsConn, dims, tile)
   if checkGTImage(indices,dims):
      fname = re.sub(r'[\[\]\(\) ]','',str(tile).replace(',','_').replace('.','p').replace('-','n'))
      img.save(fname + '_gt.png')
      np.save(fname+'.npy',indices)
      return True
   return False

def pullSaveRaw(wmsConn, profile, dims, bbx):
   u = pullRawImage(wmsConn, profile, dims, bbx)
   fname = re.sub(r'[\[\]\(\) ]','',str(bbx).replace(',','_').replace('.','p').replace('-','n'))
   f = open(fname + "_raw.png","w")
   f.write(u.read())
   f.close()

def pullSaveRawTile(wmsConn, profile, dims, tile):
   u = pullRawImageTile(wmsConn, profile, dims, tile)
   fname = re.sub(r'[\[\]\(\) ]','',str(tile).replace(',','_').replace('.','p').replace('-','n'))
   f = open(fname + "_raw.png","w")
   f.write(u.read())
   f.close()

# eventually we will use this as the curried callback function
def pullAndDoBoth(wmsConnRaw,wmsConnGT,profile, dims, bbx):
  if (pullSaveGT(wmsConnGT, dims, bbx)):
     pullSaveRaw(wmsConnRaw, profile, dims, bbx)

def pullAndDoBothTile(wmsConnRaw,wmsConnGT,profile, dims, bbx):
#     pullSaveGTTile(wmsConnGT, dims, bbx)
     pullSaveRawTile(wmsConnRaw, profile, dims, bbx)

def stupidWalk(polys, func):
  for poly in polys:
     for point in poly.points:
        func((point[0]-0.03, point[1]-0.03, point[0]+0.03, point[1]+0.03))

osm_meters_per_pixel = {20 : 0.1493, 19 : 0.2986, 18 : 0.5972, 17 : 1.1943, 16 : 2.3887 , 15 : 4.7773 , 14 : 9.5546 , 13 : 19.109, 12 : 38.219, 11 : 76.437, 10 : 152.87, 9  : 305.75, 8  : 611.50, 7  : 1222.99, 6  : 2445.98, 5  : 4891.97, 4  : 9783.94, 3  : 19567.88, 2  : 39135.76, 1  : 78271.52, 0 : 156412.0}
 
def shapeWalk(polys, dims, func):
  for poly in polys:
    shape_walker.shape_walk([x for x in poly.coords], 17, osm_meters_per_pixel, dims[0]/4, dims[0], func)

def doit(polys,config):  
  dgwmsConn=connectToDG(config)
  gtwmsConn=connectToGT(config)
  dims=(256,256)
  #currying ...for now, just use the GT, eventually we will use pullAndDoBoth
  mycallback = partial(pullAndDoBoth,dgwmsConn,gtwmsConn, 'Aerial_CIR_Profile', dims)
  # to be replaced by smart walk...which will also need zoom level, starting point(possibily), initial direction and distance to move 
  # in pixels..probably put all this in a 'config' object which is a dictionary
  shapeWalk(polys, dims, mycallback)
  

def doitTile(polys,config):  
  dgwmsConn=connectToDGTile(config)
  gtwmsConn=connectToGTTile(config)
  dims=(256,256)
  #currying ...for now, just use the GT, eventually we will use pullAndDoBoth
  mycallback = partial(pullAndDoBothTile,dgwmsConn,gtwmsConn, 'Aerial_CIR_Profile', dims)
  # to be replaced by smart walk...which will also need zoom level, starting point(possibily), initial direction and distance to move 
  # in pixels..probably put all this in a 'config' object which is a dictionary
  shapeWalk(polys, dims, mycallback)
