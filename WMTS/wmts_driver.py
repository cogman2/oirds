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
from shapely.geometry import box
from shapely.geometry import polygon
from shapely.geometry import linestring
from shapely.geometry.polygon import LinearRing


def connectToGT(config):
  return WebMapService(config['gthost'], version='1.1.1')

def connectToDG(config):
  return WebMapService("https://evwhs.digitalglobe.com/mapservice/wmsaccess?connectid=" + config['connectid'], username=config['uname'], password=config['passwd'], version='1.1.1')

# pullGTImage(wmsConn, (-48.754156,-28.497557,-48.7509017,-28.494662), (256,256))
def pullGTImage(config, wmsConn,dims, bbx):
  u = wmsConn.getmap(layers=[config['gtlayer']],  srs='EPSG:4326',  bbox=bbx, size=dims,  format='image/png')
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
  hist = np.histogram(img_array,2)[0]
  return hist[0]/float((dims[0]*dims[1])) >= 0.25 and hist[1]/float((dims[0]*dims[1])) >= 0.25

# pullRawImage(wmsConn,'Aerial_CIR_Profile', (-48.754156,-28.497557,-48.7509017,-28.494662), (256,256))
def pullRawImage(wmsConn,profile, dims, bbx):
#  print bbx,dims,profile
  return wmsConn.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:4326',  bbox=bbx, size=dims,  format='image/png', transparent=True, BGCOLOR='0xFFFFFF', FEATUREPROFILE=profile)

def pullSaveGT(config, wmsConn, dims, bbx):
   indices, img = pullGTImage(config,wmsConn, dims, bbx)
   if checkGTImage(indices,dims):
      fname = re.sub(r'[\[\]\(\) ]','',str(bbx).replace(',','_').replace('.','p').replace('-','n'))
      img.save(fname + '_gt.png')
      np.save(fname+'.npy',indices)
      return True
   return False

def pullSaveRaw(config, wmsConn, dims, bbx):
   u = pullRawImage(wmsConn,config['profile'], dims, bbx)
   fname = re.sub(r'[\[\]\(\) ]','',str(bbx).replace(',','_').replace('.','p').replace('-','n'))
   f = open(fname + "_raw.png","w")
   f.write(u.read())
   f.close()

# eventually we will use this as the curried callback function
def pullAndDoBoth(config,wmsConnRaw,wmsConnGT,dims,bbx):
  if (config['confines'].intersects(box(bbx[0],bbx[1],bbx[2],bbx[3]))):
    if (pullSaveGT(config,wmsConnGT, dims, bbx)):
       pullSaveRaw(config,wmsConnRaw, dims, bbx)

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
  mycallback = partial(pullAndDoBoth,config,dgwmsConn,gtwmsConn, dims)
  # to be replaced by smart walk...which will also need zoom level, starting point(possibily), initial direction and distance to move 
  # in pixels..probably put all this in a 'config' object which is a dictionary
  shapeWalk(polys, dims, mycallback)
  
def toShape(x):
   if (len(x) == 2):
      return linestring.LineString(x)
   return polygon.asPolygon(x)

def loadPolys(config):
  import shapefile
  sf = shapefile.Reader("/home/rwgdrummer/oirds/land-polygons-complete-4326/land_polygons")
  bbx=config['confines']
  shapes = sf.shapes()
  shapeSetOfInterest = [linestring.LineString(x.points) for x in shapes if toShape(x.points).intersects(bbx)]
  return shapeSetOfInterest

def doitFromConfig(config):
  doit(loadPolys(config),config)

def main():
   import pickle
   import sys

   if len(sys.argv) < 1:
      print "Usage: ", sys.argv[0], " configFile"
      sys.exit( 0 )

   f = open(sys.argv[1],"rb")
   config = pickle.load(f)
   f.close()
   bbxarray = [float(x) for x in config['box'].split(',')]
   bbx = box(bbxarray[0],bbxarray[1],bbxarray[2],bbxarray[3])
   config['confines']=bbx
   doitFromConfig(config)

if __name__=="__main__":
    main()  
