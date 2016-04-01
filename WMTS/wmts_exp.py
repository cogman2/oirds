from owslib.wcs import WebCoverageService

passwd = ''

wcs = WebCoverageService("https://evwhs.digitalglobe.com/deliveryservice/wcsaccess?connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0",username='roberteb',password=passwd)
wcs.contents

wcs = WebCoverageService("https://evwhs.digitalglobe.com/deliveryservice/wcsaccess?connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0",username='roberteb',password=passwd)


from urllib import urlencode
from urlparse import urlparse, urlunparse, parse_qs, ParseResult
from owslib.etree import etree
from owslib.util import openURL, testXMLValue, getXMLInteger

u = openURL("https://evwhs.digitalglobe.com/deliveryservice/wcsaccess","service=WCS&connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&request=DescribeCoverage&identifiers=f75a1d39032dd6225c6c9d7dbe9a2a17&version=1.1.1", method='Get', username='roberteb',password=passwd)

f = open("getcov.xml","w")
f.write(u.read())
f.close()


u = openURL("https://evwhs.digitalglobe.com/deliveryservice/wcsaccess","service=WCS&connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&request=GetCapabilities&version=1.1.1", method='Get', username='roberteb',password=passwd)

f = open("getcap.xml","w")
f.write(u.read())
f.close()

u.read()

u = openURL("https://evwhs.digitalglobe.com/deliveryservice/wcsaccess","service=WCS&connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&request=GetCoverage&identifier=ff45df54c35121123c6840354e69a70a&version=1.1.1&FORMAT=image/geotiff&BoundingBox=-9.098914499690142,-35.326876500164836,-8.018585999999912,-35.118364499929875,urn:ogc:def:crs:EPSG::4326&GridBaseCRS=urn:ogc:def:crs:EPSG::4326&GridCS=urn:ogc:def:cs:OGC:0.0:Grid2dSquareCS&GridType=urn:ogc:def:method:WCS:1.1:2dGridIn2dCrs&GridOrigin=-9.098912249690063,-35.118362249929795&GridOffsets=0.0000045,-0.0000045", method='Get', username='roberteb',password=passwd)
u.read()

u = openURL("https://evwhs.digitalglobe.com/deliveryservice/wcsaccess","service=WCS&connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&request=GetCoverage&identifier=f75a1d39032dd6225c6c9d7dbe9a2a17&version=1.1.1&FORMAT=image/geotiff&BoundingBox=-8.623471213009191,-34.87485778546993,-7.474644385918229,-34.74729522811673,urn:ogc:def:crs:EPSG::4326&GridBaseCRS=urn:ogc:def:crs:EPSG::4326&GridCS=urn:ogc:def:cs:OGC:0.0:Grid2dSquareCS&GridType=urn:ogc:def:method:WCS:1.1:2dGridIn2dCrs&GridOrigin=-8.62346967925233,-34.74729369435987&GridOffsets=3.06751372E-6,-3.06751372E-6", method='Get', username='roberteb',password=passwd)
u.read()

from owslib.wmts import WebMapTileService

src="/myDigitalGlobe/getbrowse?featureId=2f70a09f987b84c003809ded0b84d72c&footprintwkt=POLYGON ((-61.03964250016393 10.001272499939741, -61.03964250016393 10.00007099993966, -60.98612400010362 10.000003499939657, -60.84463049994418 10.000183499939668, -60.846286499946046 10.905209999999997, -61.03848600016263 10.882696499998495, -61.03964250016393 10.001272499939741))&viewportwkt=POLYGON ((-60.96700251102447 10.340243409146554,-60.96443027257919 10.340243409146554,-60.96443027257919 10.339170796977305,-60.96700251102447 10.339170796977305,-60.96700251102447 10.340243409146554))&featureIdType=FINISHED_FEATURE&archiveType=null"

wmts = WebMapTileService("https://evwhs.digitalglobe.com/earthservice/wmtsaccess?connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0",username='roberteb',password=passwd)
wmts.contents

wmts.contents['DigitalGlobe:ImageryTileService'].formats

u = openURL("https://evwhs.digitalglobe.com/earthservice/wmtsaccess","service=WMTS&connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&request=GetCapabilities&version=1.1.1", method='Get', username='roberteb',password=passwd)


dir(wmts.contents['DigitalGlobe:ImageryTileService'].tilematrixsetlinks['EPSG:4326'])

tile = wmts.gettile(layer='DigitalGlobe:ImageryTileService', tilematrixset='EPSG:3857', tilematrix='EPSG:3857:17', row=47785, column=47785, format="image/png")
#tile = wmts.gettile(layer='DigitalGlobe:ImageryTileService', tilematrixset='EPSG:4326', tilematrix='EPSG:4326:17', column=47785, row=76367, format="image/png")
f = open("stile.png","wb")
f.write(tile.read())
f.close()


src="https://evwhs.digitalglobe.com/tiles/earthservice/wmtsaccess?connectId=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&cdnKey=st=1457626216~exp=1457799016~acl=%2ftiles%2fearthservice%2fwmtsaccess%3fconnectId%3dc19b1c96-cd25-453a-96ef-4cd223c6d2c0%2a~hmac=4089e046ff674c88d05a814b4afe848eab06760e5efdb43c236ab2dbbca04a08&dgToken=607974af7d39943ec1e950edc04390fab00790b780281756d4ec634b8dc1149f&SERVICE=WMTS&VERSION=1.0.0&REQUEST=GetTile&TileMatrixSet=EPSG:3857&LAYER=DigitalGlobe:ImageryTileService&FORMAT=image/jpeg&STYLE=&featureProfile=Global_Currency_Profile&TileMatrix=EPSG:3857:20&TILEROW=494005&TILECOL=346711"

756d4ec634b8dc1149f&FEATURECOLLECTION=6ba149dfe589f442e9325c9593425ab4&USECLOUDLESSGEOMETRY=false&SRS=EPSG%3A3857&BBOX=-6797392.0513441535,1164288.814839805,-6794946.066439027,1166734.7997449306"

src="https://evwhs.digitalglobe.com/mapservice/wmsaccess?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=DigitalGlobe%3AImagery&STYLES=&FORMAT=image%2Fpng&TRANSPARENT=true&HEIGHT=256&WIDTH=256&BGCOLOR=0xFFFFFF&CONNECTID=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&DGTOKEN=607974af7d39943ec1e950edc04390fab00790b780281756d4ec634b8dc1149f&FEATURECOLLECTION=2f70a09f987b84c003809ded0b84d72c&USECLOUDLESSGEOMETRY=false&SRS=EPSG%3A3857&BBOX=-6786690.867384229,1157294.8267517104,-6786652.648870086,1157333.045265852"

src="https://evwhs.digitalglobe.com/mapservice/wmsaccess?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=DigitalGlobe%3AImagery&STYLES=&FORMAT=image%2Fpng&TRANSPARENT=true
&HEIGHT=256&WIDTH=256&BGCOLOR=0xFFFFFF&CONNECTID=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&DGTOKEN=607974af7d39943ec1e950edc04390fab00790b780281756d4ec634b8dc1149f&
FEATURECOLLECTION=2f70a09f987b84c003809ded0b84d72c&USECLOUDLESSGEOMETRY=false&SRS=EPSG%3A3857&BBOX=-6786729.085898371,1157294.8267517104,-6786690.867384229,1157333.045265852"

src="https://evwhs.digitalglobe.com/mapservice/wmsaccess?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=DigitalGlobe%3AImagery&STYLES=&FORMAT=image%2Fpng&TRANSPARENT=true&HEIGHT=256&WIDTH=256&BGCOLOR=0xFFFFFF&CONNECTID=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&DGTOKEN=607974af7d39943ec1e950edc04390fab00790b780281756d4ec634b8dc1149f&FEATURECOLLECTION=2f70a09f987b84c003809ded0b84d72c&USECLOUDLESSGEOMETRY=false&SRS=EPSG%3A3857&BBOX=-6786843.741440799,1157256.608237568,-6786690.867384229,1157409.4822941392"

https://evwhs.digitalglobe.com/mapservice/wmsaccess?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=DigitalGlobe:Imagery&STYLES=&FORMAT=image/png&TRANSPARENT=true&HEIGHT=256&WIDTH=256&BGCOLOR=0xFFFFFF&CONNECTID=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&DGTOKEN=607974af7d39943ec1e950edc04390fab00790b780281756d4ec634b8dc1149f&FEATURECOLLECTION=001c5087b3159c04b02b3d49e306ecaa&USECLOUDLESSGEOMETRY=false&SRS=EPSG:3857&BBOX=-6786996.615497369,1157256.608237568,-6786843.741440799,1157409.4822941392

from owslib.wms import WebMapService
wms = WebMapService("https://evwhs.digitalglobe.com/mapservice/wmsaccess?connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0", username='roberteb',password=passwd, version='1.1.1')

u = openURL("https://evwhs.digitalglobe.com/mapservice/wmsaccess","service=WMS&connectid=c19b1c96-cd25-453a-96ef-4cd223c6d2c0&request=GetCapabilities&version=1.1.1", method='Get', username='roberteb',password=passwd)

img = wms.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:3857',  bbox=(-6786996.615497369,1157256.608237568,-6786843.741440799,1157409.4822941392), size=(256,256),  format='image/png', transparent=True, BGCOLOR='0xFFFFFF',FEATURECOLLECTION='001c5087b3159c04b02b3d49e306ecaa')

img = wms.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:3857',  bbox=(-6786996.615497369,1157256.608237568,-6786843.741440799,1157409.4822941392), size=(256,256),  format='image/png', transparent=True, BGCOLOR='0xFFFFFF', FEATUREPROFILE='Aerial_CIR_Profile')

img = wms.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:4326',  bbox=(-48.753004,-28.496761, -48.751631,-28.495388), size=(256,256),  format='image/png', transparent=True, BGCOLOR='0xFFFFFF', FEATUREPROFILE='Aerial_CIR_Profile')
f = open("tile.png","wb")
f.write(img.read())
f.close()

from owslib.wms import WebMapService
wms1 = WebMapService('http://worldmapkit1.thinkgeo.com/CachedWmsServer/WmsServer.axd', version='1.1.1')
img = wms1.getmap(layers=['WorldMapKitImageryLayer'],  srs='EPSG:4326',  bbox=(-48.754156,-28.497557,-48.7509017,-28.494662), size=(256,256),  format='image/png')
f = open("tile1.png","wb")
f.write(img.read())
f.close()


wms1 = WebMapService('http://129.206.228.72/cached/osm', version='1.1.1')
[(x,wms1[x].boundingBox) for x in wms1.contents]
img = wms1.getmap(layers=['osm_auto:all'],  srs='EPSG:4326',  bbox=(-48.754156,-28.497557,-48.7509017,-28.494662), size=(256,256),  format='image/png')
f = open("tile3.png","wb")
f.write(img.read())
f.close()o

#40.714728,-73.998672
u = openURL('http://maps.googleapis.com/maps/api/staticmap?center=-28.496109,-48.752528&zoom=17&size=256x256&style=element:labels|visibility:off&style=element:geometry.stroke|visibility:off&style=feature:landscape|element:geometry|saturation:-100&style=feature:water|saturation:-100|invert_lightness:true&key=AIzaSyCA_Aa1ReaPuPrDaHTeemNo-GMsEk7xUVU')
#u = openURL('http://maps.googleapis.com/maps/api/staticmap?center=40.714728,-73.998672&zoom=17&size=256x256&style=element:labels|visibility:off&style=element:geometry.stroke|visibility:off&style=feature:landscape|element:geometry|saturation:-100&style=feature:water|saturation:-100|invert_lightness:true&key=AIzaSyCA_Aa1ReaPuPrDaHTeemNo-GMsEk7xUVU')
f = open("gtile.png","wb")
f.write(u.read())
f.close()

n=2**17
print 128 * 56543.03392 * math.cos(-28.496109 * math.pi / 180.0) / n



-28.494896748 -48.7511558042
-28.494896748 -48.7539001958
-28.4973212379 -48.7539002272
-28.4973212379 -48.7511557728

bb=-48.7539001958,-28.4973212379,-48.7511558042,-28.494896748
#bb=-28.496109,-48.752528,-28.496109+0.0002,-48.752528+0.0002
img = wms.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:4326',  bbox=bb, size=(256,256),  format='image/png', transparent=True, BGCOLOR='0xFFFFFF', FEATUREPROFILE='Aerial_CIR_Profile')
f = open("tile.png","wb")
f.write(img.read())
f.close()


img = wms.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:4326',  bbox=(-48.753204,-28.497161, -48.751831,-28.495788), size=(256,256),  format='image/png', transparent=True, BGCOLOR='0xFFFFFF', FEATUREPROFILE='Aerial_CIR_Profile')
f = open("tile.png","wb")
f.write(img.read())
f.close()

wms1 = WebMapService('http://129.206.228.72/cached/osm', version='1.1.1')
[(x,wms1[x].boundingBox) for x in wms1.contents]
img = wms1.getmap(layers=['osm_auto:all'],  srs='EPSG:4326',  bbox=bb, size=(256,256),  format='image/png')
f = open("tile1.png","wb")
f.write(img.read())
f.close()


import matplotlib
import pylab

u = openURL('http://maps.googleapis.com/maps/api/staticmap?center=-24.318441,-67.532294&zoom=17&size=320x320&style=element:labels|visibility:off&style=element:geometry.stroke|visibility:off&style=feature:landscape|element:geometry|saturation:-100&style=feature:water|saturation:-100|invert_lightness:true&key=AIzaSyCA_Aa1ReaPuPrDaHTeemNo-GMsEk7xUVU')
b = u.read()
stream = io.BytesIO(b)
img = Image.open(stream)
img = img.crop((0,0,256,256))
img.save("gtile2.png","PNG")
imga = np.asarray(img)
np.histogram(imga)
pylab.imshow(imga)
pylab.show(imga)

    
pos = (10.511424, -75.500974)

directionChange=[
direction=0
for i in xrange(1000):
   u = openURL('http://maps.googleapis.com/maps/api/staticmap?center=' + str(pos[0]) + ',' + str(pos[1]) + '&zoom=17&size=320x320&style=element:labels|visibility:off&style=element:geometry.stroke|visibility:off&style=feature:landscape|element:geometry|color=#000000&style=feature:water|color=#ffffff:true&key=AIzaSyCA_Aa1ReaPuPrDaHTeemNo-GMsEk7xUVU')
   b = u.read()
   stream = io.BytesIO(b)
   img = Image.open(stream)
   img = img.crop((0,0,256,256))
   imga = np.asarray(img)
   his = np.histogram(imga)
   percW =float(abs(his[0][9]))/65536.0
   if (percW > 0.30 && percW < 0.70):
   else:
      
   


u = openURL('http://maps.googleapis.com/maps/api/staticmap?center=-24.294036,-67.438567&zoom=17&size=320x320&style=element:labels|visibility:off&style=element:geometry.stroke|visibility:off&style=feature:landscape|element:geometry|color=#000000&style=feature:water|color=#ffffff:true&key=AIzaSyCA_Aa1ReaPuPrDaHTeemNo-GMsEk7xUVU')
b = u.read()
stream = io.BytesIO(b)
img = Image.open(stream)
img = img.crop((0,0,256,256))
img.save("gtile3.png","PNG")
imga = np.asarray(img)
np.histogram(imga)
pylab.imshow(imga)
pylab.show()




bb=-67.5340095769,-24.3200131141,-67.5305784231,-24.3168688662
img = wms.getmap(layers=['DigitalGlobe:Imagery'],  srs='EPSG:4326',  bbox=bb, size=(320,320),  format='image/png', transparent=True, BGCOLOR='0xFFFFFF', FEATUREPROFILE='Aerial_CIR_Profile')
b = img.read()
stream = io.BytesIO(b)
img = Image.open(stream)
img = img.crop((0,0,256,256))
img.save("tile2.png","PNG")



import shapefile


sf = shapefile.Reader("/Users/ericrobertson/Downloads/land-polygons-complete-4326/land_polygons")
#sf = shapefile.Reader("/Users/ericrobertson/Downloads/coastlines-split-4326/lines")
shapes = sf.shapes()

from shapely.geometry import bbox
from shapely.geometry import polygon
from shapely.geometry import linestring
from shapely.geometry.polygon import LinearRing
bbox=box(-57.494268,-37.828869,-57.4910342,-37.8276639)

bbx = box(-48.753004,-28.496761, -48.751631,-28.495388)

#bbx = box(-48.7539001958,-28.4973212379,-48.7511558042,-28.494896748)
#bbx1 = box(-48.9739001958,-28.5173212379,-48.9711558042,-28.514896748)

def toShape(x):
   if (len(x) == 2):
      return linestring.LineString(x)
   return polygon.asPolygon(x)

# find intersection

pp = [x for x in shapes if toShape(x.points).intersects(bbx)]
#poly = bbx.intersection(polygon.asPolygon(pp[0].points))
poly=polygon.asPolygon(pp[0].points)
lr = LinearRing(pp[0].points)
finalSize=(256,256)
indices = np.zeros((1,finalSize[0],finalSize[1]),dtype=np.uint8)
imLabel = Image.new("RGB", (256,256), color=(0,0,0))
#poly = toShape(pp[0].points)
#ls  = bbx.intersection(linestring.LineString(pp[0].points))
#poly  = bbx.intersection(ls)
placeCoordPolyInImage(imLabel, finalSize, poly,bbx.bounds, indices,1, (200,200,200))
convertPoly(finalSize,pp[0],bbx.bounds)

