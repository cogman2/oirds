"""
coast2img.py
author: sneddon_robert@bah.com
date: 20101204, 20160316
Hat Tip:
shp2img.py - creates a png image and world file (.pgw) from a shapefile
containing a single polygon.
author: jlawhead@geospatialpyton.com


This script requires the Python Imaging Library.
The sample shapefile used is available at:
http://geospatialpython.googlecode.com/files/Mississippi.zip
"""

import shapefile
# import Image, ImageDraw
from PIL import Image, ImageDraw
import numpy as np

def makeMask(path, file_name, entry):
    '''

    Args:
        path: the path to the shape file
        file_name: the shape file name
        entry: the bounding box tile (will be updated to reference by lat/long

    Returns: a mask in Image format (PIL)
    Creates: a 'png' mask file and a 'pgw' world file with the name "file_name+entry" in the path directory.

    '''
    # Read in a shapefile
    s = shapefile.Reader(path+file_name)
    # s = shapefile.Reader("/Users/robertsneddon/DeepLearn/oirds/tools/coastlines-split-3857/lines")
    r = s.shape(entry)
    xdist = r.bbox[2] - r.bbox[0]
    ydist = r.bbox[3] - r.bbox[1]
    iwidth = 256
    iheight = 256
    # iheight = int(iwidth*ydist/xdist)
    xratio = iwidth/xdist
    yratio = iheight/ydist
    pixels = []
    for x,y in r.points:
        px = int(iwidth - ((r.bbox[2] - x) * xratio))
        py = int((r.bbox[3] - y) * yratio)
        pixels.append((px,py))
    img = Image.new("RGB", (iwidth, iheight), "white")
    draw = ImageDraw.Draw(img)
    draw.polygon(pixels, outline="rgb(203, 196, 190)", fill="rgb(198, 204, 189)")
    img.save(file_name+str(entry)+".png")

    # print "bounding box:"
    # print r.bbox[1], ",", r.bbox[0]
    # print r.bbox[3], ",", r.bbox[2]
    temp= list()
    for point in r.points:
        print point[1],",",point[0]
        pair = (point[1],point[0])
        temp.append(pair)
    print temp
    # Create a world file
    # wld = file(filename+str(entry)+".pgw", "w")
    # wld.write("%s\n" % (xdist/iwidth))
    # wld.write("0.0\n")
    # wld.write("0.0\n")
    # wld.write("-%s\n" % (ydist/iheight))
    # wld.write("%s\n" % r.bbox[0])
    # wld.write("%s\n" % r.bbox[3])
    # wld.close
    return img

def findShape(path,file_name,entry):
    '''

    Args:
        path: path to shape file
        file_name: shape file name
        entry: value to be found

    Returns: specific shape file

    '''
    s = shapefile.Reader(path+file_name)
    # s = shapefile.Reader("/Users/robertsneddon/DeepLearn/oirds/tools/coastlines-split-3857/lines")
    r = s.shape(entry)

path  = "/Users/robertsneddon/DeepLearn/oirds/tools/coastlines-split-4326/"
file_name = "lines"
entry = 8
myImg = makeMask(path, file_name, entry)