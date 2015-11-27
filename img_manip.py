#!/usr/bin/env python
# image manipulation
from PIL import Image
import math
import numpy

def crop(img, pos, dims): 
    x_delta = dims[0]//2
    y_delta = dims[1]//2
    x_pos = int(pos[0])
    y_pos = int(pos[1])
    left   = x_pos - x_delta
    right  = x_pos + x_delta
    top    = y_pos - y_delta
    bottom = y_pos + y_delta
    return img.crop((left, top, right, bottom))

def rotate( img, angle ):
    return img.rotate( angle )

def rot_matrix( angle_deg ):
    angle = math.pi*angle_deg/180.0
    cos_theta, sin_theta = math.cos(angle), math.sin(angle)
    return numpy.matrix( [[cos_theta, -sin_theta], [sin_theta, cos_theta]] )

def rot_img_vec( vec, angle_deg, center=(0,0) ):
    np_vec = numpy.array(vec)
    np_center = numpy.array(center)
    np_vec_prime = np_vec-np_center
    m = rot_matrix( -angle_deg ) # negative angle since top of image is (0,0)
    return numpy.inner( m, np_vec_prime ) + np_center

def crop_rotate( img, pos, dims, angle ):
    rot_img = rotate( img, angle )
    center = tuple([x//2 for x in img.size])
    rot_pos = rot_img_vec( pos, angle, center )
    return crop( rot_img, rot_pos, dims )

def open( filename ):
    return Image.open( filename )
