import pdb
import numpy as np
from LatLon import *
import math
from shapely.geometry import box
import sys

def distance_for_latitude(lat, pixel_scale, step_size):
  return  np.cos(math.radians(float(lat))) * pixel_scale * step_size * 0.001 # Convert to Kilometers

def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

def shape_walk(lat_longs, zoom_level, zoom_dic, step_size, bbox_size, output_function):
    lat,long = lat_longs[0]
    pixel_scale = zoom_dic[zoom_level]
    step_dist = np.cos(float(lat)*(np.pi/180)) * pixel_scale * step_size * 0.001 # Convert to Kilometers
    directs = [-90+90*i for i in range(0,3)]
    for i in xrange(len(lat_longs)-1):
       start_points = map(point_function, lat_longs[i:i+2])
       try:
#         heading_slope = (lat_longs[i+1][0] - lat_longs[i][0])/(lat_longs[i+1][1] - lat_longs[i][1])
#         line_distance  = math.sqrt((lat_longs[i+1][0] - lat_longs[i][0])**2 + (lat_longs[i+1][1] - lat_longs[i][1])**2)
         heading = start_points[0].heading_initial(start_points[1])
         total_distance = start_points[0].distance(start_points[1])
         for bounding_box in  walk_the_line(start_points[0], heading, total_distance, step_dist, directs, pixel_scale, bbox_size):
            output_function(bounding_box.bounds)
       except ValueError:
         print sys.exc_info()[0]
         print start_points[0]
         print start_points[1]

def walk_the_line(start_point, heading, total_distance, step_dist, directs, pixel_scale, bbox_size):
    bounding_boxes = list()
    num_steps = int(np.floor(total_distance/step_dist))
    for i in range(0,num_steps):
        for dir in directs:
           bounding_boxes.append(to_bbox(move_step(start_point, dir+heading, step_dist), pixel_scale, bbox_size))
        start_point = start_point.offset(heading, step_dist)
    return bounding_boxes


def to_bbox(center_point, pixel_scale, box_size):
    lat,lon = center_point.to_string('D')
    total_scale = distance_for_latitude(float(lat),pixel_scale,box_size)
    ctr_dist = total_scale / np.sqrt(2)
    headings = [-135+i*180 for i in range(0,2)]
    bounding_box = list() 
    for heading in headings:
        temp_point = center_point.offset(heading,ctr_dist)
        lat_tmp, long_tmp = temp_point.to_string('D')
        bounding_box.append(((float(lat_tmp), float(long_tmp)))) # Need to get the correct accessor
    return box(bounding_box[0][0],bounding_box[0][1],bounding_box[1][0],bounding_box[1][1])

def move_step(begin_point, heading, step_dist):
    return begin_point if (step_dist == 0) else  begin_point.offset(heading, step_dist)

def point_function(x):
    return (LatLon(float(x[0]),float(x[1])))
