import pdb
import numpy as np
from LatLon import *
# import geopy, pyproj
from scale_map import *

# pdb.set_trace()

def coast_walk(lat_longs, zoom_level, zoom_dic, direction, step_size, map_size, output_function):
    
    lat,long = lat_longs[0]
    step_dist = np.cos(float(lat)*(np.pi/180)) * zoom_dic[zoom] * step_size * 0.001 # Convert to Kilometers
    directs = [-90+90*i for i in range(0,3)]
    all_bounding_boxes = list()
    all_outcomes = list()
#    point_function = lambda x: LatLon(float(x[0]),float(x[1]))
    for coord in lat_longs:
        for num  in coord: num  = float(num)

    for i, coord in enumerate(lat_longs):
        # walk_distance = 0
        if i == len(lat_longs) - 1: break
        start_points = map(point_function, lat_longs[i:i+2])
        heading = start_points[0].heading_initial(start_points[1])
        total_distance = start_points[0].distance(start_points[1])
        bounding_boxes = walk_the_line(start_points[0], heading, total_distance, step_dist, directs, zoom, map_size)
        outcomes = list()

        for bounding_box in bounding_boxes:
            outcomes.append(output_function(bounding_box))

        all_outcomes.append(outcomes)
        all_bounding_boxes.append(bounding_boxes)
    return all_bounding_boxes, all_outcomes

def walk_the_line(start_point, heading, total_distance, step_dist, directs, zoom, map_size):
    bounding_boxes = list()
    num_steps = int(np.floor(total_distance/step_dist))
    for i in range(0,num_steps):
        # start_point = start_point.offset(heading, step_dist)
        for j , dir in enumerate(directs):
            # temp_box = move_step(start_point, heading+dir, step_dist, zoom, map_size)
            if dir == 0: temp_box = move_step(start_point, heading, 0, zoom, map_size)
            else: temp_box = move_step(start_point, dir+heading, step_dist, zoom, map_size)
            bounding_boxes.append(temp_box)
        start_point = start_point.offset(heading, step_dist)
    return bounding_boxes



def move_step(begin_point, heading, step_dist, zoom, map_size):
    '''
    args:
         begin_point: starting point for the walk
         heading: the heading of the walk
         step_dist: the distance walked
         zoom: the level of zoom of the map
         map_size: map size in pixels
    return: bounding_box; two points lower left and upper right
    '''
    walk_point = begin_point.offset(heading, step_dist)
    lat_tmp, long_tmp = walk_point.to_string('D')
    bounding_box = scale_map(float(lat_tmp),float(long_tmp),zoom, map_size)
    return bounding_box

def point_function(x):
    return (LatLon(float(x[0]),float(x[1])))

def output_function(bounding_box):
    return 1
# Steps in function
#
# Calculate initial bounding box.

# 
# get heading
#
# take step in direction
#      find bounding box
#      call function w/bounding box
#
# step up perpendicular
#      find bounding box
#      call function w/bounding box
#
# step down perpendicular
#      find bounding box
#      call function w/bounding box
#
'''
    center_point = LatLon(lat,long)
    headings = [-135+i*180 for i in range(0,2)]
    bounding_box = list()
    for heading in headings:

        temp_point = center_point.offset(heading,ctr_dist)
        lat_tmp, long_tmp = temp_point.to_string('D')
       bounding_box.append((long_tmp, lat_tmp)) # Need to get the correct accessor
    return bounding_box
'''


lat_longs2 = [(55.162896, -4.95385), (55.163408, -4.953875), (55.163394, -4.95477), (55.16365, -4.954782), (55.163785, -4.954341), (55.163792, -4.953894), (55.163799, -4.953447), (55.164069, -4.952565), (55.164212, -4.951677), (55.163959, -4.951442), (55.163448, -4.951417), (55.163315, -4.951634), (55.163308, -4.952081), (55.163301, -4.952528), (55.163031, -4.95341), (55.162896, -4.95385)]
lat_longs3 = [(40.808859, -96.610495), (40.804574, -96.615719),(40.800153, -96.621395) ]
lat_longs5 = [(40.808859, -96.610495), (40.804574, -96.615719),(40.800153, -96.621395),(40.821972, -96.692562) ]
lat_longs = [(40.808859, -96.610495), (40.804574, -96.615719)]
lat_longs4 = [(55.162896, -4.95385), (55.163408, -4.953875), (55.163394, -4.95477),(55.16365, -4.954782), (55.163785, -4.954341)]

zoom_level = 17
zoom_dic = {19 : 0.298, 18 : 0.596, 17 : 1.193, 16 : 2.387 , 15 : 4.773 , 14 : 9.547 , 13 : 19.093, 12 : 38.187, 11 : 76.373, 10 : 152.746, 9  : 305.492, 8  : 610.984, 7  : 1222.0, 6  : 2444.0, 5  : 4888.0, 4  : 9776.0, 3  : 19551.0, 2  : 39103.0, 1  : 78206, 0 : 156412.0}
direction = "f"
# heading = -1.60121870922
# heading = -100
step_size = 256
# step_dist = 0.00272598215883*32
map_size = 256
# total_distance = 0.03186552093*32
# directs = [-90,0,90]
# start_points = map(point_function, lat_longs[0:0+2])
# heading = start_points[0].heading_initial(start_points[1])
# i=15
start_point = LatLon(lat_longs[0][0], lat_longs[0][1])
bounding_box = list()
# bounding_boxes = walk_the_line(start_point, heading, total_distance, step_dist, directs, zoom, map_size)
all_bounding_boxes, all_outcomes = coast_walk(lat_longs3, zoom_level, zoom_dic, direction, step_size, map_size, output_function)
for bounding_boxes in all_bounding_boxes:
    for bounding_box in bounding_boxes:
        for pair in bounding_box:
            print pair[1],"," , pair[0]