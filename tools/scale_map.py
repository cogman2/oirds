import numpy as np
from LatLon import *
# import geopy, pyproj


def scaleMap(lat,long,zoom, map_size):
    '''
    Args:
        lat: latitude of the map centroid in decimal format
        long: longitutde of the map centroid in decimal format
        zoom: a number between 1 and 20 giving the zoom level of the map
        map_size: the pixel size of the map, e.g., 256 (by 256)

    Returns: A list of the southwest and northeast boundingbox points. Points are ordered LONGITUDE, LATITUDE

    '''
    scale_pix = {20 : 0.1493, 19 : 0.2986, 18 : 0.5972, 17 : 1.1943, 16 : 2.3887 , 15 : 4.7773 , 14 : 9.5546 , 13 : 19.109, 12 : 38.219, 11 : 76.437, 10 : 152.87, 9  : 305.75, 8  : 611.50, 7  : 1222.99, 6  : 2445.98, 5  : 4891.97, 4  : 9783.94, 3  : 19567.88, 2  : 39135.76, 1  : 78271.52}
    total_scale = np.cos(lat*(np.pi/180)) * scale_pix[zoom] * map_size * 0.001 # Convert to Kilometers
    ctr_dist = total_scale / np.sqrt(2)
    center_point = LatLon(lat,long)
    headings = [-135+i*180 for i in range(0,2)]
    bounding_box = list()
    for heading in headings:
        temp_point = center_point.offset(heading,ctr_dist)
        lat_tmp, long_tmp = temp_point.to_string('D')
        bounding_box.append((long_tmp, lat_tmp)) # Need to get the correct accessor
    return bounding_box


def degreesToMove (lat, long, pixels, zoom, dir_str, map_size):
    '''
    given a lat long centroid and an amount of pixels to move the centroid, returns a new bounding box of a map moved by the number of "pixels"

    Args:
        lat: latitude of the map centroid in decimal format
        long: longitutde of the map centroid in decimal format
        pixels: the number of pixels to move the centroid
        zoom: a number between 1 and 20 giving the zoom level of the map
        dir_str: A letter indicating the direction to move "n","e","s" or "w" for north, east, south or west
        map_size: the map_size of the map, e.g., 256 (by 256)

    Returns: a bounding box with the centroid moved 'pixels' in 'dir_str',
        That is, a list of the southwest and northeast boundingbox points. Points are ordered LONGITUDE, LATITUDE
    '''
    scale_pix = {20 : 0.1493, 19 : 0.2986, 18 : 0.5972, 17 : 1.1943, 16 : 2.3887 , 15 : 4.7773 , 14 : 9.5546 , 13 : 19.109, 12 : 38.219, 11 : 76.437, 10 : 152.87, 9  : 305.75, 8  : 611.50, 7  : 1222.99, 6  : 2445.98, 5  : 4891.97, 4  : 9783.94, 3  : 19567.88, 2  : 39135.76, 1  : 78271.52}
    heading = {"e" : 90.0, "n" : 0.0, "w" : 270.0, "s": 180.0}
    shift_dist =  np.cos(lat*(np.pi/180)) * scale_pix[zoom] * float(pixels) * 0.001 # Convert to Kilometers
    center_point = LatLon(lat,long)
    new_center_point = center_point.offset(heading[dir_str], shift_dist)
    lat_tmp, long_tmp = new_center_point.to_string('D')
    bounding_box = scaleMap(float(lat_tmp),float(long_tmp),zoom, map_size)
    return bounding_box
