import numpy as np
from LatLon import *
import geopy, pyproj


def scale_map(lat,long,zoom_level, pixel_size):
    scale_dic = {20 : 1128.497220, 19 : 2256.994440, 18 : 4513.988880, 17 : 9027.977761, 16 : 18055.955520, 15 : 36111.911040, 14 : 72223.822090, 13 : 144447.644200, 12 : 288895.288400, 11 : 577790.576700, 10 : 1155581.153000, 9  : 2311162.307000, 8  : 4622324.614000, 7  : 9244649.227000, 6  : 18489298.450000, 5  : 36978596.910000, 4  : 73957193.820000, 3  : 147914387.600000, 2  : 295828775.300000, 1  : 591657550.500000}
    scale_pix = {20 : 0.1493, 19 : 0.2986, 18 : 0.5972, 17 : 1.1943, 16 : 2.3887 , 15 : 4.7773 , 14 : 9.5546 , 13 : 19.109, 12 : 38.219, 11 : 76.437, 10 : 152.87, 9  : 305.75, 8  : 611.50, 7  : 1222.99, 6  : 2445.98, 5  : 4891.97, 4  : 9783.94, 3  : 19567.88, 2  : 39135.76, 1  : 78271.52}
    # scale_pix = {20 : 0.07465, 19 : 0.1493, 18 : 0.2986, 17 : 0.5972, 16 : 1.1943, 15 : 2.3887 , 14 : 4.7773 , 13 : 9.5546 , 12 : 19.109, 11 : 38.219, 10 : 76.437, 9 : 152.87, 8  : 305.75, 7  : 611.50, 6  : 1222.99, 5  : 2445.98, 4  : 4891.97, 3  : 9783.94, 2  : 19567.88, 1  : 39135.76, 0  : 78271.52}

    # center=-28.496109,-48.752528&zoom=17&size=256x256&
    # output= bounding box in lat,long
    santa_catarina = ( -28.496109, -48.752528)
    # pixel_size = 256
    total_scale = np.cos(lat*(np.pi/180)) * scale_pix[zoom_level] * pixel_size * 0.001 # Convert to Kilometers
    # ctr_dist = sqrt(0.5*total_scale*total_scale)
    ctr_dist = total_scale / np.sqrt(2)
    center_point = LatLon(lat,long)
    headings = [-135+i*180 for i in range(0,2)]
    bounding_box = list()
    for heading in headings:
        temp_point = center_point.offset(heading,ctr_dist)
        lat_tmp, long_tmp = temp_point.to_string('D')
        bounding_box.append((long_tmp, lat_tmp)) # Need to get the correct accessor
    return bounding_box


def degrees_to_move (lat, long, pixels, zoom, dir_str, size):
    '''
    given a lat long centroid and amount of pixels to move the centroid, return a new bounding box
    '''
    # scale_pix = {20 : 0.07465, 19 : 0.1493, 18 : 0.2986, 17 : 0.5972, 16 : 1.1943, 15 : 2.3887 , 14 : 4.7773 , 13 : 9.5546 , 12 : 19.109, 11 : 38.219, 10 : 76.437, 9 : 152.87, 8  : 305.75, 7  : 611.50, 6  : 1222.99, 5  : 2445.98, 4  : 4891.97, 3  : 9783.94, 2  : 19567.88, 1  : 39135.76, 0  : 78271.52}
    scale_pix = {20 : 0.1493, 19 : 0.2986, 18 : 0.5972, 17 : 1.1943, 16 : 2.3887 , 15 : 4.7773 , 14 : 9.5546 , 13 : 19.109, 12 : 38.219, 11 : 76.437, 10 : 152.87, 9  : 305.75, 8  : 611.50, 7  : 1222.99, 6  : 2445.98, 5  : 4891.97, 4  : 9783.94, 3  : 19567.88, 2  : 39135.76, 1  : 78271.52}
    heading = {"e" : 90.0, "n" : 0.0, "w" : 270.0, "s": 180.0}
    shift_dist =  np.cos(lat*(np.pi/180)) * scale_pix[zoom] * float(pixels) * 0.001 # Convert to Kilometers
    # total_scale = np.cos(lat*(np.pi/180)) * scale_pix[zoom_level] * pixel_size * 0.001
    center_point = LatLon(lat,long)
    new_center_point = center_point.offset(heading[dir_str], shift_dist)
    lat_tmp, long_tmp = new_center_point.to_string('D')
    # print lat_tmp, long_tmp
    bounding_box = scale_map(float(lat_tmp),float(long_tmp),zoom, size)
    return bounding_box


lat = -28.496109
long = -48.752528
# lat = -32.145745
# long= -52.077820
# lat = -24.318441
# long = -67.532294
zoom = 17
size = 256
pixels = 16
dir_str = "w"
bounding_box = scale_map(lat,long,zoom,size)
print bounding_box
for tuple in bounding_box:
    print tuple[1],tuple[0]

new_bounding_box = degrees_to_move(lat, long, pixels, zoom, dir_str, size)
for tuple in new_bounding_box:
    print tuple[1],tuple[0]
