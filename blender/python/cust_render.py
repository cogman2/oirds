import bpy
import math
import numpy as np
import random
import img_meta

imp = img_meta.ImgMetaProcess()
imp.openFile('img_desc.csv')

imd = imp.images['img1.png']

# print all objects
for obj in bpy.data.objects:
    print(obj.name)

# print all scene names in a list
#print(bpy.data.scenes.keys())

# change input image node
tree = bpy.data.scenes['Scene'].node_tree
imgNode=[n for n in tree.nodes if n.name=='Image'][0]
imgNode.image.filepath=imd.name


scene = bpy.data.scenes["Scene"]

#create new lamp datablock
lamp_object = bpy.data.objects['Lamp']
point_object = bpy.data.objects['Point']
car_object = bpy.data.objects['PickUp']
camera_object = bpy.data.objects['Camera']


camera_angle = camera_object.rotation_euler[2]
lz = lamp_object.location[2]
truck_object_z_rot=car_object.rotation_euler[2]

off_nadir = imd.offnadir
sun_azimuth = imd.azimuth
sun_elev = imd.elevation
placement = imd.getImagePlace()

# 180 assumes the camera is aligned to the y axis
lx = lz * math.cos(sun_elev) * math.cos(90 + camera_angle - sun_azimuth)
ly = lz * math.cos(sun_elev) * math.sin(180 + camera_angle - sun_azimuth)

lamp_loc = np.array([lx,ly,lz])

truck_object_y_rot=placement[0]

car_object.rotation_euler=(math.radians(truck_object_x_rot),math.radians(truck_object_y_rot),math.radians(truck_object_z_rot))

# pixel to blender space conversion
image_factor = 0.1
car_object.location = (placement[1][0]*image_factor,placement[1][1]*image_factor,car_object.location[2])

lamp_object.location = lamp_loc

# Render Scene and store the scene
bpy.ops.render.render( write_still=True ,use_viewport=True)
bpy.data.scenes["Scene"].render.image_settings.file_format = 'PNG'
bpy.data.scenes["Scene"].render.filepath = '//plane'

# Render Scene and store the scene 
#bpy.ops.render.render( write_still=True ) 
