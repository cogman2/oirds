import bpy
import math
import numpy as np
import random

# print all objects
for obj in bpy.data.objects:
    print(obj.name)


# print all scene names in a list
print(bpy.data.scenes.keys())

# change input image node
tree = bpy.data.scenes['Scene'].node_tree
imgNode=[n for n in tree.nodes if n.name=='Image'][0]
imgNode.image.filepath='bg2.png'
#print(bpy.data.images)


pi = 3.14159265
fov = 75

scene = bpy.data.scenes["Scene"]

#create new lamp datablock
lamp_object = bpy.data.objects['Lamp']
point_object = bpy.data.objects['Point']
car_object = bpy.data.objects['PickUp']

off_nadir = 25.0
sun_elev = 51.1304
sun_azimuth = 41.2039
truck_object_x_rot=90

lz = 40
camera_angle = 90
# 180 assumes the camera is aligned to the y axis
lx = lz * math.cos(sun_elev) * math.cos(90 + camera_angle - sun_azimuth)
ly = lz * math.cos(sun_elev) * math.sin(180 + camera_angle - sun_azimuth

lamp_loc = np.array([lx,ly,lz])

lamp_len = np.sqrt((lamp_loc*lamp_loc).sum())
ax = -math.atan(lamp_loc[1]/lamp_loc[2])
ay = math.acos(lamp_loc[2]/(lamp_loc_len*math.cos(ax)))
az= math.atan(-math.cos(ax)*math.sin(ay)/math.sin(ax)) - math.atan(lamp_loc[0]/lamp_loc[1])

lamp_object.location = lamp_loc

print(lamp_object.rotation_mode)
print(lamp_object.rotation_euler)
# can leave it in radians
lamp_object.rotation_euler = (ax,ay,az)
point_object.rotation_euler = (ax,ay,az)
print(lamp_object.rotation_euler)

print(car_object.rotation_euler)
truck_object_y_rot=round(random.uniform(-180,180),4)
# insert function here to calculate the y angle.
truck_object_y_rot=0
car_object.rotation_euler=(math.radians(truck_object_x_rot),math.radians(truck_object_y_rot),math.radians(truck_object_z_rot))
print(car_object.rotation_euler)

range=[7.5,7.5]
# will use car dimesions and angle to restrict range
#print(car_object.dimensions)

print(car_object.location)
car_object.location = (3.65382,-4.90261,0)
print(car_object.location)

# Render Scene and store the scene
bpy.ops.render.render( write_still=True ,use_viewport=True)

