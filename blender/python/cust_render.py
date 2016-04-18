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

p1 = np.asarray((0.1,20.00000,25.00000))
x_m = np.sqrt((p1*p1).sum())

ax = -math.atan(p1[1]/p1[2])
ay = math.acos(p1[2]/(x_m*math.cos(ax)))
az= math.atan(-math.cos(ax)*math.sin(ay)/math.sin(ax)) - math.atan(p1[0]/p1[1])
lamp_object.location = p1

print (lamp_object.rotation_mode)
print(lamp_object.rotation_euler)
lamp_object.rotation_euler = (ax,ay,az)
point_object.rotation_euler = (ax,ay,az)
print(lamp_object.rotation_euler)

print(car_object.rotation_euler)
car_object.rotation_euler=(math.radians(90),math.radians(0),math.radians(round(random.uniform(-180,180),4)))
print(car_object.rotation_euler)

range=[7.5,7.5]
# will use car dimesions and angle to restrict range
#print(car_object.dimensions)

print(car_object.location)
car_object.location = (3.65382,-4.90261,0)
print(car_object.location)

# Render Scene and store the scene
bpy.ops.render.render( write_still=True ,use_viewport=True)

