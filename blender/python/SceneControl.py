# -*- coding: utf-8  -*-
#
# Create the image using Blender API
#
import bpy
from  math import *

class SceneControl(object):

    def __init__(self, **kwargs):
        allowed_keys  = ['image_name', 'lamp_name','scene_name','car_name','car_rotation_axis']
        self.C = bpy.context
        self.D = bpy.data
        self.O = bpy.ops
        self.image_name = 'Image'
        self.car_name = 'Group'
        self.scene_name = 'Scene'
        self.lamp_name = 'Lamp'
        self.car_rotation_axis = 'z'
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        self.deg2rad = pi/180

    def object_list(self):
        return [ob.name for ob in self.D.objects]

    #In lieu of using the contructor to set the car, a set of group name's can be used
    # Use with caution because the group may not be the car
    def find_car_from_group(self):
        self.car_names = ['Group','Group.001','Group.002','Group.003','Group.004','Group.005'] 
        objects = self.object_list()
        # print("Original car name is: ", self.car_name)
        for car_name in self.car_names:
        #    print("Car name being examined is: ",car_name)
            if car_name in objects:
                self.car_name = car_name
        #        print("chosen car is: ", self.car_name)
                break
        
    def set_car_scale(self, scale):
        self.deselect_all()
        self.select_one(self.car_name)
        obj = self.C.active_object
        obj.scale = (scale, scale, scale)

    def scale_car(self, scale_factor):
        obj = self.D.objects[self.car_name]
        obj.scale = obj.scale*scale_factor

    def set_car_rotation(self,x,y,z):
        obj = self.D.objects[self.car_name]
        obj.rotation_euler = (x,y,z)

# does not change car rotation two axis
# different cars have an initial (x,y,z) rotation
# the blender files should initialize the rotation such that the
# car appears from the top looking straight from the camera
# the axis rotate is confiurable
    def move_car(self, px, py, pz, rotation):
        obj = self.D.objects[self.car_name]
        obj.location = (px, py, pz)
        oe=obj.rotation_euler
        ra=self.car_rotation_axis
        obj.rotation_euler = (rotation if ra=='x' else oe[0],rotation if ra=='y' else oe[1], rotation if ra=='z' else oe[2])
        
    def move_lamp(self, sun_elev, sun_azimuth, sun_dist):
        px = sun_dist * cos(self.deg2rad * sun_elev) * sin(self.deg2rad * sun_azimuth)
        py = sun_dist * cos(self.deg2rad * sun_elev) * cos(self.deg2rad * sun_azimuth)
        pz = sun_dist * sin(self.deg2rad * sun_elev)
        obj = self.D.objects[self.lamp_name]
        obj.location = (px,py,pz)

    def move_lamp_zFixed(self, sun_elev, sun_azimuth, pz):
        # pz = 30
        sun_dist = pz
        px = sun_dist * (1.0/tan(self.deg2rad * sun_elev)) * sin(self.deg2rad * sun_azimuth)
        py = sun_dist * (1.0/tan(self.deg2rad * sun_elev)) * cos(self.deg2rad * sun_azimuth)
        obj = self.D.objects[self.lamp_name]
        obj.location = (px,py,pz)



    def move_lamp_guess(self, shadow_dir, shadow_len):
        # shadow_dir measured from "up" y-axis, clockwise, just like sun azimuth
        sun_dist = 30
        if shadow_len == 'l': rho = 40
        elif shadow_len == 'm': rho = 20
        else: rho = 5
        sun_azimuth = (180 +  shadow_dir) % 360
        self.move_lamp_zfixed(self, sun_elev, sun_azimuth)
        px = rho * cos(self.deg2rad * sun_azimuth)
        py = rho * sin(self.deg2rad * sun_azimuth)
        obj = self.D.objects[self.lamp_name]
        obj.location = (px,py,sun_dist)


    def move_lamp_camAngle(self, sun_elev, sun_azimuth, cam_angle, z):
        # the camera angle is from "north" 
        self.move_lamp_zFixed(sun_elev, sun_azimuth-(cam_angle - 180),z)

    def adjust_off_nadir(self, off_nadir, cam_angle) 
        pass



    def link_lamp(self):
        self.select_one(self.lamp_name)
        obj = self.D.objects[self.lamp_name].constraints.new(type='TRACK_TO')
        obj.target = self.C.object.get(self.car_name)

    def deselect_all(self):
        objects = self.object_list()
        for obj in objects:
            self.D.objects[obj].select = False

    def clean_scene(self):
#        self.save_list = ['Camera', self.car_name, self.lamp_name]
        self.save_list = ['Camera', self.lamp_name]
        self.deselect_all()
        objects = self.object_list()
        for obj in objects:
            delete_obj = True
            # print("In for loop object is: ",obj.name)
            for save_obj in self.save_list:
                if save_obj == obj: delete_obj = False
                # print("Saved object is: ",save_obj)
            if delete_obj: 
                # print( "delete_obj in if: ", delete_obj)
                # print("delete object is: ",obj.name)
                self.D.objects[obj].select = True
                self.O.object.delete()
                
    def select_one(self,obj_name):
        # print("Obj_name is: ", obj_name)
        self.deselect_all()
        self.D.objects[obj_name].select = True
        self.C.scene.objects.active = self.D.objects[obj_name]

    def set_image_path(self, pathname):
        tree = self.C.scene.node_tree
        imgNode=[n for n in tree.nodes if n.name==self.image_name][0]
        imgNode.image.filepath=pathname

    def center_camera(self,center_height):
        self.select_one("Camera")
        self.C.scene.objects.active = self.D.objects["Camera"]
        self.O.object.rotation_clear()
        self.O.object.location_clear()
        self.C.object.location[2] = center_height
        self.C.scene.render.resolution_percentage = 25
        self.C.scene.render.resolution_y = 1024
        self.C.scene.render.resolution_x = 1024


    def create_empty_with_pic(self,empty_name, z_empty, path_to_picture):
        empty = self.D.objects.new(empty_name, None)
        scene = self.C.scene
        scene.objects.link(empty)
        scene.update()
        empty.location = (-8,-8, z_empty)
        empty.rotation_euler  = (0, 0, 0)
        empty.empty_draw_type = 'IMAGE'
        img = self.D.images.load(path_to_picture)
        empty.data = img
        empty.empty_draw_size = 16
        return empty

    def append_model(self,filepath, group_name):
            # append, set to true to keep the link to the original file
        link = False 
        # append all groups from the .blend file
        with self.D.libraries.load(filepath, link=link) as (data_src, data_dst):
        ## all groups
        # data_to.groups = data_from.groups

            # only append a single group we already know the name of
            data_dst.groups = [group_name]

            # add the group instance to the scene
        scene = self.C.scene
        for group in data_dst.groups:
            ob = self.D.objects.new(group.name, None)
            ob.dupli_group = group
            ob.dupli_type = 'GROUP'
            scene.objects.link(ob)
        return [group for group in data_dst.groups]
