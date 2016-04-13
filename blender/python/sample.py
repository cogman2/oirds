import bpy

# print all objects
#for obj in bpy.data.objects:
#    print(obj.name)


# print all scene names in a list
#print(bpy.data.scenes.keys())


pi = 3.14159265
fov = 75

scene = bpy.data.scenes["Scene"]


#create new lamp datablock
lamp_data = bpy.data.lamps.new(name="New Lamp", type='SUN')

# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)

#lamp_object = bpy.data.lamps['New Lamp']
lamp_object.shadow_ray_samples = 12
lamp_object.shadow_soft_size = 2.5
lamp_object.shadow_raw_sample_method = 'Adaptive QMC'
lamp_object.shadow_adaptive_threshold = 0.601
lamp_object.sky.use_sky = True
lamp_object.energy=1.5
lamp_object.shadow_method='Ray Shadow'
lamp_object.sky.atmosphere_turbidity=3.5


# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)

# Place lamp to a specified location
lamp_object.location = (5.0, 5.0, 5.0)

# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object

# Set render resolution
scene.render.resolution_x = 480
scene.render.resolution_y = 359

# Set camera fov in degrees
scene.camera.data.angle = fov*(pi/180.0)

# Set camera rotation in euler angles
scene.camera.rotation_mode = 'XYZ'
scene.camera.rotation_euler[0] = 0.0*(pi/180.0)
scene.camera.rotation_euler[1] = 0.0*(pi/180.0)
scene.camera.rotation_euler[2] = -30.0*(pi/180.0)

# Set camera translation
scene.camera.location.x = 5.0
scene.camera.location.y = 2.5
scene.camera.location.z = 20.0

scene.render.alpha_mode = 'SKY'
scene.render.use_antialiasing=True
scene.render.antialiasing_samples=8
scene.render.use_shadows=True
scene.render.use_textures=True
scene.render.use_raytrace=True
scene.render.use_sss=False
scene.render.use_envmaps=False
scene.render.layers['RenderLayer'].use_solid=True



# Set Scenes camera and output filename 
#bpy.data.scenes["Scene"].render.file_format = 'PNG'
bpy.data.scenes["Scene"].render.image_settings.file_format = 'PNG'
bpy.data.scenes["Scene"].render.filepath = '//plane'

# Render Scene and store the scene 
bpy.ops.render.render( write_still=True ) 
