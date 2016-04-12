import pickle

config={'gthost':'http://172.31.72.136:8080/geoserver/SA/wms','gtlayer':'SA:water_overlay','uname':'roberteb','passwd':'','connectid':'c19b1c96-cd25-453a-96ef-4cd223c6d2c0', 'box':'-42.608732,-23.010076,-40.966275,-21.970913','profile':'Aerial_CIR_Profile','landshapefile':'/home/rwgdrummer/oirds/land-polygons-complete-4326/land_polygons'}
f = open("c.p","wb")
pickle.dump(config,f)
f.close()
