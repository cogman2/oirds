#!/usr/bin python
import os
from PIL import Image
import pandas as pd
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

# 1, 2, 3 = "Image Path", "Image Name", "Target Number"
# 7, 8 = "Intersection Polygon", "Average Target Centroid"
# 9, 15 = "Mode of Target Type", "Average Target Orientation"

# total = pd.DataFrame()
# for i in range(20): 
#     fname = "/data/OIRDS/DataSet_"+str(i+1)+"/DataSet"+str(i+1)+".xls"
#     book = pd.read_excel(io=fname, sheetname=0, parse_cols=[1,2,3,7,8,9,15])
#     total = total.append(book)

# total.to_csv('/data/OIRDS/datasets.csv')
total = pd.read_csv('/data/OIRDS/datasets.csv', index_col=0)

# Find the names of images with a second target.
multiples = total[total.iloc[:,2]==2].iloc[:,1]

# os.mkdir('/data/OIRDS/train/crop')
# os.mkdir('/data/OIRDS/train/no_car_crop')
with open('/data/OIRDS/train/train.txt', 'w+') as train:
    with open('/data/OIRDS/train/val.txt', 'w+') as test:
        # Crop around each object.
        for i, ctr in enumerate(total.iloc[:,4]): # Centroid coordinates
            im1 = Image.open('/data/OIRDS/train/png/'+total.iloc[i,1][:-3]+'png')
            txt = 'x y\n'+ctr.replace(']','').replace('[','')
            io_txt = StringIO(txt)
            ctr2 = pd.DataFrame.from_csv(io_txt, 
                                         sep=" ", 
                                         parse_dates=False, 
                                         index_col=None
                                         ).apply(int)
            # Crop the image to 40x40 pixels.
            w, h = im1.size
            chip_size = 40
            half_chip = chip_size/2
            ctr_x,ctr_y = ctr2.iloc[0],ctr2.iloc[1]
            # The distance from the top-left corner to the 
            #   left edge, top edge, right edge and bottom edge.
            l,u,r,low = ctr_x-half_chip,ctr_y-half_chip,ctr_x+half_chip,ctr_y+half_chip
            if ctr_x < half_chip:
                l,r = 0,chip_size
                
            if ctr_x > w-half_chip:
                l,r = w-chip_size,w
        
            if ctr_y < half_chip:
                u,low = 0,chip_size
        
            if ctr_y > h-half_chip:
                u,low = h-chip_size,h
        
            im2 = im1.crop((l,u,r,low))
            uid = total.iloc[i,1][:-4]+str(total.iloc[i,2])
            im2.save('/data/OIRDS/train/crop/'+uid+'.png')
            # # Rotate the image to 12 o'clock orientation.
            # im3 = im2.rotate(un_img.iloc[i,6])
            # im3.save('/data/OIRDS/train/rotate/'+un_img.iloc[i,1][:-3]+'png')
            
            # Tile the single-object images with "no car" chips.
            if total.iloc[i,1] not in multiples:
                # Tile to the right.
                # x
                for j in range((w-ctr_x-half_chip)/chip_size):
                    # y
                    for k in range(h/chip_size):
                        right = w - j*chip_size
                        upper = k*chip_size
                        left = right - chip_size
                        lower = upper + chip_size
                        im3 = im1.crop((left, upper, right, lower))
                        code = j*h/chip_size+k
                        uid = total.iloc[i,1][:-4]+str(code)
                        fname = '/data/OIRDS/train/no_car_crop/'+uid+'R.png'
                        im3.save(fname)
                        if code % 5 == 0:
                            test.write(fname+' 0')
                        else:
                            train.write(fname+' 1')
                # Tile to the left.
                # x
                for m in range((ctr_x-half_chip)/chip_size):
                    # y
                    for n in range(h/chip_size):
                        left = m*chip_size
                        right = left + chip_size
                        upper = n*chip_size
                        lower = upper + chip_size
                        im3 = im1.crop((left, upper, right, lower))
                        code = j*h/chip_size+k
                        uid = total.iloc[i,1][:-4]+str(code)
                        fname = '/data/OIRDS/train/no_car_crop/'+uid+'L.png'
                        im3.save(fname)
                        if code % 5 == 0:
                            test.write(fname+' 0')
                        else:
                            train.write(fname+' 1')

# bash-3.2$ ls /data/OIRDS/train/no_car_crop/ | wc -l
#    18606
# bash-3.2$ ls /data/OIRDS/train/crop/ | wc -l
#     1745

