#!/usr/bin/env python
# Select patches of a certain size around objects in the images, and
# create training and validation set files.
# Example usage to produce 40x40 pixel chips:  python crop.py 40

def main():
    import os
    from PIL import Image
    import pandas as pd
    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    chip_size = int(sys.argv[1])
    half_chip = chip_size/2
    total = pd.DataFrame()
    for i in range(20): 
        fname = "/data/oirds/DataSet_"+str(i+1)+"/DataSet"+str(i+1)+".xls"
        book = pd.read_excel(io=fname, sheetname=0, parse_cols=[1,2,3,7,8,9,15,46])
        total = total.append(book)

    total = total[['Image Path', 'Image Name', 'Target Number', 
                   'Intersection Polygon', 'Average Target Centroid',
                   'Mode of Target Type', 'Average Target Orientation',
                   'Mode of Image Size']]
    im_sizes = total.iloc[:,7].unique()
    im_sizes.sort()
    print "Image sizes:\n"
    for x in im_sizes:
        print x
    total.to_csv('/data/oirds/datasets.csv')
    total = pd.read_csv('/data/oirds/datasets.csv', index_col=0)

    # Find the names of images with a second target.
    multiples = total[total.iloc[:,2]==2].iloc[:,1]
    # Limit the ratio of vehicle to no-vehicle chips.
    limit = len(total.iloc[:,2]) - len(multiples)
    counter = 0

    if os.path.isdir('/data/oirds/crop'):
        for root, dirs, files in os.walk('/data/oirds/crop', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
            os.rmdir(root)

    if os.path.isdir('/data/oirds/no_car_crop'):
        for root, dirs, files in os.walk('/data/oirds/no_car_crop', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
            os.rmdir(root)

    os.mkdir('/data/oirds/crop')
    os.mkdir('/data/oirds/no_car_crop')

    with open('/data/oirds/train'+str(chip_size)+'.txt', 'w+') as train:
        with open('/data/oirds/val'+str(chip_size)+'.txt', 'w+') as test:
            # Crop around each object.
            for i, ctr in enumerate(total.iloc[:,4]): # Centroid coordinates
                im1 = Image.open('/data/oirds/png/'+total.iloc[i,1][:-3]+'png')
                txt = 'x y\n'+ctr.replace(']','').replace('[','')
                io_txt = StringIO(txt)
                ctr2 = pd.DataFrame.from_csv(io_txt, 
                                             sep=" ", 
                                             parse_dates=False, 
                                             index_col=None
                                             ).apply(int)
                # Crop the image around the vehicles.
                w, h = im1.size
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
                uid = total.iloc[i,1][:-4]+str(total.iloc[i,2])+'_'+str(chip_size)
                fname = '/data/oirds/crop/'+uid+'c.png'
                im2.save('/data/oirds/'fname)
                if i % 5 == 0:
                    test.write(fname+' 1\n')
                else:
                    train.write(fname+' 1\n')
                                
                # # Rotate the image to 12 o'clock orientation.
                # spin = un_img.iloc[i,6]
                # im3 = im2.rotate(spin)
                # im3.save('/data/oirds/rotate/'+uid+' '+spin+'.png')
                
                # Tile the single-object images with "no car" chips.
                if total.iloc[i,1] not in multiples and counter < limit:
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
                            uid = total.iloc[i,1][:-4]+str(code)+'_'+str(chip_size)
                            fname = 'no_car_crop/'+uid+'R.png'
                            im3.save('/data/oirds/'+fname)
                            if counter % 5 == 0:
                                test.write(fname+' 0\n')
                            else:
                                train.write(fname+' 0\n')
                            counter += 1
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
                            code = m*h/chip_size+n
                            uid = total.iloc[i,1][:-4]+str(code)+'_'+str(chip_size)
                            fname = 'no_car_crop/'+uid+'L.png'
                            im3.save('/data/oirds/'+fname)
                            if counter % 5 == 0:
                                test.write(fname+' 0\n')
                            else:
                                train.write(fname+' 0\n')
                            counter += 1

    # bash-3.2$ ls /data/oirds/no_car_crop/ | wc -l
    #    18606
    # bash-3.2$ ls /data/oirds/crop/ | wc -l
    #     1745

if __name__=="__main__":
    main()
