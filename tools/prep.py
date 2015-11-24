#!/usr/bin/env python
#=================================================================
#  Select patches of a certain size around objects in the images, 
# and create training and validation set files.  This approach
# puts all the images in the same folders, and depends on the .txt
# files to keep them separate (though filesnames also distinguish
# crop sizes and measures of rotation).
# 
# Example usage: sudo ./prep.py <crop size> <rotate>
# 
# crop size .......... dimensions in pixels (int)
# rotate ............. should the images be rotated? (bool)
#=================================================================

# Create a csv from the xls files.
def load_xl(data_dir):
    total = pd.DataFrame()
    for i in range(20): 
        fname = data_dir+'/DataSet_'+str(i+1)+'/DataSet'+str(i+1)+'.xls'
        book = pd.read_excel(io=fname, sheetname=0, parse_cols=[1,2,3,7,8,9,15,46,47])
        total = total.append(book)

    total = total[['Image Path', 'Image Name', 'Target Number', 
                   'Intersection Polygon', 'Average Target Centroid',
                   'Mode of Target Type', 'Average Target Orientation',
                   'Mode of Image Size', 'Average GSD']]
    im_sizes = total.iloc[:,7].unique()
    im_sizes.sort()
    print 'Image sizes:\n'
    for x in im_sizes:
        print x
    total.to_csv(data_dir+'/datasets.csv')
    

# rm -rf a folder
def rmrf(folder):
    if os.path.isdir(folder):
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
            os.rmdir(root)


def main():
    import os
    from PIL import Image
    import pandas as pd
    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    subprocess.call(['gsd_fix.py'])
    try:
        chip_size = int(sys.argv[1])
        half_chip = chip_size/2
        # rmrf(data+'/no_car_crop')
        os.mkdir(data+'/no_car_crop')

        # rmrf(data+'/crop')
        os.mkdir(data+'/crop')

        trainfile = data+'/train'+str(chip_size)+'.txt'
        testfile = data+'/val'+str(chip_size)+'.txt'

    except ValueError:
        crop = False
        chip_size = ''

    data = '/data/oirds'
    load_xl(data)
    total = pd.read_csv(data+'/datasets.csv', index_col=0)

    # Find the names of images with a second target.
    multiples = total[total.iloc[:,2]==2].iloc[:,1]
    # Limit the number of no-vehicle chips to the number of vehicle chips.
    train_test_proportion = 5 # a 4 to 1 ratio
    limit = len(total.iloc[:,2]) / train_test_proportion
    counter = 0

    rotate = sys.argv[2] in ['True', 'true', 'y', 'Y', 'Yes', 'yes']

    with open(trainfile, 'w+') as train:
        with open(testfile, 'w+') as test:
            # Loop over the vehicles.
            for i, ctr in enumerate(total.iloc[:,4]): # Centroid coordinates
                original = total.iloc[i,1][:-3]+'png'
                im1 = Image.open(data+'/png'+original)
                uid = total.iloc[i,1][:-4]+str(total.iloc[i,2])+'_'+str(chip_size)

                # Rotate the image.
                if rotate==True:
                    copyfile(original, data+'/rotate'+original
                    for i in range(15, 180, 15):
                        img = im1.rotate(i)
                        fname = data+'/rotate/'+uid+'_'+str(i)+'.png'
                        img.save(fname)
                        if i % train_test_proportion == 0:
                            test.write(fname+' 1\n')
                        else:
                            train.write(fname+' 1\n')
                        
                if crop = True:
                    # Crop the image around the vehicles.
                    txt = 'x y\n'+ctr.replace(']','').replace('[','')
                    io_txt = StringIO(txt)
                    ctr2 = pd.DataFrame.from_csv(io_txt, 
                                                 sep=" ", 
                                                 parse_dates=False, 
                                                 index_col=None
                                                 ).apply(int)

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
                    fname = data+'/crop/'+uid+'_0.png'
                    im2.save(fname)
                    if i % train_test_proportion == 0:
                        test.write(fname+' 1\n')
                    else:
                        train.write(fname+' 1\n')
                                    

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
                                uid = total.iloc[i,1][:-4]+str(code)+'_'+str(chip_size)
                                fname = data+'/no_car_crop/'+uid+'R.png'
                                im3.save(fname)
                                if counter < limit:
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
                                fname = data+'/no_car_crop/'+uid+'L.png'
                                im3.save(fname)
                                if counter < limit:
                                    test.write(fname+' 0\n')
                                else:
                                    train.write(fname+' 0\n')
                                counter += 1


if __name__=="__main__":
    main()
