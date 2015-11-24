#!/usr/bin/env python
# Ground sample distance normalization
#  - Zooming in to the minimum, instead of out to the maximum
# np.unique(tst.values) = [ 0.0862,  0.1524,  0.3   ]

def main():
    import os
    import pandas as pd
    import numpy as np
    from PIL import Image
    
    total = pd.read_csv('/data/oirds/datasets.csv', index_col=0)
    
    val = np.unique(total.iloc[:,8].values)
    base = val.min()

    os.mkdir('/data/oirds/zoom')
    for i, gsd in enumerate(total.iloc[:,8]):
        original = total.iloc[i,1][:-3]+'png'
        infile = '/data/oirds/png/'+original
        outfile = '/data/oirds/zoom/'+original
        try:
            im = Image.open(infile)
            size = im.size * base / gsd
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(outfile, "JPEG")
        except IOError:
            print "cannot zoom for '%s'" % infile


if __name__=='__main__':
    main()
