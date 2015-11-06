#!/usr/bin/env python
# Generate object proposals for the CNN using a given sliding window size on a 
# 256x256 pixel image.
# plt.Rectangle((bbox[0], bbox[1]),   # bottom-left corner      # 0 -> x_min, 1 -> y_min
#               bbox[2] - bbox[0],    # distance to right edge  # 2 -> x_max
#               bbox[3] - bbox[1]     # distance to top edge    # 3 -> y_max
def main():
    import numpy as np
    import sys
    stride = 1
    im_h = 256
    im_w = 256
    patch_h = sys.argv[1]
    patch_w = patch_h
    x_steps = im_w - patch_w
    y_steps = im_h - patch_h
    patch_num = y_steps * x_steps

    lst = []
    for i in range(patch_num):
        #           x_min      y_min      x_max         y_max
        lst.append([i%x_steps, i/y_steps, i%x_steps+40, i/y_steps+40])

    mat = np.array(lst)
    np.savetxt('/data/object_proposals/oirds.csv', mat, '%3.0f')

if __name__=="__main__":
    main()
