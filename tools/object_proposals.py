#!/usr/bin python
# Generate object proposals for the CNN using a sliding window of size 40x40 on a 
# 256x256 pixel image.
# plt.Rectangle((bbox[0], bbox[1]),   # bottom-left corner      # 0 -> x_min, 1 -> y_min
#               bbox[2] - bbox[0],    # distance to right edge  # 2 -> x_max
#               bbox[3] - bbox[1]     # distance to top edge    # 3 -> y_max
import numpy as np
lst = []

for i in range(216*216):
    #       x_min  y_min  x_max     y_max
    lst.append([i%216, i/216, i%216+40, i/216+40])

mat = np.array(lst)
np.savetxt('/data/object_proposals/oirds.csv', mat, '%3.0f')
