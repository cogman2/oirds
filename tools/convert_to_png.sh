#!/bin/bash
# Use ImageMagick to convert files to PNGs.
cd /data/OIRDS
for i in `seq 1 20`
do
    cd DataSet_$i
    for j in $(ls *tif | sed 's/....$//')
    do
	convert $j.tif ../train/png/$i_$j.png
    done
    cd ..
done