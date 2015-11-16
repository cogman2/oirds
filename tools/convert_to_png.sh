#!/bin/bash
# Use ImageMagick to convert files to PNGs.
cd /data/oirds
mkdir png
for i in `seq 1 20`
do
    cd DataSet_$i
    for j in $(ls *tif | sed 's/....$//')
    do
	convert $j.tif ../png #/$i_$j.png
    done
    cd ..
done