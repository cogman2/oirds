#!/bin/bash
# Shell script to install COCO2014
#
# Alter to your desired directory
MY_DIR="/opt/py-faster-rcnn/data/coco"
MY_PROP_DIR="/opt/py-faster-rcnn/data/coco_proposals"
MY_ANN_DIR="/opt/py-faster-rcnn/data/coco/annotations"
MY_API_DIR="/opt/py-faster-rcnn/data"

# Check for directory
if [ ! -d $MY_DIR ]; then
    mkdir $MY_DIR
fi
cd  $MY_DIR

# Clean directory
ls *.tar | xargs -i rm {} 
ls *.zip | xargs -i rm {} 

# Get image files
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
ls *.zip | xargs -i unzip {} 
mkdir images
mv train_2014 test_2014 val_2014 images
ls *.zip | xargs -i rm {}

# Get annotation files
if [ ! -d $MY_ANN_DIR ]; then
    mkdir $MY_ANN_DIR
fi
cd  $MY_ANN_DIR
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip
wget http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/instances_minival2014.json.zip
ls *.zip | xargs -i unzip {} 
ls *.zip | xargs -i rm {}

# Get proposals/MCG boxes data
if [ ! -d $MY_PROP_DIR ]; then
    mkdir $MY_PROP_DIR
fi
cd  $MY_PROP_DIR
wget https://data.vision.ee.ethz.ch/jpont/mcg/SCG-COCO-train2014-boxes.tgz
wget https://data.vision.ee.ethz.ch/jpont/mcg/SCG-COCO-val2014-boxes.tgz
wget https://data.vision.ee.ethz.ch/jpont/mcg/SCG-COCO-test2014-boxes.tgz
ls *.tgz | xargs -i tar -xzf {} 
ls *.tgz | xargs -i rm {}
python $MY_DIR/../lib/datasets/tools/mcg_munge.py ./SCG-COCO-train2014-boxes
python $MY_DIR/../lib/datasets/tools/mcg_munge.py ./SCG-COCO-val2014-boxes
python $MY_DIR/../lib/datasets/tools/mcg_munge.py ./SCG-COCO-test2014-boxes

# This directory change needs to be checked
if [ ! -d $MY_PROP_DIR/selective_search ]; then
    mkdir $MY_PROP_DIR/selective_search
fi
cd  $MY_PROP_DIR
mv $MY_PROP_DIR/MCG $MY_PROP_DIR/selective_search

# Removing the original boxes data. Uncomment if you're sure this is correct
# rm -r *-boxes

# Selective search data for ImageNet, Uncomment and choose appropriate directory if needed.
#FILE=selective_search_data.tgz
#wget http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/$FILE
#tar -xzf selective_search_data.tgz 

if [ ! -d $MY_API_DIR ]; then
    mkdir $MY_API_DIR
fi
cd  $MY_API_DIR
wget https://github.com/pdollar/coco/archive/master.zip
unzip master.zip
# Install COCO API
# Dependencies must already be installed, e.g., Cython
cd coco-master/PythonAPI
make
make install
cd ../..
rm master.zip
exit 0