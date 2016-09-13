#!/bin/bash
# Shell script to install COCO2014
#
# Alter to your desired directory
MY_DIRECTORY="/opt/py-faster-rcnn/data/coco"
MY_API_DIRECTORY="/opt/py-faster-rcnn/data"
if [ ! -d $MY_DIRECTORY ]; then
    mkdir $MY_DIRECTORY
fi
cd  $MY_DIRECTORY
# Clean directory
ls *.tar | xargs -i rm {} 
ls *.zip | xargs -i rm {} 
# Get image and annotation files
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip
wget https://data.vision.ee.ethz.ch/jpont/mcg/SCG-COCO-train2014-boxes.tgz
wget https://data.vision.ee.ethz.ch/jpont/mcg/SCG-COCO-val2014-boxes.tgz
wget https://data.vision.ee.ethz.ch/jpont/mcg/SCG-COCO-test2014-boxes.tgz
wget http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/instances_minival2014.json.zip
FILE=selective_search_data.tgz
wget http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/$FILE
ls *.zip | xargs -i unzip {} 
ls *.tgz | xargs -i tar -xzf {} 
tar -xzf selective_search_data.tgz 
python $MY_DIRECTORY/../lib/datasets/tools/mcg_munge.py /SCG-COCO-train2014-proposals
python $MY_DIRECTORY/../lib/datasets/tools/mcg_munge.py /SCG-COCO-val2014-proposals
python $MY_DIRECTORY/../lib/datasets/tools/mcg_munge.py /SCG-COCO-test2014-proposals
mkdir images
mv train_2014 test_2014 val_2014 images
if [ ! -d $MY_API_DIRECTORY ]; then
    mkdir $MY_API_DIRECTORY
fi
cd  $MY_API_DIRECTORY
wget https://github.com/pdollar/coco/archive/master.zip
unzip master.zip
# Install COCO API
# Dependencies must already be installed, e.g., Cython
cd coco-master/PythonAPI
make
make install
exit 0
