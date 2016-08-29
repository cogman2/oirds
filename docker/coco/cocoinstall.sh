#!/bin/bash
# Shell script to install COCO2014
#
# Alter to your desired directory
MY_DIRECTORY="/opt/py-faster-rcnn/data"
MY_API_DIRECTORY="/opt/py-faster-rcnn/data/coco"
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
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip
ls *.tar | xargs -i tar xf {} 
ls *.zip | xargs -i unzip {} 
tar -xzf selective_search_data.tgz 
if [ ! -d $MY_API_DIRECTORY ]; then
    mkdir $MY_API_DIRECTORY
fi
cd  $MY_API_DIRECTORY
# Install COCO API
wget https://github.com/pdollar/coco/archive/master.zip
unzip master.zip
cd coco-master/PythonAPI
make
make install
exit 0