case 
#!/bin/bash
# Shell script to install VOC2007

#Installing VOC2007
if [ ! -d /opt/py-faster-rcnn/data ]; then
    mkdir /opt/py-faster-rcnn/data
fi
cd  /opt/py-faster-rcnn/data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar 
wget http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar 
ls *.tar | xargs -i tar xf {} 
tar -xzf selective_search_data.tgz 
mv VOCdevkit VOCdevkit2007 
mv selective_search_data VOCdevkit2007
