#
# A script to use the PythonAPI to load and run the coco dataset on 
# the Faster R-CNN
#
import os, sys
import pickle
# set oirds root etc.
OIRDS_ROOT='/home/robertsneddon/oirds/' 
COCO_ROOT='/data2/MS_COCO/'
TOOLS_ROOT=OIRDS_ROOT+'tools/fcn/python/'                    # Path to python tools for Caffe deep learning
A_ROOT=COCO_ROOT+'annotations/'                              # Path to annotations files
A_FILE='instances_train2014.json'                            # Specific annotation file
sys.path.append(COCO_ROOT+'coco/PythonAPI/')
sys.path.append(COCO_ROOT+'coco/PythonAPI/pycocotools/')
sys.path.append(TOOLS_ROOT)
from coco import *
from gt_tool import *

def load_data(bool):
    if bool: cc_tool = COCO(A_ROOT+A_FILE)
    else: cc_tool = pickle.load( open("coco_object_save.p", "rb") )
    return cc_tool

def save_data(coco_object):
    pickle.dump(coco_object, open( "coco_object_save.p", "wb") )
