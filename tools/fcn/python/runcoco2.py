#
# A script to use the PythonAPI to load and run the coco dataset on 
# the Faster R-CNN
#
import sys
# Just adding for testing
#COCO_ROOT='/data2/MS_COCO/'
COCO_ROOT='/opt/py-faster-rcnn/data/coco/'
#OIRDS_ROOT='/home/robertsneddon/oirds/' 
OIRDS_ROOT='/opt/py-faster-rcnn/oirds/' 
TOOLS_ROOT=OIRDS_ROOT+'tools/fcn/python/'                    # Path to python tools for Caffe deep learning
#sys.path.append(COCO_ROOT+'coco/PythonAPI/')
#sys.path.append(COCO_ROOT+'coco/PythonAPI/pycocotools/')
sys.path.append('/opt/py-faster-rcnn/data/coco-master/PythonAPI')
sys.path.append('/opt/py-faster-rcnn/data/coco-master/PythonAPI/pycocotools/')
sys.path.append(TOOLS_ROOT)
import pickle
import pudb
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from gt_tool import *


A_ROOT=COCO_ROOT+'annotations/'                              # Path to annotations files
R_ROOT=COCO_ROOT+'results/'
A_FILE='instances_train2014.json'                            # Specific annotation file
#A_FILE='instances_val2014.json'                            # Specific annotation file
annType = ['segm','bbox']
annType = annType[1]
dataDir=COCO_ROOT+'images'
dataType='val2014'
resFile='%s/results/instances_%s_fake%s100_results.json'
resFile=resFile%(dataDir, dataType, annType)
configFile=OIRDS_ROOT+'example_config/create_coco_db.json'

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def load_data(bool):
    if bool: 
        cc_tool = COCO(A_ROOT+A_FILE)
    else:
        cc_tool = pickle.load(open(COCO_ROOT+'coco_object.p', 'rb'))
    return cc_tool

def save_data(coco_object):
    pickle.dump(coco_object, open(COCO_ROOT+'coco_object.p', 'wb' ))

def load_df(bool):
    if bool:
        df_info = gttool.loadCoco()
    else:
        df_info = pickle.load(open(COCO_ROOT+'df_info.p', 'rb' ))
    return df_info

def save_df(df_info):
    pickle.dump(df_info, open(COCO_ROOT+'df_info.p', 'wb' ))






#targets = ['person','dog','skateboard']
#targets = ['car']

config = json.load(open(configFile))
gttool = GTTool(config)
df_info = load_df(True)

