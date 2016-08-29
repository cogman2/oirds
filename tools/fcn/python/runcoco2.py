#
# A script to use the PythonAPI to load and run the coco dataset on 
# the Faster R-CNN
#
import sys
#COCO_ROOT='/data2/MS_COCO/'
COCO_ROOT='/opt/py-faster-rcnn/data/'
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

# set oirds root etc.


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
    if bool: cc_tool = COCO(A_ROOT+A_FILE)
    else: cc_tool = pickle.load( open("coco_object_save.p", "rb") )
    return cc_tool

def save_data(coco_object):
    pickle.dump(coco_object, open( "coco_object_save.p", "wb") )

#targets = ['person','dog','skateboard']
#targets = ['car']
'''
coco=load_data(True);
cocoDt = coco.loadRes(resFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
my_poly = anns[0]['segmentation']
my_bb = anns[0]['bbox']
my_category = anns[0]['category_id']
'''
config = json.load(open(configFile))
gttool = GTTool(config)
df_info = gttool.loadCoco()
#print df_info
