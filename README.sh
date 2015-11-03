#!/usr/bin/env sh
#=================================================================
# Tools for training, testing, and compressing Fast R-CNN networks.

#=================================================================
# 0 - Segment the OIRDS dataset into single-object test and train
# sets (train.txt and val.txt)
#=================================================================
python /opt/fast-rcnn/tools/oirds/segment.py

#=================================================================
# 1 - Create the OIRDS lmdb inputs
# N.B. set the path to the OIRDS train + val data dirs
# Use this to check the glogs:
# ls -t /tmp | grep $USER.log.INFO.201510 | xargs cat
#=================================================================
EXAMPLE=/opt/fast-rcnn/examples/oirds
DATA=/data/OIRDS/train/crop
TOOLS=/opt/fast-rcnn/caffe-fast-rcnn/build/tools
TRAIN_DATA_ROOT=$DATA/
VAL_DATA_ROOT=$DATA/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_oirds.sh to the path" \
       "where the Oirds training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_oirds.sh to the path" \
       "where the Oirds validation data is stored."
  exit 1
fi

rm -rf $EXAMPLE/oirds_train_lmdb $EXAMPLE/oirds_val_lmdb
echo "Creating train lmdb..."

# ROOTFOLDER/ LISTFILE DB_NAME
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/oirds_train_lmdb
     # --resize_height=$RESIZE_HEIGHT \
     # --resize_width=$RESIZE_WIDTH \
     # --shuffle \

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/oirds_val_lmdb
    # --resize_height=$RESIZE_HEIGHT \
    # --resize_width=$RESIZE_WIDTH \
    # --shuffle \

echo "Lightning Memory-mapped Database creation done."

# cf., train_net.py

#=================================================================
# 2 - Compute the mean image from the oirds training lmdb
# N.B. this is available in data/oirds
#=================================================================
$TOOLS/compute_image_mean $EXAMPLE/oirds_train_lmdb \
  $DATA/oirds_mean.binaryproto

echo "Mean image done."

#=================================================================
# 3 - The network definition
#=================================================================
# emacs /home/benjaminhess/oirds/train_val.prototxt
# emacs /home/benjaminhess/oirds/solver.prototxt

#=================================================================
# 4 - Train an OIRDS CNN model on CaffeNet.
# I changed the number of outputs at the end to 5.
#=================================================================
$CAFFE_HOME/build/tools/caffe train -solver /opt/fast-rcnn/models/CaffeNet/oirds/single_solver2.prototxt -gpu 0