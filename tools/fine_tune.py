#!/usr/bin/env python
# Example usage:  python fine_tune.py <patch size> <resize>
def main():
  import os, sys, subprocess
  sys.path.append("/opt/caffe/python")
  import caffe
  #=================================================================
  # 0 - Segment the OIRDS dataset into single-object test and train
  # sets (train.txt and val.txt) with the desired chip size.
  #=================================================================
  <insert crop.py here>

  #=================================================================
  # 1 - Create the OIRDS lmdb inputs
  # N.B. set the path to the OIRDS train + val data dirs
  # Use glogs.py or this to check the glogs:
  # ls -t /tmp | grep $USER.log.INFO.201510 | xargs cat
  #=================================================================
  example='/opt/caffe/examples/oirds'
  data='/data/OIRDS/train'
  tools='/opt/caffe/build/tools'
  train_data_root=data+'/'
  val_data_root=data+'/'

  # Set resize=true to resize the images to 256x256. Leave as false if images have
  # already been resized using another tool.
  resize=false
  if resize:
    resize_height=256
    resize_width=256
  else:
    resize_height=0
    resize_width=0

  patch_size = sys.argv[1]
  db_train = '/train_'+patch_size+'_lmdb'
  db_val = '/val_'+patch_size+'_lmdb'
  if os.path.isdir(example+db_train):
    os.rmdir(example+db_train)
  if os.path.isdir(db_val):
    os.rmdir(db_val)
  print "Creating train lmdb..."

  # rootfolder/ listfile db_name
  subprocess.call(['glog_logtostderr=1',
                   tools+'/convert_imageset',
                   train_data_root,
                   data+'/train.txt',
                   example+db_train
                   # '--resize_height='+resize_height,
                   # '--resize_width='+resize_width,
                   # '--shuffle'
                   ])

  print "Creating val lmdb..."

  subprocess.call(['glog_logtostderr=1',
                   tools+'/convert_imageset',
                   val_data_root,
                   data+'/val.txt',
                   example+db_val
                   # '--resize_height='+resize_height,
                   # '--resize_width='+resize_width,
                   # '--shuffle'
                   ])

  print "Lightning Memory-mapped Database creation done."

  #=================================================================
  # 2 - Compute the mean image from the oirds training lmdb
  # N.B. this is available in data/oirds
  #=================================================================
  train_mean = '/mean_'+patch_size+'.binaryproto'
  subprocess.call([tools+'/compute_image_mean',
                   example+'/train'+db_suffix,
                   data+train_mean
                   ])

  print "Mean image done"

  #=================================================================
  # 3 - The network definition
  #     Builds prototxt files from previous versions.
  #=================================================================
  os.chdir('/opt/caffe/models/finetune_flickr_style/')
  network = 'train_val_'+patch_size+'.prototxt'
  with open('train_val.prototxt', 'r') as f:
    with open(network, 'w+') as train:
      for line in f:
        if 'train_lmdb' in line:
          train.write(line[:-18]+db_train)
        elif 'oirds_mean' in line:
          train.write(line[:-24]+train_mean)
        elif 'val_lmdb' in line:
          train.write(line[:-16]+db_val)
        else:
          train.write(line)

  solver = 'solver_'+patch_size+'.prototxt'
  with open('solver.prototxt', 'r') as g:
    with open(solver, 'w+') as test:
      for line in g:
        if 'net:' in line:
          test.write(line[:-27]+solver+'\"')
        elif 'max_iter:' in line:
          test.write('max_iter: 50')
        elif 'prefix:' in line:
          test.write(line[:-14]+'train_'+patch_size)
        else:
          test.write(line)

  #=================================================================
  # 4 - Train a CNN model on OIRDS.
  #=================================================================
  subprocess.call([tools+'/caffe train',
                   '-solver '+solver,
                   '-weights 
                   '-gpu 0'
                   ])
    
if __name__='__main__':
  main()
