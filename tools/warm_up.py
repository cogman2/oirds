#!/usr/bin/env python
# Example usage:  python fine_tune.py <patch size> <resize>
def main():
  import os, sys, subprocess
  sys.path.append("/opt/caffe/python")
  import matplotlib
  # Facilitate headless usage.
  matplotlib.use('Agg')
  import caffe
  
  #=================================================================
  # 0 - Segment the OIRDS dataset into single-object test and train
  # sets (train.txt and val.txt) with the desired chip size.
  #=================================================================
  # insert crop.py here

  #=================================================================
  # 1 - Create the OIRDS lmdb inputs
  # N.B. set the path to the OIRDS train + val data dirs
  # Use glogs.py or this to check the glogs:
  # ls -t /tmp | grep $USER.log.INFO.201510 | xargs cat
  #=================================================================
  example='/opt/caffe/git/caffe/examples'
  data='/data/oirds'
  tools='/opt/caffe/bin'
  train_data_root=data+'/'
  val_data_root=data+'/'

  # Set resize=true to resize the images to 256x256. Leave as false if images have
  # already been resized using another tool.
  resize=False
  if resize:
    resize_height=256
    resize_width=256
  else:
    resize_height=0
    resize_width=0

  patch_size = sys.argv[1]
  db_train = '/oirds_train'+patch_size+'_lmdb'
  db_val = '/oirds_val'+patch_size+'_lmdb'
  if os.path.isdir(example+db_train):
    for root, dirs, files in os.walk(example+db_train, topdown=False):
      for name in files:
        os.remove(os.path.join(root, name))
      for name in dirs:
        os.rmdir(os.path.join(root, name))
      os.rmdir(root)

  if os.path.isdir(example+db_val):
    for root, dirs, files in os.walk(example+db_val, topdown=False):
      for name in files:
        os.remove(os.path.join(root, name))
      for name in dirs:
        os.rmdir(os.path.join(root, name))
      os.rmdir(root)

  print 'Creating train lmdb...'

  # rootfolder/ listfile db_name
  subprocess.call(['glog_logtostderr=1',
                   tools+'/convert_imageset',
                   train_data_root,
                   data+'/train'+patch_size+'.txt',
                   example+db_train
                   # '--resize_height='+resize_height,
                   # '--resize_width='+resize_width,
                   # '--shuffle'
                   ])

  print "Creating val lmdb..."

  subprocess.call(['glog_logtostderr=1',
                   tools+'/convert_imageset',
                   val_data_root,
                   data+'/val'+patch_size+'.txt',
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
    
if __name__=='__main__':
  main()
