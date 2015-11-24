#!/usr/bin/env python
def main():
  # Example usage:  python fine_tune.py <patch size> <resize>

  # # Fine-tuning a Pretrained Network for Style Recognition
  # 
  # In this example, we'll explore a common approach that is particularly useful in real-world applications: take a pre-trained Caffe network and fine-tune the parameters on your custom data.
  # 
  # The upside of such approach is that, since pre-trained networks are learned on a large set of images, the intermediate layers capture the "semantics" of the general visual appearance. Think of it as a very powerful feature that you can treat as a black box. On top of that, only a few layers will be needed to obtain a very good performance of the data.

  # First, we will need to prepare the data. This involves the following parts:
  # (1) Get the ImageNet ilsvrc pretrained model with the provided shell scripts.
  # (3) Compile the downloaded the OIRDS dataset into a database that Caffe can then consume.

  import os
  os.chdir('/opt/caffe')
  import sys
  sys.path.append('/opt/caffe/python')

  import matplotlib
  matplotlib.use('Agg')
  import caffe
  import numpy as np
  from pylab import *
  import subprocess

  # This downloads the ilsvrc auxiliary data (mean file, etc),
  # and a subset of 2000 images for the style recognition task.
  # subprocess.call(['data/ilsvrc12/get_ilsvrc_aux.sh'])
  # subprocess.call(['scripts/download_model_binary.py',
  #                  'models/bvlc_reference_caffenet'
  #                  ])


  # For your record, if you want to train the network in pure C++ tools, here is the command:
  # 
  # <code>
  # build/tools/caffe train \
  #     -solver models/finetune_flickr_style/solver.prototxt \
  #     -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  #     -gpu 0
  # </code>
  # 
  # However, we will train using Python in this example.

  niter = 200
  # losses will also be stored in the log
  train_loss = np.zeros(niter)
  scratch_train_loss = np.zeros(niter)

  caffe.set_device(0)
  caffe.set_mode_gpu()
  patch_size = sys.argv[1]
  subprocess.call(['tools/oirds/crop_finetune.py', patch_size])


  if not patch_size=='40':
    network = 'models/oirds/finetune_train_val'+patch_size+'.prototxt'
    with open('models/oirds/finetune_train_val40.prototxt', 'r') as f:
      with open(network, 'w+') as tv_file:
        for line in f:
          # Change the train.txt and val.txt filenames.
          if '40.txt' in line:
            tv_file.write(line.replace('40', patch_size))
          else:
            tv_file.write(line)

    proto_solve = 'models/oirds/finetune_solver'+patch_size+'.prototxt'
    with open('models/oirds/finetune_solver40.prototxt', 'r') as g:
      with open(proto_solve, 'w+') as solver_file:
        for line in g:
          # Change the train-val prototxt filename.
          if 'net:' in line:
            solver_file.write(line.replace('40', patch_size))
          # Change the snapshot prefix.
          elif 'snapshot_prefix:' in line:
            solver_file.write(line.replace('40', patch_size))
          else:
            solver_file.write(line)

  # We create a solver that fine-tunes from a previously trained network.
  solver = caffe.SGDSolver(proto_solve)
  solver.net.copy_from('git/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
  # For reference, we also create a solver that does no finetuning.
  scratch_solver = caffe.SGDSolver(proto_solve)

  # We run the solver for niter times, and record the training loss.
  for it in range(niter):
      solver.step(1)  # SGD by Caffe
      scratch_solver.step(1)
      # store the train loss
      train_loss[it] = solver.net.blobs['loss'].data
      scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
      if it % 10 == 0:
          print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])
  print 'done'


  # Let's look at the training loss produced by the two training procedures respectively.

  # In[5]:

  plot(np.vstack([train_loss, scratch_train_loss]).T)


  # Notice how the fine-tuning procedure produces a more smooth loss function change, and ends up at a better loss. A closer look at small values, clipping to avoid showing too large loss during training:

  # In[6]:

  plot(np.vstack([train_loss, scratch_train_loss]).clip(0, 4).T)


  # Let's take a look at the testing accuracy after running 200 iterations. Note that we are running a classification task of 5 classes, thus a chance accuracy is 20%. As we will reasonably expect, the finetuning result will be much better than the one from training from scratch. Let's see.

  # In[7]:

  test_iters = 200 # 10
  accuracy = 0
  scratch_accuracy = 0
  for it in arange(test_iters):
      solver.test_nets[0].forward()
      accuracy += solver.test_nets[0].blobs['accuracy'].data
      scratch_solver.test_nets[0].forward()
      scratch_accuracy += scratch_solver.test_nets[0].blobs['accuracy'].data
  accuracy /= test_iters
  scratch_accuracy /= test_iters
  
  print 'Accuracy for fine-tuning on '+patch_size+'x'+patch_size+' patches: '+str(accuracy)
  print 'Accuracy for training from scratch: '+str(scratch_accuracy)
  
  logfile = '/opt/caffe/tools/oirds/logfile-finetune.csv'

  if not os.path.isfile(logfile):
    with open(logfile, 'w') as h:
      # the header
      h.write('patch_size,accuracy,scratch_accuracy')

  with open(logfile, 'a') as i:
    i.write(patch_size+','+str(accuracy)+','+str(scratch_accuracy))

  # Huzzah! So we did finetuning and it is awesome. Let's take a look at what kind of results we are able to get with a longer, more complete run of the style recognition dataset. Note: the below URL might be occassionally down because it is run on a research machine.
  # 
  # http://demo.vislab.berkeleyvision.org/

if __name__=='__main__':
    main()
