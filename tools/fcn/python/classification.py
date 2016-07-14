import numpy as np
import matplotlib.pyplot as plt
import pudb
# %matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '/opt/py-faster-rcnn/caffe-fast-rcnn/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
'''    
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print(\Downloading pre-trained CaffeNet model...\)
    !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
'''
#caffe.set_mode_cpu()
caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt', \
    caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',caffe.TEST)
    
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
# set net to batch size of 50,
net.blobs['data'].reshape(50,3,227,227)
# REDO?? 
#net.blobs['data'].reshape(3,227,277)
# redo net.blobs['data'].reshape(50,227,3,3) ?? 
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
out = net.forward()
print "Predicted class is: ",out['prob'][0].argmax()
