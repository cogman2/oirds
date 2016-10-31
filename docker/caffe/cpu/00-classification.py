import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels,
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
import os

#if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#else:
#    ../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

caffe.set_mode_cpu()
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model,
                model_weights,  # contains the trained weights,
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
print   "mean-subtracted values: [('B', 104.0069879317889), ('G', 116.66876761696767), ('R', 122.6789143406786)]"

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension,
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel,
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255],
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,        # batch size,
                          3,         # 3-channel (BGR) images,
                          227, 227)  # image size is 227x227

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
#plt.imshow(image)
net.blobs['data'].data[...] = transformed_image
output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch,
print 'predicted class is:', output_prob.argmax()

# load ImageNet labels,
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'

#if not os.path.exists(labels_file):
#    ../data/ilsvrc12/get_ilsvrc_aux.sh
    
labels = np.loadtxt(labels_file, str, delimiter='\\t')
    
print 'output label:', labels[output_prob.argmax()]
print   "[(0.31243637, 'n02123045 tabby, tabby cat')     (0.2379719, 'n02123159 tiger cat')     (0.12387239, 'n02124075 Egyptian cat')     (0.10075711, 'n02119022 red fox, Vulpes vulpes')    (0.070957087, 'n02127052 lynx, catamount')]"
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
print zip(output_prob[top_inds], labels[top_inds])
