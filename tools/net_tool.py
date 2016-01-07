import gt_tool
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
#%matplotlib inline
import lmdb
import caffe
from PIL import Image
import numpy as np

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


def loadNet(config):
   from caffe.proto import caffe_pb2
   import json_tools
   net = caffe.Net(json_tools.getProtoTxt(config),json_tools.getModelFileName(config), caffe.TEST)
   interp_layers = [k for k in net.params.keys() if 'up' in k]
   if (json_tools.isNetSurgery(config)):
     interp_surgery(net, interp_layers)
   return net

def runcaffe (net, im, config):
   import numpy
   import json_tools
   from caffe.proto import caffe_pb2

   meanarr = json_tools.getMeanArray(config)

   transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
   rescaleFactor=255.0
   transformer.set_mean('data', meanarr/rescaleFactor)
   transformer.set_transpose('data', (2,0,1))
   ## RGB -> BGR ?
   transformer.set_channel_swap('data', (2,1,0))
   transformer.set_raw_scale('data', 1.0/rescaleFactor)

   img = (im.transpose(2,0,1) - meanarr[(2,1,0),:,:])/rescaleFactor
   img = img[(2,1,0),:,:]
   img = img[numpy.newaxis,:,:,:]
   caffe_in = img

   net.blobs['data'].reshape(caffe_in.shape[0],caffe_in.shape[1],caffe_in.shape[2],caffe_in.shape[3])
   caffe_in2 =  transformer.preprocess('data', im)[np.newaxis,:,:,:]
   return (net.forward_all(data=np.asarray([caffe_in])), net.blobs['data'].data[0], transformer)
  # net.blobs['data'].data[...] = transformer.preprocess('data', im)
  # return outputResult(net.forward(), transformer, net.blobs['data'].data[0],im, name)  

def dumpNetWeights(net):
  for ll in net.blobs:
    try:
      filters = net.params[ll][0].data
      vis_filter(ll, filters)
    except:
      continue

def vis_filter(name, data, padsize=1, padval=0):
  wc = data.shape[0]/64;
  for pltNum in range(0,wc):
    vis_square(name + str(pltNum), data[pltNum*data.shape[0]:(pltNum+1)*data.shape[0],:], padsize,padval)
# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)   
def vis_square(name, data, padsize=1, padval=0):
    data = data[:,0:3,:,:]
    data = data.transpose(0, 2, 3, 1)
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data);
    plt.savefig(name + 'plt')
    plt.close()

def convertImage (im):
   tmp= np.array(im)
   return tmp
