#!/usr/bin/env python

# Deepdream neural network visualization
# Description in this blogpost:
#
# http://googleresearch.blogspot.se/2015/07/deepdream-code-example-for-visualizing.html
#
# Code based on the deepdream ipython notebook: https://github.com/google/deepdream
# I just converted it to "pure" python and played around with it a bit.
#
# Images are saved in the frames directory. Create it before starting this script.
# Warning. If there already are images in there they will be overwritten.
# Consider yourself warned.
#
# Albert Veli
# Boomtime, the 46th day of Confusion in the YOLD 3181

import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

# Compile caffe, then put caffe/python in your PYTHONPATH
# See http://caffe.berkeleyvision.org/installation.html
#
# In my case I have caffe directly under deepdream so I
# add it using __file__ path (path of the running script).
# Change the 3 rows below if you have caffe somewhere else.
import os.path
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/caffe/python')

# And hope it all worked out
import caffe


# --------------------------------------------------------
# -------------        Customization         -------------
# --------------------------------------------------------
# Here are some things you might want to play around with

imgname = 'pirri.jpg'
# List available layers with something like:
# cat caffe/models/bvlc_googlenet/deploy.prototxt | grep 'name:'
end_layer = 'inception_3b/output'
# Scale coefficient, 0.02 = zoom in 2 percent each frame
scale = 0.01
frames = 200 # Number of frames to calculate

# No need to change much below this row. Mostly from dream.ipynb.
# --------------------------------------------------------


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# Show array on screen
def showarray(a):
    a = np.uint8(np.clip(a, 0, 255))
    im = PIL.Image.fromarray(a)
    im.show()

# Save frame to disk
def saveframe(frame, num):
    # Save in dir frames under deepdream
    d = os.path.dirname(os.path.realpath(__file__)) + '/frames'
    PIL.Image.fromarray(np.uint8(frame)).save("%s/%04d.jpg" % (d, num))
    return num + 1

# Scale frame using scale coefficient scale and return scaled frame.
def scale_frame(frame, scale, w, h):
    return nd.affine_transform(frame, [1-scale,1-scale,1], [h*scale/2,w*scale/2,0], order=1)

modelname = 'bvlc_googlenet'
#modelname = 'bvlc_alexnet'
# Subsitute with your path here
model_path = os.path.dirname(os.path.realpath(__file__)) + '/caffe/models/' + modelname + '/'
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + modelname + '.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

imgpath = os.path.dirname(os.path.realpath(__file__)) + '/' + imgname
img = np.float32(PIL.Image.open(imgpath))
frame = img
frame_i = 0
frame_i = saveframe(frame, frame_i)
h, w = frame.shape[:2]

src, dst = net.blobs['data'], net.blobs[end_layer]
src.reshape(1,3,h,w)
src.data[0] = preprocess(net, img)
net.forward(end=end_layer)
img_features = dst.data[0].copy()

def objective_L2(dst):
    dst.diff[:] = dst.data

def make_step(net, step_size=1.5, end='inception_4c/output',
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            #vis = deprocess(net, src.data[0])
            #if not clip: # adjust image contrast if clipping is disabled
            #    vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            print octave, i, end #, vis.shape
            #clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

def objective_guide(dst):
    global img_features
    x = dst.data[0].copy()
    y = img_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

# Loop through frames number of frames
# and zoom in scale percent each frame
for i in xrange(frames):
    frame = deepdream(net, frame, end=end_layer, objective=objective_guide)
    #showarray(frame)
    frame_i = saveframe(frame, frame_i)
    frame = scale_frame(frame, scale, w, h)
