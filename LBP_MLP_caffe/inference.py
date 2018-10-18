import caffe
import cv2

from LBP_Feature_Extraction import *


def inference(net, imgPath):
    
    resize = 96
    cellsize = 16

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    #load the image in the data layer
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = computeLFBFeatureVector_rotation_uniform(img, (cellsize, cellsize), size=(resize, resize), doCrob=False)
    net.blobs['data'].data[...] = transformer.preprocess('data', feature)

    # predict
    out = net.forward()
    return out['prob'].argmax()

#load the model
net = caffe.Net('LBP_MLP_deploy.prototxt',
                'LBP_MLP_iter_3199.caffemodel',
                caffe.TEST)

imgPath = 'sample_test/242.jpg'
inference(net, imgPath)