### core dependency ###
import cv2
import lmdb
import caffe

### system util ###
import os
import sys
import numpy as np
import argparse

### util ###
from Data_Preperation import *
from Model_and_Solver import *


#######################
### argument parser ###
#######################
ag = argparse.ArgumentParser()
ag.add_argument("-r", "--resize-factor", required=True, help="parameters to resize the input image to desired size")    # format: -r 128,96
ag.add_argument("-c", "--cell-size", required=True, help="parameters to define the cell size")                          # format: -c 8  (row and colume are assume to be the same)
ag.add_argument("-o", "--model-name", required=True, help="specify output model name")
ag.add_argument("-e", "--epoch-num", required=True, help="parameters to specify the number of training epochs")        # format: -e 50 (defualt is 50)
ag.add_argument("-f", "--force-training", required=False, help="force trainig even if pre-trained model is already found, used for continue traning from checkpoint")   # format: -f true (default is None)
args = vars(ag.parse_args())

# number of epoch
epochs = int(args["epoch_num"])

# resize parameters
resize = args["resize_factor"]
resize = resize.split(',')
cellSize = int(args["cell_size"])
resize_row, resize_col = int(resize[0]), int(resize[1]) 

# model path
modelName = args["model_name"]
trainModel = os.path.join(os.getcwd(), modelName+'_train.prototxt')
testModel = os.path.join(os.getcwd(), modelName+'_test.prototxt')


### prepare data
train_data_x, train_data_y = prepareData('train', resize_row, resize_col, cellSize, modelName, _oneHot=False)
trainData = exportH5PY(train_data_x, train_data_y, modelName+'_train_data_path')

test_data_x, test_data_y = prepareData('test', resize_row, resize_col, cellSize, modelName, _oneHot=False)
testData = exportH5PY(test_data_x, test_data_y, modelName+'_test_data_path')

### prepare model and sovler
with open(trainModel, 'w') as f:
    f.write(str(LBP_MLP(trainData, 100, 'train')))

with open(testModel, 'w') as f:
    f.write(str(LBP_MLP(testData, 10, 'test')))

solver_path = Solver(trainModel, testModel)
solver = caffe.get_solver(solver_path)

### training loop

monitor = Monitor()
for e in range(epochs):
    print("starting new epoch...")
    solver.step(100)

    print('epoch: ', e, 'testing...')
    #print(solver.net.blobs['loss'].data)
    loss = solver.net.blobs['loss'].data[()]
    #print(loss.shape, type(loss))
    correct = 0
    for test_it in range(10):
        solver.test_nets[0].forward()
        #correct += solver.test_nets[0].blobs['accuracy'].data
        #correct += sum(solver.test_nets[0].blobs['prob'].data.argmax(1)
        #               == solver.test_nets[0].blobs['label'].data.reshape(1, -1))
        #print(solver.test_nets[0].blobs['accuracy'].data)
        print(solver.test_nets[0].blobs['prob'].data)
        #print(solver.test_nets[0].blobs['score'].data.argmax(1))
        #print(solver.test_nets[0].blobs['label'].data.reshape(1, -1))
        
        print(solver.test_nets[0].blobs['prob'].data.argmax(1))
        print(solver.test_nets[0].blobs['label'].data.reshape(1, -1))
        #print(sum(solver.test_nets[0].blobs['prob'].data.argmax(1)
        #               == solver.test_nets[0].blobs['label'].data.squeeze()))
        correct += sum(solver.test_nets[0].blobs['prob'].data.argmax(1)
                       == solver.test_nets[0].blobs['label'].data.squeeze())
        #print(sum(solver.test_nets[0].blobs['score'].data == solver.test_nets[0].blobs['label'].data).reshape(-1, 1))
    accuracy = correct/100
    #accuracy = solver.test_nets[0].blobs['accuracy'].data
    #print(monitor.accuracy)
    monitor.update(loss, accuracy)
    #print(monitor.losses)
    #print(type(accuracy))
    #print("current accuracy: %f" % accuracy) 