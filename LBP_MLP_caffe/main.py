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
ag.add_argument("-tb", "--train-batch-size", required=False, help="parameters to define the batch size")					# format: -tb 300  (300 samples per iteration)
ag.add_argument("-vb", "--test-batch-size", required=False, help="parameters to define the batch size")					# format: -vb 100  (300 samples per iteration)
ag.add_argument("-o", "--model-name", required=True, help="specify output model name")									# format: -o LBP_MLP	
ag.add_argument("-e", "--epoch-num", required=True, help="parameters to specify the number of training epochs")         # format: -e 50 (defualt is 50)
ag.add_argument("-f", "--force-training", required=False, help="force trainig even if pre-trained model is already found, used for continue traning from checkpoint")   # format: -f true (default is None)
args = vars(ag.parse_args())

# training parameters
train_batch_size = 300
test_batch_size = 100
if args["train_batch_size"]:
	train_batch_size = int(args["train_batch_size"])
if args["test_batch_size"]:
	test_batch_size = int(args["test_batch_size"])
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
deployModel = os.path.join(os.getcwd(), modelName+'_deploy.prototxt')


### prepare data
train_data_x, train_data_y = prepareData('train', resize_row, resize_col, cellSize, modelName, _oneHot=False)
trainData = exportH5PY(train_data_x, train_data_y, modelName+'_train_data')

test_data_x, test_data_y = prepareData('test', resize_row, resize_col, cellSize, modelName, _oneHot=False)
testData = exportH5PY(test_data_x, test_data_y, modelName+'_test_data')

epoch_size = len(train_data_x)

### prepare model and sovler
caffe.set_mode_cpu()
#caffe.set_device(0)
#caffe.set_mode_gpu()

# train model
with open(trainModel, 'w') as f:
    f.write(str(LBP_MLP(trainData, train_batch_size, 'train')))

# test model
with open(testModel, 'w') as f:
    f.write(str(LBP_MLP(testData, test_batch_size, 'test')))

# inference model for deploying
with open(deployModel, 'w') as f:
    f.write(str(LBP_MLP(testData, test_batch_size, 'inference')))

# solver to train all models above
solver_path = Solver(trainModel, testModel, epoch_size, train_batch_size, epochs)
solver = caffe.get_solver(solver_path)

### training loop
monitor = Monitor()
for e in range(epochs):
    print("starting new epoch...")
    solver.step(epoch_size//train_batch_size)

    print('epoch: ', e, 'testing...')
    loss = solver.net.blobs['loss'].data[()]
    correct = 0
    for test_it in range(10):
        solver.test_nets[0].forward()
        correct += sum(solver.test_nets[0].blobs['prob'].data.argmax(1)
                       == solver.test_nets[0].blobs['label'].data.squeeze())
    print(correct)
    accuracy = correct/test_batch_size/10
    #accuracy = solver.test_nets[0].blobs['accuracy'].data
    #print(monitor.accuracy)
    monitor.update(loss, accuracy)
    print("current accuracy: %f" % accuracy) 

input('press "enter" to exit the program.')