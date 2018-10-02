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
from Define_Model_Solver import *


#######################
### argument parser ###
#######################
ag = argparse.ArgumentParser()
ag.add_argument("-r", "--resize-factor", required=True, help="parameters to resize the input image to desired size")    # format: -r 128,96
ag.add_argument("-c", "--cell-size", required=True, help="parameters to define the cell size")                          # format: -c 8  (row and colume are assume to be the same)
ag.add_argument("-o", "--model-name", required=True, help="specify output model name")
ag.add_argument("-e", "--epoch-num", required=False, help="parameters to specify the number of training epochs")        # format: -e 50 (defualt is 50)
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
modelName = args["model_name"]+'.prototxt'
modelPath = os.path.join(os.getcwd(), modelName)

### prepare data
train_data_x, train_data_y = prepareData(resize_row, resize_col, cellSize, modelName+'_train', _oneHot=False)
exportLMDB(train_data_x, train_data_y, modelName+'_train_data_lmdb')

### prepare model and sovler
with open(modelPath, 'w') as f:
    f.write(str(LBP_MLP(os.path.join(os.getcwd(), modelName+'_train_data_lmdb'), 100)))
solver_path = Solver(modelPath)
solver = caffe.get_solver(solver_path)

### training loop
for e in range(epochs):
	solver.step(1	)