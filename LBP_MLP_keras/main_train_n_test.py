### core dependency ###
import cv2
import numpy as np
import os
import sys
import argparse

### model util ###
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import SGD
import h5py               # for saving models into .h5 file


### util ###
from Util import * #train_progress_bar_util, test_progress_bar_util
from LBP_Feature_Extraction import *
from train_monitor_callback import Monitor

#######################
### argument parser ###
#######################
ag = argparse.ArgumentParser()
ag.add_argument("-r", "--resize-factor", required=True, help="parameters to resize the input image to desired size")    # format: -r 128,96
ag.add_argument("-c", "--cell-size", required=True, help="parameters to define the cell size")                          # format: -c 8  (row and colume are assume to be the same)
ag.add_argument("-e", "--epoch-num", required=False, help="parameters to specify the number of training epochs")        # format: -e 50 (defualt is 50)
ag.add_argument("-f", "--force-training", required=False, help="force trainig even if pre-trained model is already found, used for continue traning from checkpoint")   # format: -f true (default is None)
args = vars(ag.parse_args())


#############################
### prepare training data ###
#############################
# resize parameters
resize = args["resize_factor"]
resize = resize.split(',')
cellSize = int(args["cell_size"])
resize_row, resize_col = int(resize[0]), int(resize[1]) 
cur_dir = os.getcwd()
par_dir = os.path.join(cur_dir, os.pardir)

train_data_x = []
train_data_y = []
if  os.path.exists(os.path.join(cur_dir, 'LBPFeature_MLP_data.txt')) \
    and os.path.exists(os.path.join(cur_dir, 'LBPFeature_MLP_label.txt')):
    # if data are already prepared, then just load it
    sys.stdout.write("Training data already exists in this directory, load it from the file...")
    sys.stdout.flush()
    train_data_x = np.loadtxt(os.path.join(cur_dir, 'LBPFeature_MLP_data.txt'), dtype='f')
    train_data_y = np.loadtxt(os.path.join(cur_dir, 'LBPFeature_MLP_label.txt'), dtype=int)
else:
    (progress_tracker, FILE_PER_PERCENT) = train_progress_bar_util('train')  # set up the progress bar
    pre_precent = 0
    # compute each feature vector 
    for label in range(1,5):    # directory idx used as tag
        gallery_path = par_dir+'/train/'+str(label)
        imgFiles = listimages(gallery_path)
        for file in imgFiles:
            # update prcenetage
            progress_tracker += 1
            if progress_tracker // FILE_PER_PERCENT > pre_precent:
                pre_precent = progress_tracker // FILE_PER_PERCENT
                sys.stdout.write("=")
                sys.stdout.flush()
                #progress_tracker = 0

            img = cv2.imread(os.path.join(gallery_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            train_data_x.append(computeLFBFeatureVector_rotation_uniform(img, (cellSize, cellSize), size=(resize_row, resize_col), doCrob=False))
            onehot_encode = [0, 0, 0, 0]
            onehot_encode[label-1] = 1
            train_data_y.append(onehot_encode)
    train_data_x = np.array(train_data_x, dtype='f')
    train_data_y = np.array(train_data_y, dtype=int)
    np.savetxt(os.path.join(cur_dir, 'LBPFeature_MLP_data.txt'), train_data_x, fmt='%f')
    np.savetxt(os.path.join(cur_dir, 'LBPFeature_MLP_label.txt'), train_data_y, fmt='%d')

sys.stdout.write("\ncomplete!\ntrain feature vector shape: ")
sys.stdout.flush()
print(train_data_x.shape)
sys.stdout.write("train label vector shape: ")
sys.stdout.flush()
print(train_data_y.shape)



###########################
### now start training  ###
###########################
sys.stdout.write('\nNow start training...')
sys.stdout.flush()

epoch_num = 50
if args["epoch_num"]:
    epoch_num = int(args["epoch_num"])

# prepare the callback method for monitoring training process
monitor_callback = Monitor()

if os.path.exists(os.path.join(cur_dir, 'LBP_MPL_model.h5')):
    # load the model checkpoint
    sys.stdout.write('\nTrained model alreayd exists in the current directory, loading it from the file...\n\n')
    sys.stdout.flush()
    MLP_Model = load_model(os.path.join(cur_dir, 'LBP_MPL_model.h5'))
    
    # continue from checkpoint if -f option is specified with "true"
    if args["force_training"] == "true":
        print("\n\nModel checkpoint found, '-f true' is specified. Resume trainig from the checkpoint...\n\n")
        MLP_Model.fit(train_data_x, train_data_y, epochs=epoch_num, batch_size=10, callbacks=[monitor_callback], verbose=1)
        MLP_Model.save(os.path.join(cur_dir, 'LBP_MPL_model.h5'))
        
else:    
    # build the model
    MLP_Model = Sequential()
    MLP_Model.add(Dense(100, input_dim=9*(resize_row//cellSize)*(resize_col//cellSize), activation='relu'))
    MLP_Model.add(Dense(60, activation='relu'))
    MLP_Model.add(Dense(4, activation='softmax')) 

    # compile the model
    MLP_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train
    MLP_Model.fit(train_data_x, train_data_y, epochs=epoch_num, batch_size=10, callbacks=[monitor_callback], verbose=1)
    MLP_Model.save(os.path.join(cur_dir, 'LBP_MPL_model.h5'))

sys.stdout.write('\nModel preparation comlpleted!')
sys.stdout.flush()


##########################
### now start testing  ###
##########################
test_dir = os.path.join(par_dir, 'test')
for label in range(1,5):    # test directory also has all three label images
    gallery_path = os.path.join(test_dir, str(label))
    testFiles = listimages(gallery_path)
    test_data_x = []
    expect_label = []
    (progress_tracker, FILE_PER_PERCENT) = test_progress_bar_util('test', label)  # set up the progress bar
    pre_precent = 0
    for file in testFiles:
    # update prcenetage
        progress_tracker += 1
        if int(progress_tracker / FILE_PER_PERCENT) > pre_precent:
            pre_precent = int(progress_tracker / FILE_PER_PERCENT)
            sys.stdout.write("=")
            sys.stdout.flush()

        img = cv2.imread(os.path.join(gallery_path, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_data_x.append(computeLFBFeatureVector_rotation_uniform(img, (cellSize, cellSize), size=(resize_row, resize_col), doCrob=False))
        onehot_encode = [0, 0, 0, 0]
        onehot_encode[label-1] = 1
        expect_label.append(onehot_encode)

    test_data_x = np.array(test_data_x, dtype='f')
    expect_label = np.array(expect_label, dtype=int)

    sys.stdout.write("\ntesting images in folder %s ..." % os.path.join(test_dir, str(label)))
    sys.stdout.write("\ntest feature vector shape: ")
    sys.stdout.flush()
    print(test_data_x.shape)
    sys.stdout.write("expect label vector shape: ")
    sys.stdout.flush()
    print(expect_label.shape)

    scores = MLP_Model.evaluate(test_data_x, expect_label)
    print("\n%s: %.2f%%" % (MLP_Model.metrics_names[1], scores[1]*100))
    print('---------------------')

raw_input('press "enter" to exit the program.')

