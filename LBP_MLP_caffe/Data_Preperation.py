### core dependency ###
import cv2
import lmdb
import caffe

### system util ###
import os
import sys
import numpy as np

### util ###
from Util import * #train_progress_bar_util, test_progress_bar_util
from LBP_Feature_Extraction import *


#############################
### prepare training data ###
#############################

def prepareData(resize_row, resize_col, cellSize, fileName, _oneHot=False):

	cur_dir = os.getcwd()
	par_dir = os.path.join(cur_dir, os.pardir)

	train_data_x = []  # feature vector to be returned
	train_data_y = []  # label vector to be returned

	if  os.path.exists(os.path.join(cur_dir, fileName+'_data.txt')) \
	    and os.path.exists(os.path.join(cur_dir, fileName+'_label.txt')):
	    # if data are already prepared, then just load it
	    sys.stdout.write("Training data already exists in this directory, load it from the file...")
	    sys.stdout.flush()
	    train_data_x = np.loadtxt(os.path.join(cur_dir, fileName+'_data.txt'), dtype='f')
	    train_data_y = np.loadtxt(os.path.join(cur_dir, fileName+'_label.txt'), dtype=int)
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
	            if _oneHot:
	            	onehot_encode = [0, 0, 0, 0]
	            	onehot_encode[label-1] = 1
	            	train_data_y.append(onehot_encode)
	            else:
	            	train_data_y.append([label])

	    train_data_x = np.array(train_data_x, dtype='f')
	    train_data_y = np.array(train_data_y, dtype=int)
	    np.savetxt(os.path.join(cur_dir, fileName+'_data.txt'), train_data_x, fmt='%f')
	    np.savetxt(os.path.join(cur_dir, fileName+'_label.txt'), train_data_y, fmt='%d')

	sys.stdout.write("\ncomplete!\ntrain feature vector shape: ")
	sys.stdout.flush()
	print(train_data_x.shape)
	sys.stdout.write("train label vector shape: ")
	sys.stdout.flush()
	print(train_data_y.shape)

	return train_data_x, train_data_y


def exportLMDB(data_x, data_y, fileName):

	assert len(data_x) != 0 and len(data_x)==len(data_y) 

	if os.path.exists(os.path.join(os.getcwd(), fileName)):
		print('Target output file: '+' found in the current directory. Continuing... ')
	else:
		n_samples = len(data_y)
		size = data_x.nbytes * 5  # pre-defined a memory of size 5 times bigger than actual size

		dim = len(data_x.shape)
		channels = 1 if dim < 4 else data_x.shape[-3]
		height = 1 if dim < 3 else data_x.shape[-2]
		width = 1 if dim < 2 else data_x.shape[-1]

		env = lmdb.open(os.path.join(os.getcwd(), fileName), map_size=size)
		with env.begin(write=True) as txn:
		    # txn is a Transaction object
		    for i in range(n_samples):
		        datum = caffe.proto.caffe_pb2.Datum()
		        datum.channels = channels
		        datum.height = height
		        datum.width = width
		        datum.data = data_x[i].tobytes()  # or .tostring() if numpy < 1.9
		        datum.label = int(data_y[i]) 
		        str_id = '{:08}'.format(i)

		        # The encode is only essential in Python 3
		        txn.put(str_id.encode('ascii'), datum.SerializeToString())