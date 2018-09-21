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
### argument parser ###
ag = argparse.ArgumentParser()
ag.add_argument("-r", "--resize-factor", required=True, help="parameters to resize the input image to desired size")
ag.add_argument("-c", "--cell-size", required=True, help="parameters to define the cell size")
args = vars(ag.parse_args())


###################################################################
### methods below this line are used for LBP feature extraction ###
###################################################################
def findEquivalentRotation(value_256):
    equivalent_rotation = [0, 1, 1, 3, 1, 5, 3, 7, 1, 9, 5, 11, 3, 13, 7, 15, 1, 17, 9, 19, 5,
                           21, 11, 23, 3, 25, 13, 27, 7, 29, 15, 31, 1, 9, 17, 25, 9, 37, 19, 39, 5, 37, 21, 43, 11, 45,
                           23, 47, 3, 19, 25, 51, 13, 53, 27, 55, 7, 39, 29, 59, 15, 61, 31, 63, 1, 5, 9, 13, 17, 21, 25,
                           29, 9, 37, 37, 45, 19, 53, 39, 61, 5, 21, 37, 53, 21, 85, 43, 87, 11, 43, 45, 91, 23, 87, 47, 95,
                           3, 11, 19, 27, 25, 43, 51, 59, 13, 45, 53, 91, 27, 91, 55, 111, 7, 23, 39, 55, 29, 87, 59, 119, 15,
                           47, 61, 111, 31, 95, 63, 127, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 9, 25, 37,
                           39, 37, 43, 45, 47, 19, 51, 53, 55, 39, 59, 61, 63, 5, 13, 21, 29, 37, 45, 53, 61, 21, 53, 85,
                           87, 43, 91, 87, 95, 11, 27, 43, 59, 45, 91, 91, 111, 23, 55, 87, 119, 47, 111, 95, 127, 3,
                           7, 11, 15, 19, 23, 27, 31, 25, 39, 43, 47, 51, 55, 59, 63, 13, 29, 45, 61, 53, 87, 91, 95, 27, 59,
                           91, 111, 55, 119, 111, 127, 7, 15, 23, 31, 39, 47, 55, 63, 29, 61, 87, 95, 59, 111, 119, 127, 15, 31, 47, 63,
                           61, 95, 111, 127, 31, 63, 95, 127, 63, 127, 127, 255]

    return equivalent_rotation[value_256]


def findUniform(rotation_invariant):
    uniform = [1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               25, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 
               43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58]

    if   uniform[rotation_invariant] == 1:  return 1
    elif uniform[rotation_invariant] == 2:  return 2
    elif uniform[rotation_invariant] == 4:  return 3
    elif uniform[rotation_invariant] == 7:  return 4
    elif uniform[rotation_invariant] == 11: return 5
    elif uniform[rotation_invariant] == 16: return 6
    elif uniform[rotation_invariant] == 22: return 7
    elif uniform[rotation_invariant] == 29: return 8
    elif uniform[rotation_invariant] == 58: return 9
    else:                                   return 0
 

def computeLBPImage_rotation_uniform(srcImg):
    LBPImage = np.zeros(srcImg.shape, dtype=int)
    pad_srcImg = cv2.copyMakeBorder(srcImg, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

    for r in range(len(srcImg)):
        for c in range(len(srcImg[r])):
            LBP_value = 0
            if(pad_srcImg[r-1][c-1] >= pad_srcImg[r][c]): LBP_value |= 1<<7
            if(pad_srcImg[r-1][c] >= pad_srcImg[r][c]):   LBP_value |= 1<<6
            if(pad_srcImg[r-1][c+1] >= pad_srcImg[r][c]): LBP_value |= 1<<5
            if(pad_srcImg[r][c+1] >= pad_srcImg[r][c]):   LBP_value |= 1<<4
            if(pad_srcImg[r+1][c+1] >= pad_srcImg[r][c]): LBP_value |= 1<<3
            if(pad_srcImg[r+1][c] >= pad_srcImg[r][c]):   LBP_value |= 1<<2
            if(pad_srcImg[r+1][c-1] >= pad_srcImg[r][c]): LBP_value |= 1<<1
            if(pad_srcImg[r][c-1] >= pad_srcImg[r][c]):   LBP_value |= 1
            LBPImage[r][c] = findUniform(findEquivalentRotation(LBP_value))

    return LBPImage

def computeLFBFeatureVector_rotation_uniform(inputImg, cellSize, size, doCrob):
    if doCrob:
        x, y = int(len(LBPImage[0])/5.0 + 0.5), int(len(LBPImage)/4.0)
        w = h = int(len(LBPImage[0])*0.6 + 0.5)
        if y+h > len(LBPImage): h = len(LBPImage) - y
        srcImg = inputImg[y:y+h, x:x+w]
        srcImg = cv2.resize(srcImg, inputImg.shape)
    else: srcImg = cv2.resize(inputImg, size)

    LBPImage = computeLBPImage_rotation_uniform(srcImg)
    cellHeight, cellWidth = cellSize[0], cellSize[1]
    numOfCell_x, numOfCell_y = len(LBPImage[0])//cellWidth, len(LBPImage)//cellHeight

    feature_vector = np.zeros(9*numOfCell_x*numOfCell_y, dtype='f')

    for row_cell in range(numOfCell_y):
        for col_cell in range(numOfCell_x):
            vector_offset = (row_cell*numOfCell_x + col_cell) * 9
            total_count = 0
            for cell_r in range(cellHeight):
                for cell_c in range(cellWidth):
                    if LBPImage[row_cell*cellHeight+cell_r][col_cell*cellWidth+cell_c]:
                        feature_vector[vector_offset + LBPImage[row_cell*cellHeight+cell_r][col_cell*cellWidth+cell_c]-1] += 1
                        total_count += 1
            for hist_bin in range(9):
                feature_vector[vector_offset + hist_bin] /= float(total_count)

    return feature_vector
###################################################################
### methods below this line are used for LBP feature extraction ###
###################################################################

# load all images from current directory
def listimages(img_dir):
    if not os.path.exists(img_dir):
        raise Exception("image dir is not exists")
    files = []
    if os.path.isfile(img_dir):
        files.append(img_dir)
    else:
        files = os.listdir(img_dir)

    if len(files) < 1:
        raise Exception("No picture needs to be detected")
    return files


######### This is util method for percentage bar display #######
def train_progress_bar_util(action):
    cur_dir = os.getcwd()
    toolbar_width = 40
    TOTAL_NUM_FILES = 0
    for label in range(1,4):    # directory idx used as tag
        gallery_path = os.path.join(cur_dir, action, str(label))
        imgFiles = listimages(gallery_path)
        TOTAL_NUM_FILES += len(imgFiles)
    sys.stdout.write("preparing %sing data...\n" % action)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    FILE_PER_PERCENT = TOTAL_NUM_FILES / 40.0 
    progress_tracker = 0

    return (progress_tracker, FILE_PER_PERCENT)

def test_progress_bar_util(action, label):
    cur_dir = os.getcwd()
    toolbar_width = 40
    gallery_path = os.path.join(cur_dir, action, str(label))
    imgFiles = listimages(gallery_path)
    TOTAL_NUM_FILES = len(imgFiles)
    sys.stdout.write("\npreparing %sing data...\n" % action)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    FILE_PER_PERCENT = TOTAL_NUM_FILES / 40.0
    progress_tracker = 0
    return (progress_tracker, FILE_PER_PERCENT)

#############################
### prepare training data ###
#############################
# resize parameters
resize = args["resize_factor"]
resize = resize.split(',')
cellSize = int(args["cell_size"])
resize_row, resize_col = int(resize[0]), int(resize[1]) 
cur_dir = os.getcwd()

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
    for label in range(1,4):    # directory idx used as tag
        gallery_path = cur_dir+'/train/'+str(label)
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
            #(w, h) = img.shape
            #print('image shape(h, w) %d * %d now resizing to 128 * 64' % (h, w))
            #resized_img = cv2.resize(img, (128, 64))
            train_data_x.append(computeLFBFeatureVector_rotation_uniform(img, (cellSize, cellSize), size=(resize_row, resize_col), doCrob=False))
            onehot_encode = [0, 0, 0]
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
if os.path.exists(os.path.join(cur_dir, 'LBP_MPL_model.h5')):
    sys.stdout.write('\nTrained model alreayd exists in the current directory, loading it from the file...')
    sys.stdout.flush()
    MLP_Model = load_model(os.path.join(cur_dir, 'LBP_MPL_model.h5'))
else:    
    # build the model
    MLP_Model = Sequential()
    MLP_Model.add(Dense(300, input_dim=9*(resize_row//cellSize)*(resize_col//cellSize), activation='relu'))
    MLP_Model.add(Dense(100, activation='relu'))
    MLP_Model.add(Dense(60, activation='relu'))
    MLP_Model.add(Dense(3, activation='softmax')) 

    # compile the model
    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    MLP_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train
    MLP_Model.fit(train_data_x, train_data_y, epochs=5, batch_size=10)
    MLP_Model.save(os.path.join(cur_dir, 'LBP_MPL_model.h5'))

sys.stdout.write('\nModel preparation comlpleted!')
sys.stdout.flush()


###########################
### now start training  ###
###########################
test_dir = os.path.join(cur_dir, 'test')
for label in range(1,4):    # test directory also has all three label images
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
        test_data_x.append(computeLFBFeatureVector_rotation_uniform(img, (8, 8), size=(resize_row, resize_col), doCrob=False))
        onehot_encode = [0, 0, 0]
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

