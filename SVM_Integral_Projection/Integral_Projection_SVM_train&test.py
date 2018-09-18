import cv2
import Image 
import os
import sys
import numpy as np

dynamic_range = 255   # 8-bit grayscale
cur_dir = os.getcwd()


def horizontal_projection(img):
    hp = []
    for c in range(len(img[0])):
        col_sum = 0
        for r in range(len(img)):
            col_sum += img[r][c]
        hp.append(col_sum/dynamic_range)
    return hp

def vertial_projection(img):
    return [sum(img[r])/dynamic_range for r in range(len(img))]

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
def progress_bar_util(action):
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
    FILE_PER_PERCENT = TOTAL_NUM_FILES / 40 
    progress_tracker = 0
    return (progress_tracker, FILE_PER_PERCENT)


#############################
### prepare training data ###
#############################
train_data_x = []
train_data_y = []
if  os.path.exists(os.path.join(cur_dir, 'integral_projection_data.txt')) \
    and os.path.exists(os.path.join(cur_dir, 'integral_projection_label.txt')):
    # if data are already prepared, then just load it
    sys.stdout.write("Training data already exists in this directory, load it from the file...")
    sys.stdout.flush()
    train_data_x = np.loadtxt(os.path.join(cur_dir, 'integral_projection_data.txt'), dtype='f')
    train_data_y = np.loadtxt(os.path.join(cur_dir, 'integral_projection_label.txt'), dtype=int)
else:
    (progress_tracker, FILE_PER_PERCENT) = progress_bar_util('train')  # set up the progress bar
    # compute each feature vector 
    for label in range(1,4):    # directory idx used as tag
        gallery_path = cur_dir+'/train/'+str(label)
        imgFiles = listimages(gallery_path)
        for file in imgFiles:
            # update prcenetage
            progress_tracker += 1
            if progress_tracker == FILE_PER_PERCENT:
                sys.stdout.write("-")
                sys.stdout.flush()
                progress_tracker = 0

            img = cv2.imread(os.path.join(gallery_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (w, h) = img.shape
            #print('image shape(h, w) %d * %d now resizing to 128 * 64' % (h, w))
            resized_img = cv2.resize(img, (128, 64))
            train_data_x.append(vertial_projection(resized_img) + horizontal_projection(resized_img))
            train_data_y.append(label)
    train_data_x = np.array(train_data_x, dtype='f')
    train_data_y = np.array(train_data_y, dtype=int)
    np.savetxt(os.path.join(cur_dir, 'integral_projection_data.txt'), train_data_x, fmt='%f')
    np.savetxt(os.path.join(cur_dir, 'integral_projection_label.txt'), train_data_y, fmt='%d')

sys.stdout.write("\ntrain feature vector shape: ")
sys.stdout.flush()
print(train_data_x.shape)
sys.stdout.write("train label vector shape: ")
sys.stdout.flush()
print(train_data_y.shape)



##############################
### now start the training ###
##############################
#svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                    svm_type = cv2.SVM_C_SVC,
#                    C=2.67, gamma=5.383 )
if os.path.exists(os.path.join(cur_dir, 'Integral_Projection_Model_SVM.dat')):
    # trained model already exist, load the model
    sys.stdout.write('Trained model found, load it from file...')
    sys.stdout.flush()
    SVM_MODEL = cv2.ml.SVM_load(os.path.join(cur_dir, 'Integral_Projection_Model_SVM.dat'))
else:    
    SVM_MODEL = cv2.ml.SVM_create()
    SVM_MODEL.setKernel(cv2.ml.SVM_LINEAR)
    SVM_MODEL.setType(cv2.ml.SVM_C_SVC)
    SVM_MODEL.setC(2.67)
    SVM_MODEL.setGamma(5.383)
    SVM_MODEL.train(train_data_x, cv2.ml.ROW_SAMPLE, train_data_y)
    SVM_MODEL.save('Integral_Projection_Model_SVM.dat')


#########################
### now start testing ###
#########################
test_data_x = []
expect_label = []

if  os.path.exists(os.path.join(cur_dir, 'test_data.txt')) \
    and os.path.exists(os.path.join(cur_dir, 'test_label.txt')):
    # if data are already prepared, then just load it
    sys.stdout.write("Test data already exists in this directory, load it from the file...")
    sys.stdout.flush()
    test_data_x = np.loadtxt(os.path.join(cur_dir, 'test_data.txt'), dtype='f')
    expect_label = np.loadtxt(os.path.join(cur_dir, 'test_label.txt'), dtype=int)
else:
    (progress_tracker, FILE_PER_PERCENT) = progress_bar_util('test')  # set up the progress bar
    test_dir = os.path.join(cur_dir, 'test')
    for label in range(1,4):    # test directory also has all three label images
        gallery_path = os.path.join(test_dir, str(label))
        testFiles = listimages(gallery_path)
        for file in testFiles:
        # update prcenetage
            progress_tracker += 1
            if progress_tracker == FILE_PER_PERCENT:
                sys.stdout.write("-")
                sys.stdout.flush()
                progress_tracker = 0

            img = cv2.imread(os.path.join(gallery_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (w, h) = img.shape
            #print('image shape(h, w) %d * %d now resizing to 128 * 64' % (h, w))
            resized_img = cv2.resize(img, (128, 64))
            test_data_x.append(vertial_projection(resized_img) + horizontal_projection(resized_img))
            expect_label.append(label)
    test_data_x = np.array(test_data_x, dtype='f')
    expect_label = np.array(expect_label, dtype=int)
    np.savetxt(os.path.join(cur_dir, 'test_data.txt'), test_data_x, fmt='%f')
    np.savetxt(os.path.join(cur_dir, 'test_label.txt'), expect_label, fmt='%d')


test_result = np.squeeze(np.array(SVM_MODEL.predict(test_data_x)[1], dtype=int)) 

sys.stdout.write("\ntest feature vector shape: ")
sys.stdout.flush()
print(test_data_x.shape)
sys.stdout.write("expect label vector shape: ")
sys.stdout.flush()
print(expect_label.shape)
sys.stdout.write("test result vector shape: ")
sys.stdout.flush()
print(test_result.shape)
print('---------------------')

mask = test_result==expect_label
correct = np.count_nonzero(mask)*100.0/test_result.size  # calculate percentage accuracy
print('Model accuracy: %f%%' % correct)



