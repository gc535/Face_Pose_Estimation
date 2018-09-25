#### This program used the LBP feature of human face image as the input feature vector to train a MLP network for face pose estimation. 
This program combines the training process and the testing process.

All training data and testing data are included in the upper directory.

Dependency: keras (tensorflow, or other DL backend), opencv-python, h5py


#### run the program using:
```
python main_train_n_test.py -r 128,96 -c 16 [[optional] -e 50] [[optional] -f true]  
```
-r: resize image

-c cell size

-e number of epoch to train

-f continue training from pre-trained checkpoint


Current Model config:

    1st layer: input-dim=9*NumOfCell output-dim=100     activation=relu
    2nd layer:                       output-dim=60     activation=relu
    3rd layer:                       output-dim=3       activation=softmax
    

Current Training Config:

    optimizer: adam  (embeded learning rate, momentum, decay)
    loss function: categorical_crossentropy
    epoch, batch = 180, 10

Current Accuracy:

    frontal face ~= 100%
    less rotated ~= 100%
    more rotated ~= 100%
    none face    ~= 99.75%


  
          
