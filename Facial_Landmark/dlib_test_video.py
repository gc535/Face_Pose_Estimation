import cv2
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import dlib

### config params ###
faces = 1
eyes_pair = 1


### argument parser ###
ag = argparse.ArgumentParser()
ag.add_argument("-p", "--shape-predictor", required=True, help="path to face landmark predictor")
ag.add_argument("-i", "--image", required=False, help="[OPTIONAL] path to image")
args = vars(ag.parse_args())

### initialize detectors
#face_cascade = cv2.CascadeClassifier('/home/brandon/OpenCV/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_alt.xml')
#eye_cascasde = cv2.CascadeClassifier('/home/brandon/OpenCV/opencv-3.4.1/data/haarcascades/haarcascade_eye.xml')
#mouth_cascade = cv2.CascadeClassifier('/home/brandon/DeepCam/pose_estimation/Mouth.xml')
predictor = dlib.shape_predictor(args["shape_predictor"])
detector = dlib.get_frontal_face_detector()

### utils ###
# convert rect obj to opencv bounding box
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

# find desired 6 landmark and convert its cordinate(x,y) into np array
def select_and_shape_to_np(landmarks, dtype="int"):
    # initialize the list of (x, y)-coordinates
    landmarks_six = np.zeros((6, 2), dtype=float)
    landmarks_indices = [36, 45, 30, 48, 54, 8] #https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    """
    {
        0: (x, y)   # left eye corner 
        1: (x, y)   # right eye corner 
        2: (x, y)   # nose 
        3: (x, y)   # left mouth corner 
        4: (x, y)   # right mouth corner 
        5: (x, y)   # chin   
    }
    """
    for (i, landmark_idx) in enumerate(landmarks_indices):
        landmarks_six[i] = (landmarks.part(landmark_idx).x, landmarks.part(landmark_idx).y)
 
    # return the list of (x, y)-coordinates
    return landmarks_six


### 3D Model Coordinates ###
"""
Model Coordinates: points used to describe the coordinates of the 3D modelin some arbitrary reference 
frame / coordinate system
"""
# 3D model points.
model_points = np.array( [  (-225.0, 170.0, -135.0),       # Left eye left corner
                            (225.0, 170.0, -135.0),        # Right eye right corne
                            (0.0, 0.0, 0.0),               # Nose tip
                            (-150.0, -150.0, -125.0),      # Left Mouth corner
                            (150.0, -150.0, -125.0),       # Right mouth corner
                            (0.0, -330.0, -65.0)  ])       # Chin

"""
### get image frame from img###
frame = cv2.imread('pose.jpg')
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
"""

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ### Camera Internals ###
        """
        This section use estimated params for camera interals: x_focal_len, y_focal_len, principle_coor(x, y)
        Since those params vary on the kinds of used camera, the estimated result is just a closed guess and
        is NOT expected to be accurate enough for application that requires high accuracy.
        ESTIMATION: 
            focal_lens are estimated to be the height and width of the image
            principle_coor is the center of the image
        """
        x_focal_len, y_focal_len = len(frame_gray[0]), len(frame_gray)
        principle_coor = (x_focal_len/2, y_focal_len/2)
        # put into matrix format
        camera_params = np.array([ [x_focal_len, 0, principle_coor[0]],
                                   [0, y_focal_len, principle_coor[1]],
                                   [0, 0, 1]], 
                                   dtype = "double")                       

        ### detection and predict landmarks ###
        face_rgns = detector(frame_gray, 1)
        for (i, face_rgn) in enumerate(face_rgns):
            # apply landmark detector on face region
            landmarks = predictor(frame_gray, face_rgn)
            landmarks_six = select_and_shape_to_np(landmarks)
            
            # find face region and draw bouding box
            (x, y, w, h) = rect_to_bb(face_rgn)
            #face_rgn_gray = frame_gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # mark out facial landmarks
            for (x, y) in landmarks_six:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            #print "Camera Matrix :\n {0}".format(camera_params)
         
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, landmarks_six, camera_params, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
         
            print "Rotation Vector:\n {0}".format(rotation_vector)  #[0]:probably pitch?  [1]: probably roll?  [2]: probably yaw?
            #print "Translation Vector:\n {0}".format(translation_vector)
         
         
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
         
         
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_params, dist_coeffs)
            p1 = ( int(landmarks_six[2][0]), int(landmarks_six[2][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
         
            cv2.line(frame, p1, p2, (255,0,0), 2)

            
            
        cv2.imshow('local cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


video_capture.release()
cv2.destoryAllWindows()