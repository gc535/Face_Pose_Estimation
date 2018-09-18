import cv2
import tensorflow as tf
import numpy as np
import os
import sys

faces = 1
eyes_pair = 1


face_cascade = cv2.CascadeClassifier('/home/brandon/OpenCV/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascasde = cv2.CascadeClassifier('/home/brandon/OpenCV/opencv-3.4.1/data/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/home/brandon/DeepCam/pose_estimation/Mouth.xml')

frame = cv2.imread('pose.jpg')
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# face region: [x, y, width, height] 
#  => lower left: (x, y) upper right: (x+w, y+h)
faces_pos = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
for (x, y, w, h) in faces_pos:
    # find face region and draw bouding box
    face_rgn = frame[y:y+h, x:x+w]
    face_rgn_gray = frame_gray[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # find eyes and draw bounding boxes
    eye_pos = eye_cascasde.detectMultiScale(face_rgn_gray)
    eyes_center = [(0, 0), (0, 0)]  # center of each eyes
    if(len(eye_pos) == 2*eyes_pair):
        for i in range(len(eye_pos)):
            (ex, ey, ew, eh) = eye_pos[i]
            eyes_center[i] = (ex+ew/2, ey+eh/2)
            cv2.circle(face_rgn, eyes_center[i], 3, (0, 255, 255), -1)
            cv2.rectangle(face_rgn, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 0)

    mouth_pos = mouth_cascade.detectMultiScale(face_rgn_gray)
    (mx, my, mw, mh) = mouth_pos[1]
    cv2.rectangle(face_rgn, (mx, my), (mx+mw, my+mh), (0, 0, 255), 0)
    mouth_center = (mx+mw/2, my+mh/2)
    cv2.circle(face_rgn, mouth_center, 3, (0, 0, 255), -1)
    
    raw = np.arctan(float((eyes_center[1][1]-eyes_center[0][1])) / float((eyes_center[1][0]-eyes_center[0][0])))
    print(raw)



cv2.imshow('local cam', frame)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destoryAllWindows()