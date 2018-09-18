import cv2
import tensorflow as tf
import numpy as np
import os
import sys

face_cascade = cv2.CascadeClassifier('/home/brandon/OpenCV/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascasde = cv2.CascadeClassifier('/home/brandon/OpenCV/opencv-3.4.1/data/haarcascades/haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # face region: [x, y, width, height] 
        #  => lower left: (x, y) upper right: (x+w, y+h)
        faces_pos = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x, y, w, h) in faces_pos:
            # find face region and draw bouding box
            face_rgn = frame_gray[y:y+h, x:x+w]
            cv2.rectangle(frame_gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # find eyes and draw bounding boxes
            eye_pos = eye_cascasde.detectMultiScale(face_rgn)
            for (ex, ey, ew, eh) in eye_pos:
                cv2.rectangle(face_rgn, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 0)
        cv2.imshow('local cam', frame_gray)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.imwrite('pose.jpg', frame)
cv2.imwrite('pose_gray.jpg', frame_gray)
video_capture.release()
cv2.destoryAllWindows()
