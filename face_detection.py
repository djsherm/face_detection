# -*- coding: utf-8 -*-
"""
Live Face Detection and Filter

Created on Mon Dec 12 11:45:54 2022

@author: Daniel
"""

##### LOAD LIBRARIES #####
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request as urlreq

printed = False

# load pre-trained classifier for frontal face
haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
LBFmodel = "lbfmodel.yaml"

landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("File exists")
else:
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")

def detect_faces(test_image, cascade=haar_cascade_face, landmark_detector=landmark_detector, scaleFactor=1.1):
    image_copy = test_image.copy()
    
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    
    # detect landmarks
    
    if faces_rect != ():
        _, landmarks = landmark_detector.fit(gray_image, faces_rect)
        # for (x,y,w,h) in faces_rect:
        #     cv2.rectangle(image_copy, (x,y), (x+w, y+h), (0, 255, 0), 4) #box around each face
        for landmark in landmarks:
            for x, y in landmark[0]:
                cv2.circle(image_copy, (int(x), int(y)), 1, (0, 0, 255), 2)
    else: #no face detected
        return image_copy
        
    return image_copy

# 1. Get access to the computer's webcam via python
window = 'myWindow'

cv2.namedWindow(window)
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    
# 2. Find and highlight bounding box of face

while rval:
    frame = np.flip(frame, axis=1) # flip the image horizontally to look like a mirror image
    
    detected = detect_faces(frame)
        
    cv2.imshow(window, detected) # ensure that frame is a numpy array after whatever processing we want to do
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    
    #rval = False # temp change rval to only get 1 frame
    if key == 27: # exit on ESC
        break
    
vc.release()
cv2.destroyWindow(window)

# test_image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# image = Image.fromarray(test_image_gray)

# image.show()

# 3. Overlay filter

#hypnotoad - overlay circles around the eyes and change through colors