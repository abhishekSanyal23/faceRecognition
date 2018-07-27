#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:17:29 2018

This code will generate dataset for face detection.
@author: abhishek
"""

import cv2
import numpy as np

#Detect the face with haar-cascade face detection.
face_cascade = cv2.CascadeClassifier('../data/object_detect_xmls/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

id = input('Enter your id ')
sampleNum = 0

while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        #incrementing the sample number
        sampleNum = sampleNum + 1
        
        #Saving the captured dataset in the folder
        cv2.imwrite('../data/faces/user.'+ id + '.' + str(sampleNum) + '.jpg', gray[y:y+h, x:x+w])
        cv2.imshow('frame', img)
        
        #Check and break
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cam.release()
            cv2.destroyAllWindows()
            break


        
        


