#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 22:58:55 2018

@author: abhishek
"""


import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('../data/object_detect_xmls/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('../data/object_detect_xmls/haarcascade_smile.xml')

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray)
        for (fx, fy, fw, fh) in smile:
            cv2.rectangle(roi_color, (fx,fy), (fx+fw, fy+fh), (0,255,0), 2)
            
    cv2.imshow('camera',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cam.release()
        cv2.destroyAllWindows()
        break
