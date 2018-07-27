#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 20:55:40 2018
From the trained recognizer predict it
@author: abhishek
"""

import cv2
import numpy as np

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('../data/object_detect_xmls/face_detect.yml')
face_cascade = cv2.CascadeClassifier('../data/object_detect_xmls/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        ID = recognizer.predict(gray[y:y+h, x:x+w])
        if (ID==1):
            ID = 'Abhishek'
        if (ID==2):
            ID = 'Rashmi'
        elif (ID==3):
            ID = 'Goutam'
        elif (ID==4):
            ID = 'Sayan'
        else:
            ID = 'Unknown'
        cv2.putText(img, str(ID), (x,y+h), font, 1, (255,255,255), 3, cv2.LINE_AA)
    
    cv2.imshow('camera',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cam.release()
        cv2.destroyAllWindows()
        break
