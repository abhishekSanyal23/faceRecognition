#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 20:18:52 2018
This code will train with the face data from dataset generated
@author: abhishek
"""
import cv2, os
import numpy as np
from PIL import Image

#Create recognizer
recognizer = cv2.face.createLBPHFaceRecognizer()
face_cascade = cv2.CascadeClassifier('../data/object_detect_xmls/haarcascade_frontalface_default.xml')

def getImagesAndLabel(path):
    #get all the files from the folder
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    
    #Create empty face list
    faceSamples = []
    
    #Create empty id list
    ids = []
    
    #now looping through all the image files and loading the images and paths in numpy
    for imagePath in imagePaths:
        #Loading the images and convert it into gray scale
        pilImage = Image.open(imagePath).convert('L')
        
        #Convert PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        
        #Getting the id name from the image database
        faceId = int(os.path.split(imagePath)[-1].split(".")[1])
        
        #Extract the faces from the training image sample
        faces = face_cascade.detectMultiScale(imageNp)
        
        
        #Create a face list for a group
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            ids.append(faceId)
    
    return faceSamples, ids

faces, ids = getImagesAndLabel('../data/faces')
recognizer.train(faces, np.array(ids))
recognizer.save('../data/object_detect_xmls/face_detect.yml')
        


