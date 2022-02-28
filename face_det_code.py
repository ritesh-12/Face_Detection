# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:31:31 2021

@author: Debian_Boy
"""

import cv2
face_coll = cv2.CascadeClassifier('face_detection.xml')
img1 = cv2.imread('myImage.jpg')
faces = face_coll.detectMultiScale(img1, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img1, (x,y),(x+w, y+h), (255, 0, 0), 2)
cv2.imwrite("face_detected.png", img1)
print('Successfully saved')


