import numpy as np
import pandas as pd
import os
import time
import random
import matplotlib.pyplot as plt

from glob import glob

import PIL
import cv2

def hog(image_name):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #image_path = glob.glob('data/*.jpg')
    #print(image_path)

    #for image_name in image_path:

    img = cv2.imread(image_name)

    if img.shape[1] < 400:
        (height, width) = img.shape[:2]
        ratio = float(height)/float(width)
        img = cv2.resize(img,(400, 350))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects,weights = hog.detectMultiScale(img_gray, winStride =(2,2), padding=(10,10), scale = 1.02)

    for i, (x,y,w,h) in enumerate(rects):
        if weights[i] < 0.1:
            continue
        elif weights[i] > 0.1 and weights[i] < 0.3:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)

        elif weights[i] > 0.3 and weights[i] < 0.7:
            cv2.rectangle(img, (x,y), (x+w, y+h), (50,122,255),2)

        elif weights[i] > 0.7:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)


    cv2.imshow('Image', img)
    cv2.waitKey(0)