#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:40:06 2020

@author: tayfunkaraderi
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageChops
from PIL import Image
import os



def convert_b_w(img):
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 0:
                img[i,j,0] = 255; img[i,j,1] = 255; img[i,j,2] = 255;
                
    return img

a35='34 - Turborotalita quinqueloba'
a34='33 - Turborotalita humilis'
a33='32 - Tenuitella iota'
a32='31 - Sphaeroidinella dehiscens'
a31='30 - Pulleniatina obliquiloculata'
a30='29 - Orbulina universa'
a29='28 - Neogloboquadrina pachyderma'
a28='27-Neogloboquadrina incompta'
a27='26 - Neogloboquadrina dutertrei'
a26='25 - Hastigerina pelagica'
a25='24 - Globoturborotalita tenella'
a24='23 - Globoturborotalita rubescens'
a23='22 - Globorotaloides hexagonus'
a22='21 - Globorotalia ungulata'
a21='20 - Globorotalia tumida'
a20='19-Globorotalia truncatulinoides'
a19='18-Globorotalia scitula'
a18='17 - Globorotalia menardii'
a17='16 - Globorotalia inflata'
a16='15 - Globorotalia hirsuta'
a15='14 - Globorotalia crassaformis '
a14='13 - Globoquadrina conglomerata'
a13='12 - Globigerinoides sacculifer'
a12='11 - Globigerinoides ruber'
a11='10 - Globigerinoides elongatus'
a10='9 - Globigerinoides conglobatus'
a9='8 - Globigerinita uvula'
a8='7- Globigerinita glutinata'
a7='6 - Globigerinella siph '
a6='5 - Globigerinella calida'
a5='4 -Globigerinella adamsi'
a4='3 - Globigerina falconensis'
a3='2 - Globigerina bulloides'
a2='1 - Candeina nitida'
a1='0 - Beella digitata'

b35='035'
b34='034'
b33='033'
b32='032'
b31='031'
b30='030'
b29='029'
b28='028'
b27='027'
b26='026'
b25='025'
b24='024'
b23='023'
b22='022'
b21='021'
b20='020'
b19='019'
b18='018'
b17='017'
b16='016'
b15='015'
b14='014'
b13='013'
b12='012'
b11='011'
b10='010'
b9='009'
b8='008'
b7='007'
b6='006'
b5='005'
b4='004'
b3='003'
b2='002'
b1='001'


Directory = '/Users/tayfunkaraderi/Desktop/Forams_All/Forams Dataset/'+a35+'/images/'
Directory2 = '/Users/tayfunkaraderi/Desktop/Forams_All/Forams Dataset - location/'+b35+'/'

count = 0
for file in os.listdir(Directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        count += 1
        file_name = str(file)
        #print(file_name, count)
        # Read image
        #print(file)
        #path = '/Users/tayfunkaraderi/Desktop/Forams-Data/724748_ex307637_obj00122.jpg'
        #path = '/Users/tayfunkaraderi/Desktop/Forams-Data/001.png'
        
        
        img1 = cv2.imread(Directory + file_name)
        img = convert_b_w(img1)
        #print(img)
        plt.imshow(img1)
        
        # Displaying the image 
        #plt.imshow(img)
        #cv2.imshow('image', img)
        
        ## (1) Convert to gray, and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        ## (2) Morph-op to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
        
        ## (3) Find the max-area contour
        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        
        ## (4) Crop and save it
        x,y,w,h = cv2.boundingRect(cnt)
        dst = img[y:y+h, x:x+w]
        #cv2.imwrite("003-22.png", dst)
        
        #plt.imshow(dst)
        
        
        d2_img = dst.sum(axis=2)/3
        d1_x_img = d2_img.sum(axis=0)/(d2_img.shape[0])
        x_crop = np.where(d1_x_img < 240)[0]
        xcrop0 = x_crop[0] + 10; xcrop1 = x_crop[-1] - 10 
        
        
        # Crop image
        d2_img_crop = dst[10:-10, xcrop0:xcrop1]
        # Resize image
        dim = (416, 416)
        resized = cv2.resize(d2_img_crop, dim, interpolation = cv2.INTER_AREA)
        
        #cv2.imwrite(Directory + "-%d.png"%count, resized)
        cv2.imwrite(Directory2 + file_name, resized)
        
        plt.imshow(resized)
        
    
    #img = cv2.imread("002-2.png")
    #img2 = convert_b_w(img)
    
    #plt.imshow(img)
    #plt.imshow(img2)


             
