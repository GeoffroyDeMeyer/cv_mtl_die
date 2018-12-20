import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import os
import bresenham


def pixel_is_white(image,x,y) :
    return np.array_equal(image[y][x],np.array([255,255,255]))

def pixel_is_black(image,x,y) :
    return np.array_equal(image[y][x],np.array([0,0,0]))


def signature_denoise_on_image(img):
    if len(img.shape) > 2:
        raise ValueError('Please pass black and white image')
        
    h,w = img.shape

    image = np.copy(img)

    image[img == 0] = 255
    image[img == 255] = 0


    threshold = 50
    minLineLength = 20
    maxLineGap = 2


    #determine every lines into the document regarding some parameters 

    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold, 0, minLineLength, maxLineGap)
            
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    somme1 = 0

    for line in lines : 
        line = line[0]
        length = np.linalg.norm(np.array([line[0], line[1]])- np.array([line[2], line[3]]))
        somme1 += length
    
    somme1 = somme1 / len(lines)


    for x in range(0, len(lines)):
    
        for x1,y1,x2,y2 in lines[x]:
        
            length = np.linalg.norm(np.array([x1, y1])- np.array([x2, y2]))
        
            if length >  20  :
            
                if x2 != x1 :
            
                    if 0.3 < abs((y2-y1)/(x2-x1)) < 5:

                    
                        #determine lines between two extremal points
                        coordinates = list(bresenham.bresenham(x1,y1,x2,y2)) 
                    
      
                        coordinates = np.array(coordinates)
        
                    
                        #List with unique element for y axis 
                        unique = np.unique(coordinates[:,0])
                    
                    
                    

                        L1 = []

                        for elmt in unique : 
                            L1 += [coordinates[ coordinates[:,0] == elmt]]
                        #List of group having the same y-axis value
                
                        for elmt in L1 :
                            #Order each group of this list regarding x-axis
                            elmt = np.sort(elmt, axis=0)
                        
                            lim_r = elmt[-1][0] + 1
                            lim_l = elmt[0][0] - 1
                            ordo = elmt[0][1]

                        if (lim_r < w and pixel_is_white(image, lim_r, ordo)) or (lim_l >= 0 and pixel_is_white(image, lim_l, ordo)):
                            for pixel in elmt :
                                
                                image[ordo][pixel[0]] = [255,255,255]
                        
                            if lim_r < w and pixel_is_white(image, lim_r, ordo) :
                                if lim_l >= 0 and pixel_is_white(image, lim_l, ordo) :
                                    for pixel in elmt : 
                                        image[ordo][pixel[0]] = [255,255,255]     
                                elif  lim_l >= 0 and pixel_is_black(image, lim_r, ordo): 
                                    for pixel in elmt :
                                        if pixel[0] >= elmt[int(len(elmt)/2)][0] :
                                            image[ordo][pixel[0]] = [255,255,255]
                                
                            if lim_l >= 0 and pixel_is_white(image, lim_l, ordo) :     
                                if  lim_r < w and pixel_is_black(image, lim_r, ordo): 
                                    for pixel in elmt : 
                                        if pixel[0] <= elmt[int(len(elmt)/2)][0] :
                                            image[ordo][pixel[0]] = [255,255,255] 

    return image 