import cv2
import numpy as np
import bresenham


def calcul_width(pixel,img):
    x = pixel[0]
    y = pixel[1]
    
    h,w = img.shape
    lim_l = x
    lim_r = x
    
    while lim_l - 1 >=0 and img[y][lim_l - 1] == 255 :
        lim_l -= 1
    while lim_r + 1 <w and img[y][lim_r + 1] == 255 :
        lim_r += 1
        
        
    width = lim_r - lim_l + 1
    
    img = np.array(img)
    
    return (width,lim_l,lim_r)

def signature_denoise_2(img):

    image = np.copy(img)

    image[img == 0] = 255
    image[img == 255] = 0

    kx, ky = (1,3)

    kernel = np.ones((kx, ky),np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)

    eroded = cv2.erode(dilated, kernel, iterations=1)

    threshold = 50
    minLineLength = 5
    maxLineGap = 2

    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold, 0, minLineLength, maxLineGap)

    if len(lines) > 3500 :
        return img        

    image = np.copy(img)

    somme1 = 0

    for line in lines : 
        line = line[0]
        length = np.linalg.norm(np.array([line[0], line[1]])- np.array([line[2], line[3]]))
        somme1 += length
        
    somme1 = somme1 / len(lines)


    y_global = []
    lim_l_global = []
    lim_r_global = []
    median_width_global = []
    diff = []

    for x in range(0, len(lines)):
        
        for x1,y1,x2,y2 in lines[x]:
            
            length = np.linalg.norm(np.array([x1, y1])- np.array([x2, y2]))
            
            if length > 30:    
                
                if x2 != x1:
                
                    if 0.3 < abs((y2-y1)/(x2-x1)): 
                        

                        coordinates = list(bresenham.bresenham(x1,y1,x2,y2)) #Couple de pixel (x,y) entre deux points extremes
                        
                        coordinates = np.array(coordinates)
                        
                        median_width = []
                        lim_ls = []
                        lim_rs = []
                        y = []
                    
                        for elmt in coordinates:
                            width, lim_l, lim_r = calcul_width(elmt,eroded)
                            median_width += [width]
                            lim_ls += [lim_l]
                            lim_rs += [lim_r]
                            y += [elmt[1]]
                        
                        
                        median_width = np.array(median_width, dtype = 'int')
                        m_width = np.median(median_width)
                        m_width = int(m_width)
                        
                        lim_ls = np.array(lim_ls, dtype = 'int')
                        lim_rs = np.array(lim_rs, dtype = 'int')
                        y = np.array(y)
                        
                        y_global += [y]
                        lim_l_global += [lim_ls]
                        lim_r_global += [lim_rs]
                        median_width_global += [m_width]
                        
                    

                        
    y_global = np.array(y_global)
    lim_l_global = np.array(lim_l_global)
    lim_r_global = np.array(lim_r_global)
    median_width_global = np.array(median_width_global, dtype = 'int')

    if len(median_width_global) > 0:
        median_width_global = np.median(median_width_global) 
    else :
        median_width_global = 0 

    diff = lim_r_global - lim_l_global
    diff -= int(median_width_global)



    for pos,elmt in enumerate(diff) :
        diff[pos] = diff[pos] <(median_width_global //2)
        lim_l_global[pos] = lim_l_global[pos][diff[pos]]
        lim_r_global[pos] = lim_r_global[pos][diff[pos]]
        y_global[pos] = y_global[pos][diff[pos]]


    for pos_1, elmt in enumerate(y_global):   
        
        for pos_2, ordo in enumerate(elmt):      
                        
            image[ordo][lim_l_global[pos_1][pos_2] - 1 : lim_r_global[pos_1][pos_2] +1] = 255
                                            
    return image