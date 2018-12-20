import numpy as np
import cv2

def remove_black_boxes_on_image(image) :

    inversed = np.copy(image)

    #we look for 255s components
    inversed[image == 0] = 255
    inversed[image == 255] = 0

    scale_0 = image.shape[0]/2316
    scale_1 = image.shape[1]/1649

    kx, ky = (int(10 * scale_0) + 10,int(10 * scale_1) + 10)

    # huge erosion to make non-box components vanish
    kernel = np.ones((kx, ky),np.uint8)
    eroded = cv2.erode(inversed,kernel,iterations = 1)

    kernel = np.ones((kx + 1, ky + 1),np.uint8)
    dilated = cv2.dilate(eroded, kernel, iterations=1 )

    nbb = np.copy(image)

    nbb[dilated==255] = 255
    
    return nbb