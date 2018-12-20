import numpy as np
import os
import collections
import ntpath
import copy
import cv2
import sys

import skimage
from skimage import io, util, segmentation, img_as_uint
from skimage.transform import rescale
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel, threshold_local, threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray

from scipy.ndimage.measurements import center_of_mass, minimum_position, sum
from scipy import ndimage
from skimage.morphology import disk
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [20, 15]

def blackWhiteRatio(img_thresh):
    white = img_thresh[(img_thresh == False)].size
    if white == 0:
        blackWhiteRatio = 1
    else:
        blackWhiteRatio = img_thresh[(img_thresh == True)].size / white
    return blackWhiteRatio

def sort_boxes(contours):
    """Sort the contours from left to right and return the bounded boxes of the contours."""
    if len(contours) > 0:
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        # Sort boxes by x-coordinate + width
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b:(b[1][0]+b[1][2])))
        return boundingBoxes
    else:
        return []

def keepLabel(label, keptLabels):
    """Apply this function on label objects to keep objects with IDs in keptLabels"""
    if label in keptLabels:
        return label
    else:
        return 0
keep = np.vectorize(keepLabel)

def crop(img, lastSpace = 40):
    """Crop the right part of the image not containing text (keep only centered big boxes)
    """
    img_final = copy.deepcopy(img)
    img2gray = copy.deepcopy(img)
    # Binarise image, inverting black and white (cv2 finds contours of white on top of black)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray , img2gray , mask =  mask)
    ret, new_img = cv2.threshold(image_final, 180 , 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours = cv2.findContours(new_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[1]

    # Keep right most coordinate of the boxes kept
    maxWidth = 0
    height= img_final.shape[0]
    width = img_final.shape[1]
    precX = 0
    for box in sort_boxes(contours):
        # get rectangle bounding contour
        [x, y, w, h] = box
        # We skip the box if 1) it's too small 2) It's on the top 3) It's on the bottom 4) The space between the box and box kept on the left is too big
        if (h < 10 and w < 10) or (y < 10 and (y+h < 0.2*height)) or (y+20 > height) or (x - precX > lastSpace):
            continue
        # the x coordinate of the object is on the right
        if maxWidth < (x + w):
            maxWidth = x + w
        # It's the right most box
        if x + w > precX:
            precX = x+w
    # We don't crop if it's too much on the left and there's still text on the right
    if maxWidth < (0.4 * width) and blackWhiteRatio(img_final[0 : height , maxWidth : width-1]) > 0.05:
        maxWidth = width
    cropped = img_final[0 : height , 0 : maxWidth+10]
    return cropped

def remove_border(img, h, field, key, borderLimitFactor = 0.1, removeTop = True, removeBottom = True):
    """Remove border based on the type of field. Handle the special case of dates.
    name = B
    """  
    if field == "Date":
        return remove_border_date(img, h, key, borderLimitFactor) 
    else: 
        return remove_border_standard(img, h, borderLimitFactor, removeTop, removeBottom)

def remove_border_standard(img, h, borderLimitFactor = 0.1, removeTop = True, removeBottom = True, minBorder = 10):
    """Remove either objects under first percentile, or using IQR. Mode: percentile or IQR. Remove objects close to border too
    name = B
    """    
    label_objects, nb_labels = ndi.label(img)
    indexLabels = np.unique(label_objects)[1:]
    centers = center_of_mass(label_objects, labels=label_objects, index=indexLabels)
    kept_labels = []  
    for i in indexLabels:
        # Check if it's touching border and not including text (y axis)
        slice_y, slice_x = ndimage.find_objects(label_objects == i)[0]
        borderLimit = borderLimitFactor*h
        if not ((slice_y.start < minBorder and (slice_y.stop < borderLimit) and removeTop) or (slice_y.start + borderLimit > h and (slice_y.stop > (h - minBorder)) and removeBottom)):
            kept_labels.append(i)    
    return (keep(label_objects, set(kept_labels)) > 0)
  
def remove_border_date(img, h, key = 0.1, borderLimitFactor = 0.5):
    """Remove either objects under first percentile, or using IQR. Mode: percentile or IQR. Remove objects close to border too
    name = B
    """    
    half = int(h/2)
    topHalf = img[0 : half, 0:]
    bottomHalf = img[half : h-1, 0:]
    bwTop = blackWhiteRatio(topHalf)
    bwBottom = blackWhiteRatio(bottomHalf)
    if key == "bd_tha":
        if bwTop > 0.1 and bwBottom > 0.1:
            img = remove_border_standard(img, h, removeTop = False, borderLimitFactor = 0.6, minBorder = h)
            img = remove_border_standard(img, h, removeTop = True, removeBottom = False, borderLimitFactor = 0.2, minBorder=15)
            #print("BW top "+str(bwTop))
            #print("BW bottom "+str(bwBottom))
            img[int(0.6*h):h-1, 0:] = False
        else:
            img = remove_border_standard(img, h)
    elif key == "bd_eng":
         img = remove_border_standard(img, h, borderLimitFactor = 0.35, removeTop = True, removeBottom = True, minBorder = 30)
    else:
         img = remove_border_standard(img, h, borderLimitFactor = 0.3, removeTop = True, removeBottom = True, minBorder = 20)
    return img

        
def remove_small_outlier(img, percentile = 10, mode = 'percentile', iqrFactor = 1):
    """Remove either objects under first percentile, or using IQR. Mode: percentile or IQR.
    name = S
    """
    label_objects, nb_labels = ndi.label(img)
    indexLabels = np.unique(label_objects)[1:]
    sizes = sum(label_objects, labels=label_objects, index=indexLabels)
    for i in range(len(sizes)):
        sizes[i] = sizes[i] / indexLabels[i]
    perc_low, perc_high = np.percentile(sizes, [percentile, 100 - percentile])
    iqr = perc_high - perc_low  
    #print("Lower bound: "+str(perc_low) + " limit 15")
    if mode == 'percentile':
        lower_bound = perc_low
    else:
        lower_bound = perc_low - (iqr * iqrFactor)
    lower_bound = min(lower_bound, 7)
    kept_labels = []
    counter = 0
    for i in indexLabels:
        if sizes[counter] > lower_bound:
            kept_labels.append(i)
        counter = counter + 1
    return keep(label_objects, set(kept_labels)) > 0

def keep_big_component(img, key = "", kernel_erosion = 20, kernel_dilatation = 50, variance = 1):
    """Erode the image through y-axis, dilate through x-axis and keep biggest component
    name = K
    """
    if True:#variance > 30 and (blackWhiteRatio(img) > 0.06):
        image = np.copy(img)
        image[img == True] = 255
        image[img == False] = 0
        image = np.array(image, dtype=np.uint8)

        kx_1, ky_1 = (kernel_erosion,1)

        # huge erosion through y_axis to make non-box components vanish
        kernel_1 = np.ones((kx_1, ky_1), np.uint8)
        eroded = cv2.erode(image, kernel_1, iterations=1)
        #plt.subplot(2,3,2), #plt.imshow(eroded, cmap='gray')
        
        kx_2, ky_2 = (1, kernel_dilatation)

        # huge dilatation through x-axis to make non-box components vanish
        kernel_2 = np.ones((kx_2, ky_2),np.uint8)
        dilated = cv2.dilate(eroded, kernel_2, iterations=1 )
        dil_k = kx_1 + 2
        if key == "name_tha" or key == "address_l1":
            dil_k = kx_1 + 6
        if key == "address_l2":
            dil_k = kx_1 + 2
        dilated = cv2.dilate(dilated,np.ones((dil_k, 2), np.uint8), iterations = 1) 
        #plt.subplot(2,3,3), #plt.imshow(dilated, cmap='gray')
        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=4)

        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        sizes = list(sizes)
        img3 = np.copy(img)
        if len(sizes) > 0:
            max_size = max(sizes)
            max_label = sizes.index(max_size)

            left = stats[1:][max_label][0]
            top = stats[1:][max_label][1]
            width = stats[1:][max_label][2]
            height = stats[1:][max_label][3]

            img2 = np.zeros(output.shape)
            img2[output == max_label +1] = 255              
            final = cv2.erode(img2, kernel_2, iterations = 1)
            #plt.subplot(2,3,4),#plt.imshow(final, cmap='gray')  

            for y in range(0, img.shape[0]):
                for x in range(0,img.shape[1]):
                    if final[y][x] == 0:
                        img3[y][x] = False
                    elif img3[y][x] and final[y][x] == 255:
                        img3[y][x] = True
        #plt.subplot(2,3,5), #plt.imshow(img3, cmap='gray')
        return np.array(img3, dtype=bool)
    return img

def weighted_median(weights, data):
    sorted_data, sorted_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(sorted_weights)
    if any(weights > midpoint):
        return (data[weights == np.max(weights)])[0]
    cumulative_weight = np.cumsum(sorted_weights)
    below_midpoint_index = np.where(cumulative_weight <= midpoint)[0][-1]
    if cumulative_weight[below_midpoint_index] - midpoint < sys.float_info.epsilon:
        return np.mean(sorted_data[below_midpoint_index:below_midpoint_index+2])
    return sorted_data[below_midpoint_index+1]

def remove_center_of_mass_outlier(img, field = "", percentile = 10, mode = 'percentile', iqrFactor = 1):
    """Compute the weighted median of centers regarding to objects sizes and take upper and lower text
    name = "M"
    """
    k=np.ones((2, 2),np.uint8)
    eroded_img = skimage.morphology.erosion(img, k)
    label_objects, nb_labels = ndi.label(img)
    if nb_labels < 2:
        return img
    indexLabels = np.unique(label_objects)[1:]
    centers = center_of_mass(label_objects, labels=label_objects, index=indexLabels)
    centersY = np.apply_along_axis(lambda x: x[0], 1, centers)
    sizes = sum(label_objects, labels=label_objects, index=indexLabels)

    weightedMedian = weighted_median(sizes, centersY)
    y = img.shape[0]
    pad = int(1/3 * y)
    if field == "Date":
        pad = int(1/2 * y)
    upper_bound = min(weightedMedian + pad, y - 1)
    lower_bound = max(weightedMedian - pad - 3, 0)
    #print("Lower bound centers "+str(lower_bound))
    #print("Upper bound centers "+str(upper_bound))
    img[0:int(lower_bound), 0:] = False
    img[int(upper_bound) : y -1, 0:] = False

    return img

def opening(img, bwRatio):
#Define kernel regarding the category of image
    image = np.copy(img)
    image[img == True] = 255
    image[img == False] = 0
    image = np.array(image, dtype=np.uint8)
    if bwRatio < 0.15:
        kx, ky = (1,1)
    else:
        kx, ky = (2,2)
    kernel_1 = np.ones((kx, ky),np.uint8)

    opening = cv2.morphologyEx(image,cv2.MORPH_OPEN, kernel_1 )

    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=4)
    
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    size_thresh = 20

    sizes = list(sizes)
    #print(str(sizes))

    img2 = np.zeros(output.shape)

    for max_label, size in enumerate(sizes) : 
        if size >= size_thresh :
            img2[output == max_label +1] = 255

    return img2 > 0
