import numpy as np
from PIL import Image
import glob
import pytesseract
from IPython.display import display
import os
import collections
import ntpath
import copy
import cv2

import skimage
from skimage import io, util, segmentation, img_as_uint
from skimage.transform import rescale
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel, threshold_local
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray

from scipy.ndimage.measurements import center_of_mass
from skimage.filters import threshold_otsu, rank
from scipy import ndimage
from scipy.ndimage.measurements import minimum_position
from skimage.morphology import disk
from scipy.ndimage.measurements import center_of_mass, sum
from noise_removal.denoisingFunctions import *

def denoiseBoxFromPath(path, savePath='.', order = "BSC", cleanVariance = 30, binThresh = 1001, cropLastSpace = 40, cropBefore = True, cropAfter = False, borderLimitFactor = 0.3, percentileOutlier = 5, modePercentile='percentile' , iqrFactor=1):
    """Read skimage from path, apply denoise function and save it to folder savePath."""
    img = io.imread(path, as_grey=True)
    cleaned = denoiseBoxFromImageWithType(img, "Name", "name_tha", savePath, clearBorder, minSizeDark, minSizeMid, minSizeLight, binThresh)
    if clearBorder:
        borderCleaned = "clearedBorder"
    else:
        borderCleaned =  "borderNotCleared"

    #plt.subplot(1,2,1),#plt.imshow(img, cmap='gray')
    #plt.subplot(1,2,2), #plt.imshow(cleaned, cmap='gray')
    #plt.show()
    cleaned = Image.fromarray(np.array(cleaned, dtype='uint8'))
    cleaned.save(savePath+"/cleaned_"+borderCleaned+ntpath.basename(path))
    
def denoiseBoxFromImageWithType(v,field, k):
    if field == 'Name':
        if k == 'name_tha':
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50, removeTop = False, cleanVariance = 15, binThresh = 1001, cropLastSpace = 50, cropBefore = True, cropAfter = True, borderLimitFactor = 0.35, percentileOutlier = 5, modePercentile='percentile' , iqrFactor=1)
        elif k == 'name_eng':
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50,cleanVariance = 15, binThresh = 1001, cropLastSpace = 50, cropBefore = True, cropAfter = True, borderLimitFactor = 0.35, percentileOutlier = 7, modePercentile='percentile' , iqrFactor=1)
        else:
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50,cleanVariance = 15, binThresh = 1001,  cropLastSpace = 50, cropBefore = True, cropAfter = True, borderLimitFactor = 0.35, percentileOutlier = 7, modePercentile='percentile' , iqrFactor=1)

    elif field == 'Date':
        if k == 'bd_eng':
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50,cleanVariance = 15, binThresh = 1001, cropLastSpace = 80, cropBefore = True, cropAfter = False, borderLimitFactor = 0.35, percentileOutlier = 5, modePercentile='percentile' , iqrFactor=1)
        elif k == 'ed_eng':
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50,cleanVariance = 15, binThresh = 1001, cropLastSpace = 80, cropBefore = True, cropAfter = False, borderLimitFactor = 0.35, percentileOutlier = 5, modePercentile='percentile' , iqrFactor=1)
        elif k == 'bd_tha':
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50,cleanVariance = 15, binThresh = 1001, cropLastSpace = 80, cropBefore = True, cropAfter = False, borderLimitFactor = 0.35, percentileOutlier = 5, modePercentile='percentile' , iqrFactor=1)
        else:
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50,cleanVariance = 15, binThresh = 1001, cropLastSpace = 80, cropBefore = True, cropAfter = False, borderLimitFactor = 0.35, percentileOutlier = 5, modePercentile='percentile' , iqrFactor=1)

    elif field == 'Address':
        if k == 'address_l1':
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50,removeTop = True, cleanVariance = 15, binThresh = 1001, cropLastSpace = 40, cropBefore = True, cropAfter = True, borderLimitFactor = 0.35, percentileOutlier = 7, modePercentile='percentile' , iqrFactor=1)
        else:
            return denoiseBoxFromImage(v, field, k, order = "BSKOM", kernel_erosion = 3, kernel_dilatation = 50, removeTop = True, cleanVariance = 15, binThresh = 1001, cropLastSpace = 40, cropBefore = True, cropAfter = True, borderLimitFactor = 0.35, percentileOutlier = 7, modePercentile='percentile' , iqrFactor=1)
    
def denoiseBoxFromImage(img, field, k, order = "BS", secondRound = False, kernel_dilatation = 50, kernel_erosion = 20, removeTop = True, removeBottom = True, cleanVariance = 30, binThresh = 1001, cropLastSpace = 40, cropBefore = True, cropAfter = False, borderLimitFactor = 0.35, percentileOutlier = 5, modePercentile='percentile' , iqrFactor=1):
    """Denoise an image (of format np.ndarray with values 0 to 255) and return the cleaned Pillow image. The order 
    of functions applied is simply B for removing border, S for removing small sizes and C for removing outliers in center of mass"""
    #plt.subplot(4,3,1), #plt.imshow(img, cmap='gray')
    img1 = copy.deepcopy(img)
    if cropBefore:
        img = crop(img1, cropLastSpace)

    img_init = copy.deepcopy(img)
    
    ySize, xSize = img.shape
    img_init_bool = img > 230
    
    # Binarise image with local threshold
    selem = disk(10)
    local_otsu = rank.otsu(img, selem)
    local_thresh = threshold_local(img, 1001)
    img_thresh = img > local_otsu
    # Black/white inverted
    reverse = util.invert(img_thresh)
    variance = 0
    # Apply each function in order to the image 
    # Apply each function in order to the image 
    for funType in order:
        label_objects, nb_labels = ndi.label(reverse)
        if len(label_objects) > 0:
            indexLabels = np.unique(label_objects)[1:]
            centers = center_of_mass(label_objects, labels=label_objects, index=indexLabels)
            if funType == "M":
                reverse = remove_center_of_mass_outlier(reverse, field = field, percentile = percentileOutlier, mode = modePercentile, iqrFactor=iqrFactor)
                if len(centers) > 0:
                    centersY = np.apply_along_axis(lambda x: x[0], 1, centers)
                    variance = np.var(centersY)
                    #print("Variance y "+str(variance))
                    bw = blackWhiteRatio(reverse)
                    #print("BW ratio: "+ str(bw))
                    #plt.subplot(4,3,6), #plt.imshow(reverse, cmap='gray')
            elif len(centers) > 0:
                centersY = np.apply_along_axis(lambda x: x[0], 1, centers)
                centersX = np.apply_along_axis(lambda x: x[1], 1, centers)
                #print(funType)
                variance = np.var(centersY)
                #print("Variance y "+str(variance))
                bw = blackWhiteRatio(reverse)
                #print("BW ratio: "+ str(bw))
                if cleanVariance < np.var(centersY):
                    if funType == "B":
                        reverse = remove_border(reverse, ySize, key = k, field = field, borderLimitFactor = borderLimitFactor, removeBottom = removeBottom, removeTop = removeTop)
                        #plt.subplot(4,3,2), #plt.imshow(reverse, cmap='gray')
                    elif funType == "S":
                        reverse = remove_small_outlier(reverse, percentile=percentileOutlier, mode=modePercentile , iqrFactor=iqrFactor)
                        #plt.subplot(4,3,3), #plt.imshow(reverse, cmap='gray')
                    elif funType == "K":
                        reverse = keep_big_component(reverse, key = k, kernel_erosion = kernel_erosion, kernel_dilatation = kernel_dilatation, variance=np.var(centersY))
                        #plt.subplot(4,3,4), #plt.imshow(reverse, cmap='gray')
                    elif funType == "O":
                        reverse = opening(reverse, bw)
                        #plt.subplot(4,3,5), #plt.imshow(reverse, cmap='gray')
            else:
                variance = 0
            
    bw = blackWhiteRatio(reverse)
    cleaned = np.invert(reverse)  
    # Matrix containing true if the pixel has been removed
    pixelsRemoved = cleaned.__xor__(img_init_bool)
    img_init[pixelsRemoved] = 255
    #plt.subplot(4,3,7), #plt.imshow(img_init, cmap='gray')
    #plt.show()
    if cropAfter:
        img_init = crop(img_init, cropLastSpace)
    return (Image.fromarray(np.array(img_init, dtype='uint8')), bw, variance)
