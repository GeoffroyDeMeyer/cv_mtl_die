
import numpy as np
import cv2


def text_mask(img) :

    image = np.copy(img)

    image[img == 0] = 255
    image[img == 255] = 0

    #scale_0 = img.shape[0]/2316
    #scale_1 = img.shape[1]/1649

    #kx, ky = ((int(10 * scale_0),int(10 * scale_1)))

    kx, ky = (1,2)

    # huge erosion to make non-box components vanish
    kernel = np.ones((kx, ky),np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1 )


    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=4)


    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.

    sizes = stats[:, -1]
    nb_components = nb_components - 1

    sizes = list(sizes)

    mean_size = 0
    for i in range (0,len(sizes)) :
        mean_size += sizes[i]
    mean_size /= len(sizes)

    max_size = max(sizes)
    max_label = sizes.index(max_size)

    img2 = np.zeros(output.shape)

    for i in range(0, nb_components):
        if sizes[i] >= max_size:
            img2[output == i + 1] = 255
        

        
    eroded = cv2.erode(img2,kernel,iterations = 1)
        
    img3 = np.copy(img)

    for y in range(0,img.shape[0]):
        for x in range(0,img.shape[1]):
            if eroded[y][x] == 0 :
                img3[y][x] = 255

    return img3 