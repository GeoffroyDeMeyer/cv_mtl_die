
import numpy as np
import cv2


def text_detect(img) :


    image = np.copy(img)

    image[img == 0] = 255
    image[img == 255] = 0


    kx_1, ky_1 = (20,1)

    # huge erosion through y_axis to make non-box components vanish
    kernel_1 = np.ones((kx_1, ky_1),np.uint8)
    eroded = cv2.erode(image, kernel_1, iterations=1 )


    kx_2, ky_2 = (1,50)

    # huge dilatation through x-axis to make non-box components vanish
    kernel_2 = np.ones((kx_2, ky_2),np.uint8)
    dilated = cv2.dilate(eroded, kernel_2, iterations=1 )




    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=4)




    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.

    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    sizes = list(sizes)


    max_size = max(sizes)
    max_label = sizes.index(max_size)


    img2 = np.zeros(output.shape)

    img2[output == max_label +1] = 255
        
    eroded = cv2.erode(img2,kernel_2,iterations = 1)

    
    
    
    #Réservé à la partie thai name    
    #kx_3, ky_3 = (60,1)
    #kernel_1 = np.ones((kx_3, ky_3),np.uint8)
    
    dilated = cv2.dilate(eroded,kernel_1,iterations = 1) 


    img3 = np.copy(img)

    for y in range(0,img.shape[0]):
        for x in range(0,img.shape[1]):
            if dilated[y][x] == 0 :
                img3[y][x] = 255
    
    return img3

