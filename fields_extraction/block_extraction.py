import cv2
import numpy as np

"""
The function is used to invert black and white pixels of a binary image
@param image : The image to invert

@return the inverted image
"""
def invert(image) :
    inverted = np.copy(image)
    inverted[image == 255] = 0
    inverted[image == 0] = 255
    return inverted



"""
This function removes the small components in an image
@param image : the image from which we want to remove components
@param size  : the size from which components should be preserved
@param invert_img : true iff the components are currently black

@return the image without the small components
"""
def remove_small(image, size, invert_img=True) :
 
    #invert image if needed
    if invert_img :
        inverted = invert(image)
    else :
        inverted = np.copy(image)

    result = np.copy(image)

    #compute components, ignore background (component 0)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=4)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1


    for i in range(0, nb_components):

        #current  component is too small
        if sizes[i] < size :

            #fill component with background color
            if invert_img :
                result[output == i + 1] = 255
            else :
                result[output == i + 1] = 0

    return result

"""
This function is used to check whether a point is located in a box
@param box   : the box, ((w, x), (y, z)) -> (top_left, bottom_right)
@param point : the point which we want to check the location

@return true iff the point is located in the box
"""
def in_box(box, point) :
    return (point[0] > box[0][0] and point[0] < box[1][0] and point[1] > box[0][1] and point[1] < box[1][1])


"""
This function is used to scale the coordinates of several boxes
@param boxes : the map of boxes to scale
@param scale : the scale to apply to the boxes

@return the map of scaled boxes
"""
def scale_boxes(boxes, scale) :
    for k in boxes.keys() :
        boxes[k] = boxes[k]*scale
    return boxes

#returns the width of the box
def box_width(box) :
    return box[1][1] - box[0][1]

#returns the height of the box
def box_height(box) :
    return box[1][0] - box[0][0]

"""
This function is used to attribute a field to a keypoint
@param keypoints : the keypoints we want to know the fields
@param boxes     : the boxes that define the fields (address, name, etc.)
"""
def define_field(keypoints, boxes) :
    fields = []
    for i, kp in enumerate(keypoints) :
        found = False

        #try all the fields
        for field in boxes.keys() :
            box = boxes[field]

            #keypoint is in box
            if in_box(box, kp) :
                fields.append(field)
                found = True
        
        #keypoints in no fields        
        if not found :
            fields.append('not interesting')
    return fields


"""
This function checks whether the extracted block has the good size
@param block_img_shape   : the shape of the adress block that has been extracted from the image
@param cropped_img_shape : the shape of the cropped scan image
@param margin            : how strictly we want the block shape to match the ideal shape

@return true iff the block has an acceptable shape
"""
def size_ok(block_img_shape, cropped_img_shape, margin = 0.15) :
    return block_img_shape[1] < ((0.21 + margin) * cropped_img_shape[1]) and \
           block_img_shape[1] > ((0.21 - margin) * cropped_img_shape[1]) and \
           block_img_shape[0] < ((0.3  + margin) * cropped_img_shape[0]) and \
           block_img_shape[0] > ((0.3  - margin) * cropped_img_shape[0]) and \
           np.abs((block_img_shape[1] / block_img_shape[0]) - 1.41) < 0.4


"""
this function returns an estimate of the address box in scan coordinates
@param src_point : a template keypoint in the address block box
@param dst_point : the matching scan keypoint
@param boxes     : the boxes map

@return the top-left and right-bottom coordinates of the estimated box
"""
def crop_box(src_point, dst_point, boxes) :
    start_x = int(dst_point[0] - (src_point[0] - boxes['interest_roi'][0][0]))
    start_y = int(dst_point[1] - (src_point[1] - boxes['interest_roi'][0][1]))

    end_x = int(start_x + box_height(boxes['interest_roi']))
    end_y = int(start_y + box_width(boxes['interest_roi']))

    return (start_x, start_y), (end_x, end_y)



"""
This function is used to extract the adress block from a scan image
@param image : the scan image from which to extract the address block
@param scale : the scale difference between scan and template card
@param src   : the template keypoints
@param dst   : the scan keypoints 

@return the cropped address block
@return true iff the block is considered nicely cropped
"""
def apply_block_extraction_on_image(image, scale, src, dst) :

    #image should be black and white
    if len(image.shape) == 3 :
        image = cv2.cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

    org = np.copy(image)

    #binarize image
    _, image = cv2.threshold( image, 250,255,cv2.THRESH_BINARY )

    #remove black boxes
    #image = remove_black_boxes_on_image(image)

    #remove small connected components
    nocc = remove_small(image, int(30 * scale))


    inverted = invert(nocc)
    width = int(15 * scale)

    #erode to get rid of noise
    kernel_b = np.ones((int(3 * scale), int(3 * scale)),np.uint8)
    result_er = cv2.erode(inverted, kernel_b)

    #remove small pixels
    result_er = remove_small(result_er, int(5*scale), invert_img=False)

    #Dilate vertically to link the accents
    kernel_v = np.ones(( width, 1),np.uint8)
    result_v = cv2.dilate(result_er, kernel_v)

    # dilate horizontally to link the characters all together
    kernel_h = np.ones((1, width),np.uint8)
    result_h = cv2.dilate(result_v, kernel_h)



    mask = result_h

    spotted = np.copy(image)
    spotted[mask == 0] = 255

    #keep big enough components
    mask = remove_small(mask, int(500*scale), invert_img=False)


    image[mask == 0] = 255

    # define the ratio for each field
    boxes = dict()
    boxes["interest_roi"] = np.array([[370, 325], [483, 407]])


    #address block is interest_roi
    boxes = scale_boxes(boxes, scale)
    fields = define_field(src, boxes)

    #extract component part
    to_keep = np.copy(mask)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(to_keep, connectivity=4)

    indices = [i for i, f in enumerate(fields) if f == "interest_roi"]

    if 'interest_roi' in fields :
        msg='found_kp'

        #extract component part
        best_tl_x = 0
        best_tl_y = 0
        best_br_x = 0
        best_br_y = 0


        for a, i in enumerate(indices) :

            #compute block box
            p1, p2 = crop_box(src[i], dst[i], boxes)

            if a == 0 :
                best_tl_x = p1[0]
                best_tl_y = p1[1]
                best_br_x = p2[0]
                best_br_y = p2[1]

            #keep combo of largest box
            if p1[0] < best_tl_x :
                best_tl_x = p1[0]

            if p1[1] < best_tl_y :
                best_tl_y = p1[1]

            if p2[0] > best_br_x :
                best_br_x = p2[0]

            if p2[1] > best_br_y :
                best_br_y = p2[1]


        #crop the box
        cropped = org[best_tl_y:best_br_y, best_tl_x:best_br_x]

        #the ratio is always good
        return cropped, True


    else :


        #choose connected component with closest centroid
        centroids = centroids[1:]
        approx_center = np.array([int(image.shape[1]*0.62*scale), int(image.shape[0]*0.81*scale)])
        distances = [np.linalg.norm(np.array(x) - approx_center) for x in centroids]
        if len(distances) == 0 :
            return None
        index = np.argsort(distances)[0] + 1

        #define bounding box of component
        index = index
        left = stats[index, cv2.CC_STAT_LEFT]
        top = stats[index, cv2.CC_STAT_TOP]
        width = stats[index, cv2.CC_STAT_WIDTH]
        height = stats[index, cv2.CC_STAT_HEIGHT]
        comp_roi = org[top:(top + height), left:(left + width)]
        return comp_roi, size_ok(comp_roi.shape, image.shape)
