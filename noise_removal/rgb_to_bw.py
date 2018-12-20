import cv2

# Convert color to grey scale image

def color_to_bw(img) :
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    return bw_img