import cv2

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

# from skimage.feature import hog
# from skimage import data, exposure
w = -1
h = -1
def main ():
    number_of_sliding_window = 50
    path = "D:\Work\Project\\training_set\Symbol_Test\\"
    file = "01.jpg"
    img = cv2.cvtColor(cv2.imread(path+file),cv2.COLOR_BGR2GRAY)
    bounding_box = find_size_slide(img)
    sliding_windows_size = [50,100]
    percent_step = 50
    orientations=16
    pixels_per_cell=(10, 10)
    cells_per_block=(3, 1)


    sliding_windows = extract_sliding_window(img,bounding_box,sliding_windows_size,percent_step)
    find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block)

def find_size_slide(img):
    mask = cv2.inRange(img,(0),(180))
    kernel = np.ones((5,2), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    kernel = np.ones((12,12), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    temp = mask.copy()
    contourmask , contours, hierarchy = cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    top = []
    left = []
    right = []
    bottom = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        top.append(y)
        left.append(x)
        right.append(x+w)
        bottom.append(y+h)
    top.sort()
    left.sort()
    right.sort()
    bottom.sort()
    return [top[0],bottom[len(bottom)-1],left[0],right[len(right)-1]]

def find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block):
    for img in sliding_windows:
        fd, hog_image = hog(img, orientations=16, pixels_per_cell=(, 256),
                        cells_per_block=(1, 1), visualize=True, feature_vector=True)


def extract_sliding_window(img,bounding_box,sliding_windows_size,percent_step):
    sliding_windows = []
    top,bottom,left,right = bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
    width = int(((bottom-top)/2))
    height = bottom - top
    step = int(width*(percent_step/100))
    while left < right:
        temp_img = img[top:bottom, left:left+width]
        if temp_img.shape != (height,width):
            temp_img = cv2.copyMakeBorder(temp_img,0,0,0,width - temp_img.shape[1],cv2.BORDER_CONSTANT,value=255)
        crop_img = cv2.resize(temp_img, (sliding_windows_size[0], sliding_windows_size[1]))
        # cv2.imshow("test",crop_img)
        # cv2.waitKey()
        sliding_windows.append(crop_img)
        left += int(step)
    return sliding_windows
main()