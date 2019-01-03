import cv2
import os
import numpy as np
from skimage.feature import hog

range_color_char = 235
sliding_windows_size = [50,100]
percent_step = 50
orientations = 8
pixels_per_cell = (25, 25)
cells_per_block = (2, 2)

def k_means_symbol(path):
    img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
    bounding_box,width = size_window(img)
    sliding_windows = extract_sliding_window(img,bounding_box,sliding_windows_size,percent_step,width)
    feature_vector = find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block)
    return feature_vector

def size_window(img):
    top = []
    left = []
    right = []
    bottom = []
    width = 0
    mask = filter_image(img)
    __ , contours, __ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
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
    width = ratio_width(img)
    return [top[0],bottom[len(bottom)-1],left[0],right[len(right)-1]],width

def filter_image(img):
    mask = cv2.inRange(img,(0),(range_color_char))
    kernel = np.ones((5,2), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    kernel = np.ones((12,12), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel)
    return mask

def ratio_width(img):
    scale = []
    for i in range (img.shape[0]):
        s = 0
        for j in range(img.shape[1]):
            if img[i][j] < range_color_char:
                s += 1
        scale.append(s)
    ratio = max(scale) * 0.4
    j = len(scale)-1
    check_1 = False
    check_2 = False
    for i in range(len(scale)):
        if scale[i] > ratio and not check_1:
            width_top = i
            check_1 = True
        if scale[j] > ratio and not check_2:
            width_bottom = j
            check_2 = True
        if check_1 and check_2:
            break
        j -= 1
    width = int(1.6*(width_bottom-width_top))
    return width

def extract_sliding_window(img,bounding_box,sliding_windows_size,percent_step,width):
    sliding_windows = []
    top,bottom,left,right = bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
    height = bottom - top
    step = int(width*(percent_step/100))
    while left < right:
        temp_img = img[top:bottom, left:left+width]
        if temp_img.shape != (height,width):
            temp_img = cv2.copyMakeBorder(temp_img,0,0,0,width - temp_img.shape[1],cv2.BORDER_CONSTANT,value=255)
        crop_img = cv2.resize(temp_img, (sliding_windows_size[0], sliding_windows_size[1]))
        sliding_windows.append(crop_img)
        left += int(step)
    return sliding_windows

def find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block):
    feature_vector = []
    # path_save_file = path_out + str(j) + "\\"
    # i = 1
    for img in sliding_windows:
        fd, hog_image = hog(img, orientations, pixels_per_cell, cells_per_block, visualize=True, feature_vector=True)
        feature_vector.append(fd)
        # cv2.imwrite(path_save_file + str(i) + "_2.jpg", hog_image)
        # i += 1
    return feature_vector

def make_dir(path,num):
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    for i in range (1,num+1):
        try:
            os.stat(path + str(i))
        except:
            os.mkdir(path + str(i))