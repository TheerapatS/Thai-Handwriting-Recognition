#!/usr/bin/env python3 

import numpy as np
import cv2
import os
from random import randint
from scipy.ndimage import rotate

dictionary_path = "D:\Work\Project\Thai-Handwriting-Recognition\code\dictionary.txt"
data_path = "D:\Work\Project\\training_set\\68PersonsBmpChar\\"
test_path = "D:\Work\Project\Thai-Handwriting-Recognition\code\\test.txt"
path_out = "D:\Work\Project\Dictionary_word\\"
range_color_char = 235
number_of_word = 100
rotate_rand_size = 5
squeeze_flag = True
squeeze_min_size = 0
squeeze_max_size = 5
fade_flag = True
fade_ratio = 15
erode_dilate_flag = False
kernel_size = 3
margin_min_size = -5
margin_max_size = 5
def main():
    word_count = 0
    with open(dictionary_path, encoding="utf-8-sig") as file:
    # with open(test_path, encoding="utf-8-sig") as file:
        all_words = file.readlines()
    all_words = [x.strip() for x in all_words] 
    for word in all_words:
        word_count = word_count + 1
        print ("Word number :: " + str(word_count))
        char_order = []
        for ascii_w in word:
            char_order.append(ord(ascii_w))
        word_synthesis(char_order,number_of_word,word_count)
        print ()
    
def create_plain_img (n):
    img = np.zeros([150,50+(60*n)],dtype=np.uint8)
    for i in range(150):
        for j in range(0,len(img[i])):
            img[i][j] = 255
    return img

def check_type_character (num):
    # alphabet          3585 - 3630                             as 1
    # vowel             3631 - 3641 & 3648 - 3655
        # normal vewel  3631 - 3632 & 3634 & 3648 - 3654        as 2
        # upper vewel   3633 & 3636 - 3639 & 3655               as 3
        # lower vewel   3640 - 3641                             as 4
        # special vewel 3635                                    as 5
    # tone marks        3656 - 3659                             as 6
    # orthography       3660                                    as 7
    if 3585 <= num <= 3630: # alphabet 
        return 1
    elif (3631 <= num <= 3641) or (3648 <= num <= 3655):
        if (3631 <= num <= 3632) or (3648 <= num <= 3654) or num == 3634: # normal vowel
            return 2
        elif num == 3633 or num == 3655 or (3636 <= num <= 3639): # upper vewel
            return 3
        elif 3640 <= num <= 3641: # lower vewel
            return 4
        elif num == 3635: # special vowel
            return 5
    elif 3656 <= num <= 3659: # tone mask
        return 6
    elif num == 3660: # orthography
        return 7
    else:
        return 0

def resize_img (img,char_type,c):
    if char_type == 1: # alphabet
        if c == 3620 or c == 3622: # tall alphabet
            r = randint(-2,2)
            if squeeze_flag :
                s = randint(squeeze_min_size,squeeze_max_size)
            else:
                s = 0
            img = cv2.resize(img, (40 + r + s,45 + r)) 
        else:
            r = randint(-2,2)
            if squeeze_flag :
                s = randint(squeeze_min_size,squeeze_max_size)
            else:
                s = 0
            img = cv2.resize(img, (40 + r + s,40 + r)) 
    elif char_type == 2: # normal vowel
        if 3650 <= c <= 3652: # tall vowel
            r = randint(-2,2)
            if squeeze_flag :
                s = randint(squeeze_min_size,squeeze_max_size)
            else:
                s = 0
            img = cv2.resize(img, (40 + r + s,55 + r))
        else : # other
            r = randint(-2,2)
            if squeeze_flag :
                s = randint(squeeze_min_size,squeeze_max_size)
            else:
                s = 0 
            img = cv2.resize(img, (40 + r + s,40 + r))
    elif char_type == 3: # upper vowel
        r = randint(-2,2)
        if squeeze_flag :
                s = randint(squeeze_min_size,squeeze_max_size)
        else:
                s = 0
        img = cv2.resize(img, (40 + r + s,25 + r))
    elif char_type == 4: # lower vowel
        if c == 3640: # u
            r = randint(-2,2)
            img = cv2.resize(img, (20 + r + s,13 + r))
        else: # uu
            r = randint(-2,2)
            img = cv2.resize(img, (20 + r,20 + r))
    # elif char_type == 5:
    elif char_type == 6: # tone mask
        if c == 3656: # eak
            r = randint(-2,2)
            img = cv2.resize(img, (5 + r,15 + r))
        else : # other
            r = randint(-2,2)
            img = cv2.resize(img, (25 + r,15 + r)) 
    elif char_type == 7: # orthography
        r = randint(-2,2)
        if squeeze_flag :
                s = randint(squeeze_min_size,squeeze_max_size)
        else:
                s = 0
        img = cv2.resize(img, (25 + r + s,25 + r))
    return img

def extrack_char (path,char_type,c):
    img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
    bounding_box = find_bounding_box(img)
    t_img = img[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
    img = resize_img(t_img,char_type,c)
    rotate_rand = randint(-1 * rotate_rand_size, rotate_rand_size)
    rotate_img = rotate(img, rotate_rand)
    # cv2.imshow("uimg",rotate_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    bounding_box = find_bounding_box(rotate_img)
    if bounding_box[0]-1 >= 0:
        bounding_box[0] = bounding_box[0]-1
    if bounding_box[1]+1 < img.shape[0]:
        bounding_box[1] = bounding_box[1]+1
    if bounding_box[2]-1 >= 0:
        bounding_box[2] = bounding_box[2]-1
    if bounding_box[3]+1 < img.shape[1]:
        bounding_box[3] = bounding_box[3]+1
    
    if erode_dilate_flag :
        e_or_d = randint(0,1)
        k_size = randint(0,kernel_size)
        kernel = np.ones((k_size,k_size),np.uint8)
        if e_or_d == 0:
            rotate_img = cv2.erode(rotate_img,kernel,iterations = 1)
        else :
            rotate_img = cv2.dilate(rotate_img,kernel,iterations = 1)
    if fade_flag:
        fade_init = randint(0,fade_ratio) 
        for i in range(rotate_img.shape[0]):
            for j in range(rotate_img.shape[1]):
                if rotate_img[i][j] < range_color_char:
                    rotate_img[i][j] = rotate_img[i][j] + fade_init
    crop_img = rotate_img[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
    return crop_img

def find_bounding_box (img):
    mask = cv2.inRange(img,(0),(0))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 255:
                img[i][j] = mask[i][j]
    mask = cv2.inRange(img,(0),(range_color_char))
    temp = mask.copy()
    __, contours, __ = cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
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

def word_synthesis (char_order,n,word_count):
    now_percent = 0
    print ("::" ,end = " ")
    temp_n = n
    while n > 0:
        x = 60
        y = 20
        position_order = []
        type_order = []
        img = create_plain_img(len(char_order))
        for c in char_order:
            path = data_path +  str(c) + "\\"
            # print (path)
            n_file = len(next(os.walk(path))[2])
            path_file = path + str(randint(1, n_file)) + ".bmp"
            char_type = check_type_character(c)
            if char_type != 0:
                char_img = extrack_char(path_file,char_type,c)
                type_order.append(char_type)
                if char_type == 1 or char_type == 2: # normal char and vowel
                    if 3650 <= c <= 3652 or c == 3611 or c == 3615 or c == 3613 or c == 3628: # tall vowel & tall char
                        x = x - 15
                    for i in range(char_img.shape[0]):
                        for j in range(char_img.shape[1]):
                            img[x+i][y+j] = min(char_img[i][j],img[x+i][y+j])
                    if 3650 <= c <= 3652 or c == 3611 or c == 3615 or c == 3613 or c == 3628: # tall vowel & tall char
                        x = x + 15
                    position_order.append([x,y])
                    y = y + char_img.shape[1] + randint(margin_min_size,margin_max_size)
                elif char_type == 3: # upper vowel
                    if c != 3633: # not mai hun a gad
                        x_temp = 48
                        y_temp = position_order[len(position_order)-1][1]
                        for i in range(char_img.shape[0]):
                            for j in range(char_img.shape[1]):
                                img[x_temp+i][y_temp+j] = min(char_img[i][j],img[x_temp+i][y_temp+j])
                    else: # upper vowel
                        x_temp = 48
                        y_temp = position_order[len(position_order)-1][1] + 10
                        for i in range(char_img.shape[0]):
                            for j in range(char_img.shape[1]):
                                img[x_temp+i][y_temp+j] = min(char_img[i][j],img[x_temp+i][y_temp+j])
                    position_order.append([x_temp,y_temp])
                elif char_type == 4: # lower vowel
                    x_temp = 95
                    y_temp = position_order[len(position_order)-1][1] + 22
                    for i in range(char_img.shape[0]):
                            for j in range(char_img.shape[1]):
                                img[x_temp+i][y_temp+j] = min(char_img[i][j],img[x_temp+i][y_temp+j])
                    position_order.append([x_temp,y_temp])
                # elif char_type == 5:
                elif char_type == 6: # tone mask
                    if type_order[len(type_order)-1] == 3: # before is upper vowel
                        x_temp = 30
                        y_temp = position_order[len(position_order)-1][1] + 15
                        for i in range(char_img.shape[0]):
                                for j in range(char_img.shape[1]):
                                    img[x_temp+i][y_temp+j] = min(char_img[i][j],img[x_temp+i][y_temp+j])
                        position_order.append([x_temp,y_temp])
                    else: # before is lower vowel
                        x_temp = 48
                        y_temp = position_order[len(position_order)-1][1] + 25
                        for i in range(char_img.shape[0]):
                                for j in range(char_img.shape[1]):
                                    img[x_temp+i][y_temp+j] = min(char_img[i][j],img[x_temp+i][y_temp+j])
                        position_order.append([x_temp,y_temp])
                elif char_type == 7: # orthography
                    x_temp = 42
                    y_temp = position_order[len(position_order)-1][1] + 5
                    for i in range(char_img.shape[0]):
                            for j in range(char_img.shape[1]):
                                img[x_temp+i][y_temp+j] = min(char_img[i][j],img[x_temp+i][y_temp+j])
                    position_order.append([x_temp,y_temp])
        cut_and_save_img(img,n,word_count)

        percent = (((temp_n - n)/temp_n)*100)
        while percent-now_percent >= 2:
            now_percent = now_percent + 2
            print("#",end = "")
        n = n - 1


def cut_and_save_img (img,n,word_count):
    bounding_box = find_bounding_box(img)
    crop_img = img[bounding_box[0]-3:bounding_box[1]+3, bounding_box[2]-3:bounding_box[3]+3]
    created_folder(path_out,word_count)
    cv2.imwrite(path_out + str(word_count) + "\\" + str(n) + ".bmp", crop_img)

def created_folder(path_out,word_count):
    try:
        os.stat(path_out)
    except:
        os.mkdir(path_out)
    try:
        os.stat(path_out + str(word_count) + "\\")
    except:
        os.mkdir(path_out + str(word_count) + "\\")
main ()