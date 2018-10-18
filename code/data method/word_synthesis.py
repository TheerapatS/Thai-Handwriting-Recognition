#!/usr/bin/env python3 

import numpy as np
import cv2
import os
from random import randint

dictionary_path = "E:\Work\Thai-Handwriting-Recognition\code\dictionary.txt"
data_path = "E:\Work\\68PersonsBmpChar\\"
test_path = "E:\Work\Thai-Handwriting-Recognition\code\\test.txt"
def main():
    
    with open(dictionary_path, encoding="utf-8-sig") as file:
    # with open(test_path, encoding="utf-8-sig") as file:
        all_words = file.readlines()
    all_words = [x.strip() for x in all_words] 
    # print (all_words)
    for word in all_words:
        char_order = []
        for ascii_w in word:
            print (ord(ascii_w))
            char_order.append(ord(ascii_w))
        word_synthesis(char_order,1)
        break
        print()
    
def create_plain_img (n):
    img = np.zeros([150,50+(40*n)],dtype=np.uint8)
    for i in range(150):
        for j in range(0,len(img[i])):
            img[i][j] = 255
    return img

def check_type_character (num):
    # alphabet 3585 - 3630                                  as 1
    # vowel 3631 - 3641 & 3648 - 3655
    ##### normal vewel 3631 - 3632 & 3634 & 3648 - 3654     as 2
    ##### upper vewel 3633 & 3636 - 3639 & 3655             as 3
    ##### lower vewel 3640 - 3641                           as 4
    ##### special vewel 3635                                as 5
    # tone marks 3656 - 3659                                as 6
    # orthography 3660                                      as 7
    if 3585 <= num <= 3630:
        return 1
    elif (3631 <= num <= 3641) or (3648 <= num <= 3655):
        if (3631 <= num <= 3632) or (3648 <= num <= 3654) or num == 3634:
            return 2
        elif num == 3633 or num == 3655 or (3636 <= num <= 3639):
            return 3
        elif 3640 <= num <= 3641:
            return 4
        elif num == 3635:
            return 5
    elif 3656 <= num <= 3659:
        return 6
    elif num == 3660:
        return 7
    else:
        return 0

def extrack_char (path):
    print (path)
    img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(img,(0),(160))
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
    bounding_box = [top[0],bottom[len(bottom)-1],left[0],right[len(right)-1]]
    crop_img = img[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
    # cv2.imshow("aaa",img)
    # cv2.waitKey()
def word_synthesis (char_order,n):
    # for i in range(n):
    x = 20
    y = 80
    img = create_plain_img(len(char_order))
    for c in char_order:
        path = data_path +  str(c) + "\\"
        n_file = len(next(os.walk(path))[2])
        path_file = path + str(randint(1, n_file)) + ".bmp"
        extrack_char(path_file)
        char_type = check_type_character(c)


main ()