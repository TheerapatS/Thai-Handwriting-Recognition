#!/usr/bin/env python3 

import numpy as np
import cv2
# alphabet 3585 - 3630
# vowel 3631 - 3641 & 3648 - 3655
#  
def main():
    dictionary_path = "E:\Work\Thai-Handwriting-Recognition\code\dictionary.txt"
    data_path = "E:\Work\\68PersonsBmpChar\\"
    with open(dictionary_path, encoding="utf-8-sig") as file:
        all_words = file.readlines()
    all_words = [x.strip() for x in all_words] 
    # print (all_words)
    for word in all_words:
        word_form = []
        for ascii_w in word:
            word_form.append(ord(ascii_w))
        img = create_plain_img(len(word_form))
    
def create_plain_img (n):
    img = np.zeros([150,50+(40*n)],dtype=np.uint8)
    return img

main ()