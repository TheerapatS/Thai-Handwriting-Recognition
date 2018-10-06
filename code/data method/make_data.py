import numpy as np
import cv2
import os

from operator import itemgetter

def crop_img (img,x):
    shape = img.shape
    w = 200
    img = img[x:x+w, 0:shape[0]]
    # cv2.imshow("cropped", cv2.resize(img,(0,0),fx=0.4,fy=0.4))
    # cv2.waitKey(0)
    mask = cv2.inRange(img,(0,0,0),(200,200,200))
    temp = mask
    _ , contours, _ = cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        img = img[y+8:y+h-8, x+8:x+w-8]
    return img

def crop_alphabet(img,x,y,w,h):
    w_h_img = img.shape
    # print (w_h_img)
    y_start = max(0,y-5)
    y_stop = min(y+h+5,w_h_img[0])
    x_start = max(0,x-5)
    x_stop = min(x+w+5,w_h_img[1])
    img = img[y_start:y_stop, x_start:x_stop]
    return [x+(w/2),img]

def find_alphabet(img,path_out,set_count):
    x_start = [0,53,105,169,219,284,346,405,460,535,598,653,703,760,816,878,950,1012,1071,1136,1198,1257,1312,1378]
    x_stop = [48,100,164,214,279,341,400,455,530,593,648,698,755,811,873,945,1007,1066,1132,1193,1252,1307,1373]
    mask = cv2.inRange(img,(0,0,0),(220,220,220))
    kernel = np.ones((11,11),np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    # kernel = np.ones((13,13),np.float32)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((11,11),np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((0,7),np.float32)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # print (mask[0][0])
    # for i in x_stop:
    #     for j in range(0,img.shape[0]):
    #         img[j][i][0] = 0
    #         img[j][i][1] = 0
    #         img[j][i][2] = 255
    # cv2.imshow("img",cv2.resize(img,(0,0),fx=0.5,fy=0.5))
    # cv2.waitKey(0)
    temp = mask
    _ , contours, _ = cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        list_of_alphabet = []
        # cv2.imshow("img",cv2.resize(img,(0,0),fx=0.5,fy=0.5))
        # cv2.waitKey(0)
        for i in contours:
            x,y,w,h = cv2.boundingRect(i)
            if 200 <= (w*h) <= 8000 and 10 <= w <= 90:
                list_of_alphabet.append(crop_alphabet(img,x,y,w,h))
                # for j in range(y,y+h):
                #     for k in range(x,x+w):
                #         img[j][k][0] = 0
                #         img[j][k][1] = 0
                #         img[j][k][2] = 255
        # cv2.imshow("img",cv2.resize(img,(0,0),fx=0.5,fy=0.5))
        # cv2.waitKey(0)
    
    list_of_alphabet = sorted(list_of_alphabet,key=lambda x: x[0])
    for i in list_of_alphabet:
        colum = -1
        c = 0
        for j in range(0,len(x_stop)):
            if i[0]<=x_stop[j] and i[0] >= x_start[j]:
                colum = j
        if colum == -1:
            path_out_char = path_out + "not_know"
        else :
            if set_count == 0 or set_count == 2:
                path_out_char = path_out + str(161+colum)
            elif set_count == 1 or set_count == 3:
                path_out_char = path_out + str(161+colum+23)
            elif set_count == 4 or set_count == 5:
                path_out_char = path_out + str(161+colum+46)
        for file in os.listdir(path_out_char):
            c += 1  
        # cv2.imshow("img",cv2.resize(i[1],(0,0),fx=1,fy=1))
        # cv2.waitKey(0)
        cv2.imwrite(path_out_char + "\\" + '{0:5}'.format(c+1) + '.bmp', cv2.resize(i[1],(60,60)))
    
if __name__ == "__main__":
    floder = '68PersonsBmp'
    path_in = 'D:\Work\Project\\training_set\\' + floder + '\\'
    path_out = 'D:\Work\Project\\training_set\\' + floder + 'Char\\'
    try:
        os.stat(path_out)
    except:
        os.mkdir(path_out)
    for i in range (0,70):
        if i < 69:
            try:
                os.stat(path_out + str(161+i))
            except:
                os.mkdir(path_out + str(161+i))
        else :
            try:
                os.stat(path_out + "not_know")
            except:
                os.mkdir(path_out + "not_know")
    set_count = 0
    x = [1320,1450,1600,1750,1900,2050]
    for file in os.listdir(path_in):
        print (file)
        img = cv2.imread(path_in + file)
        set_count = 0
        for i in x:
            # print(set_count)
            temp = crop_img(img,i)
            # cv2.imshow("img",cv2.resize(temp,(0,0),fx=0.5,fy=0.5))
            # cv2.waitKey(0)
            find_alphabet(temp,path_out,set_count)
            set_count += 1

