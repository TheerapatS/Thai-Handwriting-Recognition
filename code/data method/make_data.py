import numpy as np
import cv2
import os
import random as rng
from PIL import Image

if __name__ == "__main__":
    # src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # src_gray = cv2.blur(src_gray, (3,3))
    # canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    image_x_array = [390, 1417, 1560, 1707, 1852, 1997]
    path_in = 'D:\Work\Project[261491 & 261492]\\training_set\\68PersonsBmp\\'
    path_out = 'D:\Work\Project[261491 & 261492]\\training_set\\68PersonsBmpChar\\'
    img = cv2.imread(path_in + "is01001.bmp", 0)
    out = np.zeros(shape=(img.shape[0],img.shape[1]))
    imm,contours, hierarchy =   cv2.findContours(img.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    screenCnt = None
    
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("Game Boy Screen", image)
    cv2.waitKey(0)
    print (hierarchy)
    for i in contours:
        print(i)
    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # for i in range(len(contours)):
    #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    #     cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    # cv2.imshow('Contours', drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for file in os.listdir(path_in):
    #     name_count = 0
    #     # img = cv2.imread(path_in + str(file), 0)
    #     print (file)
    #     out = np.zeros(shape=(60,60))
    #     for image_x in image_x_array:
    #         skip = 0
    #         for i in range (0,img.shape[1]):
    #             skip += 1
    #             if img[image_x][i] <= 230 and skip >= 30:
    #                 if 35 < i < img.shape[1]-40:
    #                     ii = 0
    #                     for x in range(image_x-22,image_x+38):
    #                         jj = 0
    #                         for y in range(i-23,i+37):
    #                             out[ii][jj] = img[x][y]
    #                             jj += 1
    #                         ii += 1
    #                     cv2.imwrite(path_out + file + "_" + str(name_count).zfill(4) + ".png" ,out)
    #                     name_count += 1
    #                     skip = 0

                        # img2 = cv2.imread('test.png',0)
                        # cv2.imshow('image2',img2)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
        # small = cv2.resize(img, (0,0), fx=0.35, fy=0.35) 
        # cv2.imshow('small',small)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()