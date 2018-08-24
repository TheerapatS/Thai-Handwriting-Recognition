import numpy as np
import cv2
import os
from PIL import Image

if __name__ == "__main__":
    image_x_array = [390, 1417, 1560, 1707, 1852, 1997]
    path_in = 'D:\Work\Project[261491 & 261492]\\training_set\\68PersonsBmp\\'
    path_out = 'D:\Work\Project[261491 & 261492]\\training_set\\68PersonsBmpChar\\'
    for file in os.listdir(path_in):
        name_count = 0
        img = cv2.imread(path_in + str(file), 0)
        print (file)
        out = np.zeros(shape=(60,60))
        for image_x in image_x_array:
            skip = 0
            for i in range (0,img.shape[1]):
                skip += 1
                if img[image_x][i] <= 230 and skip >= 30:
                    if 35 < i < img.shape[1]-40:
                        ii = 0
                        for x in range(image_x-22,image_x+38):
                            jj = 0
                            for y in range(i-23,i+37):
                                out[ii][jj] = img[x][y]
                                jj += 1
                            ii += 1
                        cv2.imwrite(path_out + file + "_" + str(name_count).zfill(4) + ".png" ,out)
                        name_count += 1
                        # img2 = cv2.imread('test.png',0)
                        # cv2.imshow('image2',img2)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        skip = 0
        # small = cv2.resize(img, (0,0), fx=0.35, fy=0.35) 
        # cv2.imshow('small',small)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()