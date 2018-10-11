import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans

# from skimage.feature import hog
# from skimage import data, exposure
w = -1
h = -1
def main ():
    number_of_sliding_window = 50
    path = "E:\Work\Symbol_Test\\"
    path_out = "E:\Work\\Symbol_Test_out\\"
    # file = "01.jpg"
    
    
    sliding_windows_size = [50,100]
    percent_step = 70
    orientations = 8
    pixels_per_cell = (25, 25)
    cells_per_block = (2, 2)
    number_clusters = 25
    all_feature_vector = []
    train_data = []

    for i in range (1,132):
        try:
            os.stat(path_out + str(i))
        except:
            os.mkdir(path_out + str(i))
    j = 1
    for file in os.listdir(path):
        # print (file)
        print (file)
        img = cv2.cvtColor(cv2.imread(path+file),cv2.COLOR_BGR2GRAY)
        bounding_box = find_size_slide(img)
        sliding_windows = extract_sliding_window(img,bounding_box,sliding_windows_size,percent_step)
        path_save_file = path_out + str(j) + "\\"
        for i in range(1,len(sliding_windows)+1):
            cv2.imwrite(path_save_file + str(i) + "_1.jpg", sliding_windows[i-1])
        print ("made sliding window for " + file)
        feature_vector = find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block,j,path_out)
        # print (len(feature_vector))
        print ("extracted feature for " + file)
        all_feature_vector.append(feature_vector)
        for i in feature_vector:
            train_data.append(i)
        j += 1
    # for i in feature_vector:
    #     print(i)
    print ("ready for train model")
    model = k_means(number_clusters,train_data)
    print ("trained model")
    predict_class(model,all_feature_vector)
    print ("all done!!")

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

def find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block,j,path_out):
    feature_vector = []
    path_save_file = path_out + str(j) + "\\"
    i = 1
    maxx = 0
    for img in sliding_windows:
        fd, hog_image = hog(img, orientations, pixels_per_cell, cells_per_block, visualize=True, feature_vector=True)
        # print (type(hog_image))
        feature_vector.append(fd)
        
        # cv2.imshow("img",img)
        # cv2.moveWindow("img", 50,100)
        # cv2.imshow("hog",hog_image)
        # cv2.moveWindow("hog", 200,100)
        # cv2.waitKey()
        # for p in hog_image:
            # for q in p:
                # q = int(q*255)
        cv2.imwrite(path_save_file + str(i) + "_2.jpg", hog_image)
        i+=1
        for p in hog_image:
            for q in p:
                maxx = max(q,maxx)
    print (maxx)
    return feature_vector

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

def k_means(number_clusters,data):
    model = KMeans(n_clusters=number_clusters)
    model.fit(np.array(data))
    # all_predictions = model.predict(np.array(data))
    # print(all_predictions)
    return model

def predict_class(model,all_feature_vactor):
    for i in range(len(all_feature_vactor)):
        # for j in range(len(all_feature_vactor[i])):
            predicted_label = model.predict(all_feature_vactor[i])
            print (predicted_label)
        # print ()

main()