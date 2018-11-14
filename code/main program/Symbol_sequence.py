import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.externals import joblib

# from skimage.feature import hog
# from skimage import data, exposure
w = -1
h = -1
range_color_char = 235
def main ():
    path = "D:\Work\Project\Dictionary_word\\"
    path_out = "D:\Work\Project\\training_set\Dictionary_sliding\\"
    path_out_test = "D:\Work\Project\\training_set\Symbol_Test_out\\"
    # file = "01.jpg"
    
    sliding_windows_size = [50,100]
    percent_step = 50
    orientations = 8
    pixels_per_cell = (25, 25)
    number_clusters = 30
    train_all_feature_vector = []
    test_all_feature_vector = []
    cells_per_block = (2, 2)
    train_data = []
    # img = cv2.cvtColor(cv2.imread("D:\Work\Project\Dictionary_word\\2\\1512.bmp"),cv2.COLOR_BGR2GRAY)
    # find_size_slide(img)
    try:
        os.stat(path_out)
    except:
        os.mkdir(path_out)
    for i in range (1,501):
        try:
            os.stat(path_out + str(i))
        except:
            os.mkdir(path_out + str(i))
    j = 1
    print (ord('\n'))
    for folder in os.listdir(path):
        sub_path = path + str(folder) + "\\"
        # sub_path = "D:\Work\Project\\training_set\Symbol_Test\\"
        for file in os.listdir(sub_path):
            print ("Extract feature of sliding window " + str(folder) + "\\" + file)
            img = cv2.cvtColor(cv2.imread(sub_path+file),cv2.COLOR_BGR2GRAY)
            bounding_box,width = find_size_slide(img)
            sliding_windows = extract_sliding_window(img,bounding_box,sliding_windows_size,percent_step,width)
            path_save_file = path_out + str(j) + "\\"
            # for i in range(1,len(sliding_windows)+1):
            #     cv2.imwrite(path_save_file + str(i) + "_1.jpg", sliding_windows[i-1])
            feature_vector = find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block,j,path_out)
            # print (len(feature_vector))
            print ("extracted feature for " + file)
            train_all_feature_vector.append(feature_vector)
            for i in feature_vector:
                train_data.append(i)
            j += 1
    # for i in feature_vector:
    #     print(i)

    print ("ready for train model")
    model = k_means(number_clusters,train_data)
    model = save_load_model(model,False)
    # model = save_load_model(0,True)
    dict_symspell = save_class_data(model,train_all_feature_vector,"Dictionary_word_class_label.txt")
    cluster_histo(number_clusters,train_all_feature_vector,model)
    save_dictionary_symspell("Dictionary_symspell.txt",dict_symspell)
    try:
        os.stat(path_out_test)
    except:
        os.mkdir(path_out_test)
    for i in range (1,133):
        try:
            os.stat(path_out_test + str(i))
        except:
            os.mkdir(path_out_test + str(i))
    j = 1
    data_test_path = "D:\Work\Project\\training_set\Symbol_Test\\"
    for file in os.listdir(data_test_path):
        print ("Train Symbol_Test\\" + file)
        img = cv2.cvtColor(cv2.imread(data_test_path+file),cv2.COLOR_BGR2GRAY)
        bounding_box,width = find_size_slide(img)
        sliding_windows = extract_sliding_window(img,bounding_box,sliding_windows_size,percent_step,width)
        path_save_file = path_out_test + str(j) + "\\"
        # for i in range(1,len(sliding_windows)+1):
        #     cv2.imwrite(path_save_file + str(i) + "_1.jpg", sliding_windows[i-1])
        feature_vector = find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block,j,path_out)
        # print (len(feature_vector))
        # print ("extracted feature for " + file)
        test_all_feature_vector.append(feature_vector)
        # for i in feature_vector:
        #     train_data.append(i)
        j += 1
    
    print ("trained model")
    predict_class(model,test_all_feature_vector)
    __ = save_class_data(model,test_all_feature_vector,"Test_data_class_label.txt")
    # save_dictionary_symspell("dictionary_symspell.txt",dict_symspell)
    print ("all done!!")

def save_load_model(model,load_flag):
    filename = 'KMeans_model.sav'
    if load_flag:
        loaded_model = joblib.load(filename)
        return loaded_model
    else:
        joblib.dump(model, filename)
        return model

def find_size_slide(img):
    mask = cv2.inRange(img,(0),(range_color_char))
    kernel = np.ones((5,2), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    kernel = np.ones((12,12), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel)
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
    # print (top,left,bottom,right)
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
    return [top[0],bottom[len(bottom)-1],left[0],right[len(right)-1]],width


def find_hog(sliding_windows,orientations,pixels_per_cell,cells_per_block,j,path_out):
    feature_vector = []
    path_save_file = path_out + str(j) + "\\"
    i = 1
    maxx = 0
    for img in sliding_windows:
        fd, hog_image = hog(img, orientations, pixels_per_cell, cells_per_block, visualize=True, feature_vector=True)
        feature_vector.append(fd)
        # cv2.imwrite(path_save_file + str(i) + "_2.jpg", hog_image)
        i+=1
    return feature_vector

def extract_sliding_window(img,bounding_box,sliding_windows_size,percent_step,width):
    sliding_windows = []
    top,bottom,left,right = bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
    # width = int(((bottom-top)/2))
    height = bottom - top
    step = int(width*(percent_step/100))
    while left < right:
        temp_img = img[top:bottom, left:left+width]
        if temp_img.shape != (height,width):
            # print(width - temp_img.shape[1])
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
        # print ()

def save_class_data(model,all_feature_vector,file):
    f = open(file, "w")
    dict_symspell = {}
    for i in range(len(all_feature_vector)):
        # for j in range(len(all_feature_vactor[i])):
            predicted_label = model.predict(all_feature_vector[i])
            s = ''
            for j in predicted_label:
                f.write(str(j) + " ")
                s += chr(j+33)
            if s in dict_symspell:
                dict_symspell[s] += 1
            else :
                dict_symspell[s] = 1
            f.write("[" + str(len(predicted_label)) + "]\n")
    f.close()
    return dict_symspell

def cluster_histo(number_clusters,all_feature_vector,model):
    histo = np.zeros(number_clusters)
    for i in range(len(all_feature_vector)):
        predicted_label = model.predict(all_feature_vector[i])
        for j in predicted_label:
            histo[j] += 1
    plt.plot(np.arange(number_clusters),histo, color="green")
    plt.show()
    # plt.close()

def save_dictionary_symspell(file,dict_symspell):
    f = open(file,"w")
    for i in dict_symspell:
        f.write(i + " 10000\n")
    f.close()

main()