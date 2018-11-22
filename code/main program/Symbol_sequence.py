import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import codecs
from symspellpy.symspellpy import SymSpell, Verbosity 

# from skimage.feature import hog
# from skimage import data, exposure
w = -1
h = -1
range_color_char = 235
def main ():
    path = "D:\Work\Project\Dictionary_word\\"
    path_out = "D:\Work\Project\\training_set\Dictionary_sliding\\"
    path_out_test = "D:\Work\Project\\training_set\Symbol_Test_out\\"
    path_dict = "D:\Work\Project\Thai-Handwriting-Recognition\code\dictionary.txt"
    path_predict_word = "D:\Work\Project\Thai-Handwriting-Recognition\code\predict_word.txt"
    # file = "01.jpg"
    # f_predict = codecs.open(path_predict_word,"w","utf-8")
    f_dict = codecs.open(path_dict, encoding='utf-8')
    dictionary_size = 0
    dictionary_words = []
    for line in f_dict :
        # print (line.encode('utf-8'))
        dictionary_size += 1
        dictionary_words.append(line)

    print (dictionary_size)
    sliding_windows_size = [50,100]
    percent_step = 50
    orientations = 8
    pixels_per_cell = (25, 25)
    number_clusters = 20
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
    # print (ord('\n'))
    other_words = []
    for folder in os.listdir(path):
        sub_path = path + str(folder) + "\\"
        # sub_path = "D:\Work\Project\\training_set\Symbol_Test\\"
        t = []
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
            t.append(feature_vector)
            for i in feature_vector:
                train_data.append(i)
            j += 1
        other_words.append(t)
    # for i in feature_vector:
    #     print(i)
    print ("ready for train model")

    ##### save use this #####
    # model = k_means(number_clusters,train_data)
    # model = save_load_model(model,False,number_clusters)

    ##### load use this #####
    model = save_load_model(0,True,number_clusters)

    dict_symspell,symbol_to_word_ref = save_class_data(model,train_all_feature_vector,"Dictionary_word_class_label.txt",dictionary_size)
    # cluster_histo_each_word(number_clusters,other_words,model,True)
    # cluster_histo(number_clusters,train_all_feature_vector,model,True)
    cal_edit_dist(other_words,model,number_clusters)
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

    initial_capacity = len(dict_symspell)
    max_edit_distance_dictionary = 6
    prefix_length = 7
    sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,prefix_length)
    dictionary_symspell_path = os.path.join(os.path.dirname(__file__),"Dictionary_symspell.txt")
    term_index = 0
    count_index = 1
    if not sym_spell.load_dictionary(dictionary_symspell_path, term_index, count_index):
        print("Dictionary file not found")

    j = 1
    name_of_file = []
    data_test_path = "D:\Work\Project\\training_set\Symbol_Test\\"
    for file in os.listdir(data_test_path):
        print ("Train Symbol_Test\\" + file)
        name_of_file.append(file)
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
    # cluster_histo_each_word(number_clusters,[test_all_feature_vector],model,False)
    # cluster_histo(number_clusters,test_all_feature_vector,model,False)
    print ("trained model")
    predict_class(model,test_all_feature_vector,symbol_to_word_ref,sym_spell,dictionary_words,path_predict_word,name_of_file)
    __,__ = save_class_data(model,test_all_feature_vector,"Test_data_class_label.txt",dictionary_size)
    # save_dictionary_symspell("dictionary_symspell.txt",dict_symspell)
    print ("all done!!")

def save_load_model(model,load_flag,number_clusters):
    filename = 'KMeans_model_'+str(number_clusters)+'.sav'
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

def predict_class(model,all_feature_vactor,symbol_to_word_ref,sym_spell,dictionary_words,path_predict_word,name_of_file):
    max_edit_distance_lookup = 5
    ans_words = []
    for i in range(len(all_feature_vactor)):
        predicted_label = model.predict(all_feature_vactor[i])
        s = ''
        predict_word = {}
        for j in predicted_label:
             s += chr(j+33)
        suggestion_verbosity = Verbosity.CLOSEST
        suggestions = sym_spell.lookup(s,suggestion_verbosity,max_edit_distance_lookup)
        # print ("real " + s)
        for suggestion in suggestions:
            if symbol_to_word_ref[suggestion.term] not in predict_word:
                predict_word[symbol_to_word_ref[suggestion.term]] = symbol_to_word_ref[suggestion.term]
        print(predict_word)
        if len(predict_word) == 1:
            for key in predict_word:
                # print (predict_word[key])
                # print (type(predict_word[key]))
                ans_words.append(name_of_file[i] + " " + dictionary_words[predict_word[key]-1])
        elif len(predict_word) != 0:
            co = 1
            for key in predict_word:
                ans_words.append(str(co) + " " + name_of_file[i] + " " + dictionary_words[predict_word[key]-1])
                co += 1
        else:
            ans_words.append(name_of_file[i] + " " + "None \r\n")
        
    # print (ans_words)
    with codecs.open(path_predict_word, "w", "utf-8-sig") as f_predict:
        for i in ans_words:
            f_predict.write(u"{}".format(i))
        # print (predict_word)
        

def save_class_data(model,all_feature_vector,file,dictionary_size):
    f = open(file, "w")
    symbol_to_word_ref = {}
    dict_symspell = {}
    count = 0
    for i in range(len(all_feature_vector)):
        predicted_label = model.predict(all_feature_vector[i])
        s = ''
        for j in predicted_label:
            f.write(str(j) + " ")
            s += chr(j+33)
        if s in dict_symspell:
            dict_symspell[s] += 1
        else :
            dict_symspell[s] = 1
            symbol_to_word_ref[s] = int(count/(len(all_feature_vector)/dictionary_size))+1
            # print ([s,int(count/(len(all_feature_vector)/dictionary_size))+1])
        f.write("[" + str(len(predicted_label)) + "]\n")
        count += 1
            
    f.close()
    return dict_symspell,symbol_to_word_ref

def cluster_histo(number_clusters,all_feature_vector,model,flag):
    histo = np.zeros(number_clusters)
    if flag:
        s = 'train'
    else:
        s = 'test'
    for i in range(len(all_feature_vector)):
        predicted_label = model.predict(all_feature_vector[i])
        for j in predicted_label:
            histo[j] += 1
    plt.bar(np.arange(number_clusters),histo, color="green")
    # plt.show()
    plt.savefig('histo_all_clus_'+ s +'_'+ str(number_clusters) +'.png')
    plt.close()

def cluster_histo_each_word(number_clusters,other_words,model,flag):
    c = 0
    if flag:
        s = 'train'
    else:
        s = 'test'
    for i in other_words:
        c += 1
        histo = np.zeros(number_clusters)
        for j in range(len(i)):
            predicted_label = model.predict(i[j])
            for k in predicted_label:
                histo[k] += 1
        plt.bar(np.arange(number_clusters),histo, color="green")
        plt.savefig('histo_each_word_'+ s +'_'+ str(c) +'_'+ str(number_clusters) +'.png')
        plt.close()

def save_dictionary_symspell(file,dict_symspell):
    f = open(file,"w")
    for i in dict_symspell:
        f.write(i + " 10000\n")
    f.close()

def editDistance(str1, str2, m , n): 
    dp = [[0 for x in range(n+1)] for x in range(m+1)] 
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
    return dp[m][n] 

def cal_edit_dist(other_words,model,number_clusters):
    f = open('edit_dist_train_' + str(number_clusters) +'.txt','w')
    w = 1
    for i in other_words:
        edit_dist_avg = 0
        count = 0
        for j in range(len(i)):
            predict_1 = model.predict(i[j])
            s1 = ''
            for p in predict_1:
                s1 += chr(p+33)
            for k in range(j+1,len(i)):
                predict_2 = model.predict(i[k])
                s2 = ''
                for p in predict_2:
                    s2 += chr(p+33)
                # print (s1,s2)
                edit_dist_avg +=  editDistance(s1,s2,len(s1),len(s2))
                # print (edit_dist_avg)
                count += 1
        edit_dist_avg = edit_dist_avg/count
        f.write(str(w) + ' : ' + str(edit_dist_avg) + '\n')
        w += 1

main()