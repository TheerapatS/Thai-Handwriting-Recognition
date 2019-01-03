import os
from Symbol_sequence import *
from k_means_cluster import *

def main():
    number_clusters = 30
    model = train_data(number_clusters)
    test_data(model)

def train_data(number_clusters):
    path_train_data = "D:\Work\Project\Dictionary_word\\"
    sliding_windows_train = "D:\Work\Project\\training_set\Dictionary_sliding\\"
    print ("Do you want to train new model or load : train(t)\load(l) :: " , end='')
    flag_operator_load_save = input()

    ##### save use this #####
    if flag_operator_load_save == 't':
        print("Train new data !!")
        make_dir(sliding_windows_train,500)
        train_sliding_windows_data = []
        for folder in os.listdir(path_train_data):
            sub_path_train_data = path_train_data + str(folder) + "\\"
            for file in os.listdir(sub_path_train_data):
                print ("Make Sliding window " + str(folder) + "\\{0:20}".format(file))
                feature_vector = k_means_symbol(sub_path_train_data + file)
                for i in feature_vector:
                    train_sliding_windows_data.append(i)
        print ("##########-- Train Model --##########")
        model = k_means(number_clusters,train_sliding_windows_data)
        save_model(model,number_clusters)
        return model
    ##### load use this #####
    elif flag_operator_load_save == 'l' :
        print("Load model !!")
        model = load_model(number_clusters)
        return model

def test_data(model):
    path_test_data = "D:\Work\Project\\training_set\Symbol_Test\\"
    sliding_windows_test = "D:\Work\Project\\training_set\Symbol_Test_out\\"
    
    make_dir(sliding_windows_test,132)
    test_image_data = []
    for file in os.listdir(path_test_data):
        print ("Make Sliding window {0:20}".format(file))
        feature_vector = k_means_symbol(path_test_data + file)
        prediction_label = k_means_prediction(model,feature_vector)
        test_image_data.append(feature_vector)
        print (prediction_label)
        
main()