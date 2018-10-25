import cv2
import os
import numpy as np
from sklearn.externals import joblib

def main():
    model = load_model()


def predict_class(model,all_feature_vactor):
    f = open("predicted_label.txt", "w")
    for i in range(len(all_feature_vactor)):
        # for j in range(len(all_feature_vactor[i])):
            predicted_label = model.predict(all_feature_vactor[i])
            print (predicted_label)
        # print ()

def load_model():
    filename = 'KMeans_model.sav'
    loaded_model = joblib.load(filename)
    return loaded_model