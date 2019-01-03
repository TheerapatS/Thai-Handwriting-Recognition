from sklearn.cluster import KMeans
from sklearn.externals import joblib
import codecs
import numpy as np

def k_means(number_clusters,data):
    model = KMeans(n_clusters=number_clusters)
    model.fit(np.array(data))
    return model

def save_model(model,number_clusters):
    filename = 'KMeans_model_' + str(number_clusters) + '.sav'
    joblib.dump(model, filename)

def load_model(number_clusters):
    filename = 'KMeans_model_' + str(number_clusters) + '.sav'
    loaded_model = joblib.load(filename)
    return loaded_model

def k_means_prediction(model,data):
    predicted_label = model.predict(data)
    return predicted_label
