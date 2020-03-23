import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

MODELDIR = r'filepath' #filepath to optimal model
model = tf.keras.models.load_model(MODELDIR)   #load model
CATEGORIES = ['Dog', 'Cat']

def prepare(filepath):
    # convert image to feature
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(50,50))
    return new_array.reshape(-1,50,50,1)

def model_pred(filepath):
    # gives prediction for image in filepath using neural net in MODELDIR
    prediction = model.predict([prepare(filepath)])
    return CATEGORIES[int(prediction[0][0])]
