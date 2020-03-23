
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = r'C:\PetImages'
CATEGORIES = ['Dog', 'Cat']

training_data = list()
IMG_SIZE = 50 #pixels

def create_training_data():
    #generate labels and features from data set
    for c in CATEGORIES:
        path = os.path.join(DATADIR, c)
        class_num = CATEGORIES.index(c)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)

create_training_data()

X =list()
y=list()

for feature, label in training_data:
    X.append(feature)
    y.append(label)

# features as a vector
X=np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# save data
import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()