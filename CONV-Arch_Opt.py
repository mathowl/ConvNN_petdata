import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Flatten
import pickle
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import time
import os


X=pickle.load(open('X.pickle','rb')) #features
y=pickle.load(open('y.pickle','rb')) #labels

X= tf.keras.utils.normalize(X,axis=1) #normalize
y=np.array(y) # keras requires array

# layer vairety
dense_layers = [2,3]   
layer_sizes = [64,128]
conv_layers = [2,3]


# neural net
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer, int(time.time()))
            MODEL_NAME=r'C:\CONV-ArchVariation\bestmodel\{}.model'.format(NAME) #model save location
            print(NAME)

            PROB = r"Cats-v-Dogs-CNN\{}".format(NAME)
            LOGDIR = r'C:\ML-logs\{}'.format(PROB) #logs for tensorboard, to open tensorboard go to cmd: tensorboard --logdir=LOGDIR 
            print(LOGDIR)
            
            tensorboard = TensorBoard(log_dir = LOGDIR)

            model = Sequential()
            model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size = (2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,(3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size = (2,2)))

            model.add(Flatten())
            for l in range(dense_layer-1):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
            
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics =['accuracy'])

            checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # only save epoch that min val_loss            
            es=EarlyStopping(monitor='val_loss', mode='min', baseline=0.4, verbose= 1, patience=8)
            callbacks_list=[tensorboard,checkpoint,es]

            model.fit(X,y, batch_size=32, validation_split = 0.1,epochs=16,callbacks=callbacks_list)  