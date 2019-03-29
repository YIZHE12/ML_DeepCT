import numpy as np
from keras.layers import Dropout
#import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Reshape, Dense, Activation, Flatten
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import RMSprop
from keras import regularizers

import matplotlib.pyplot as plt
from skimage.transform import iradon
#import tensorflow as tf
#from tensorflow.python.framework import ops
#import math
import time
#import scipy.io as sio
from generate_input import load_images_normalize_shift_rotate_large

# from generate_input import load_images_from_folder
## it has options whether to train model from the beginning or pick up a model that has been trained
## it has L1 regulisation in the Conv2, and use RMS optimizer
# from automap import *
#%matplotlib inline

n_im = 60  # How many images to load
no_angles = 2 # number of projection angle
img_size = 64 # size of the images, need to be even number
tic1 = time.time()
print ('Reading data ... ')
#X_train, Y_train = load_images_0to1('Data\ImageNet', n_im, img_size, no_angles, imrotate=False)
X_train, Y_train  = load_images_normalize_shift_rotate_large('Data\ImageNet', n_im, img_size, no_angles, shift = True, imrotate = True)

#X_test, Y_test = load_images_0to1('Data\Test', 200, img_size, no_angles, imrotate=False)
X_test, Y_test  = load_images_normalize_shift_rotate_large('Data\Test', 200, img_size, no_angles, shift = False, imrotate = False)
#Y_train=load_images_y('Data\ImageNet', n_im, img_size, -0.5, 0.5, shift = False, imrotate = False)
#X_train=create_sino_X (Y_train, no_angles, -0.5, 0.5)

#X_test=load_images_y('Data\Test', n_im, img_size, -0.5, 0.5, shift = False, imrotate = False)
#Y_test=create_sino_X (X_test, no_angles, -0.5, 0.5)
    
toc1 = time.time()
print('Time to load data = ', (toc1 - tic1))
print('X_train.shape at input = ', X_train.shape)
print('Y_train.shape at input = ', Y_train.shape)


def init_model(input_shape):
    start_time = time.time()
    print ('Compiling Model ... ')
    model = Sequential()

    
    model.add(Flatten(input_shape=(1,img_size,no_angles)))# FC 1 , output shape (_,img_size*no_angles)
    model.add(Dense(img_size*no_angles))# FC 2 , output shape (_,img_size*no_angles*2)    
    model.add(Activation('tanh'))
    
    model.add(Dense(int(img_size*img_size)))# FC 3 , output shape (_,img_size*img_size*4)      
    model.add(Activation('tanh'))
    model.add(Dropout(0.5, seed = 1))

    
    model.add(Reshape((1,int(img_size),int(img_size)) ) )    

    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'elu', data_format="channels_first", kernel_regularizer=regularizers.l1(0.0001))) # C1
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'elu', data_format="channels_first" )) # C2
    #model.add(ZeroPadding2D(padding=(1,1)))      
    model.add(Conv2DTranspose(filters=1, kernel_size=(3,3), strides=(1,1), padding='same', activation='elu', data_format="channels_first" ) ) # C3
   
    rms = RMSprop(lr=0.00002, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rms, metrics=['mean_squared_error'])
    print ('Model complied in {0} seconds'.format(time.time() - start_time))
    #filepath="weights.best.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    return model

init_model=init_model(X_train.shape)

# checkpoint

#init_model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])
init_model.fit(X_train, Y_train, epochs = 1, batch_size = 50)
### START CODE HERE ### (1 line)

init_model.summary()
init_model.save('test.h5')  # creates a HDF5 file 'my_model.h5'



preds = init_model.evaluate(x = X_test, y = Y_test)

### END CODE HERE ##
print()
print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))

imgID=30
theta = np.linspace(0., 180., no_angles, endpoint=True)

img_result=init_model.predict(X_test)
img_result_test = np.squeeze(img_result[imgID,0,:,:])
ori_test = np.squeeze(Y_test[imgID,0,:,:])
sino_test= np.squeeze(X_test[imgID,0,:,:])
reconstruction_fbp = iradon(sino_test, theta=theta, circle=True)

img_result_tr=init_model.predict(X_train)
img_result_train = np.squeeze(img_result_tr[imgID,0,:,:])
ori_train = np.squeeze(Y_train[imgID,0,:,:])
sino_train= np.squeeze(X_train[imgID,0,:,:])
reconstruction_fbp_train = iradon(sino_train, theta=theta, circle=True)




plt.subplot(231), plt.imshow(ori_test, cmap='gray')
plt.title('Original test set'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(img_result_test, cmap='gray')
plt.title('Recon'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(reconstruction_fbp, cmap='gray')
plt.title('Recon by iradon'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(ori_train, cmap='gray')
plt.title('Original train set'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(img_result_train, cmap='gray')
plt.title('Recon'), plt.xticks([]), plt.yticks([])

plt.subplot(236), plt.imshow(reconstruction_fbp_train, cmap='gray')
plt.title('Recon by iradon'), plt.xticks([]), plt.yticks([])

plt.show()


