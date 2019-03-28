'''
test whether Keras is functioning well in this computer
'''

print ('loading modules...')

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, LeakyReLU
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import RMSprop
from keras import regularizers

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from skimage.transform import iradon

import time
from keras_generate_input import load_images_normalize_shift_rotate_large
from keras.callbacks import ModelCheckpoint, Callback
import scipy.io as sio



n_im = 60  # How many images to load
no_angles = 16 # number of projection angle
img_size = 200 # size of the images, need to be even number
tic1 = time.time()
print ('Reading data ... ')

X_train, Y_train  = load_images_normalize_shift_rotate_large('Data\ImageNet', n_im, img_size, no_angles, shift = True, imrotate = True)
X_test, Y_test  = load_images_normalize_shift_rotate_large('Data\Test', 200, img_size, no_angles, shift = False, imrotate = False)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
toc1 = time.time()
print('Time to load data = ', (toc1 - tic1))
print('X_train.shape at input = ', X_train.shape)
print('Y_train.shape at input = ', Y_train.shape)



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

data_path = r'C:\Users\yizhe\Desktop\Generate images\results\Automap\temp'
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1,period=1)

history = LossHistory()
model.fit(X_train, Y_train, epochs = 2, batch_size = 4,callbacks=[checkpointer, history])

model.summary()
model.save('modelwhole.h5')  # creates a HDF5 file 'my_model.h5'

print (history.losses)
sio.savemat(data_path + 'history.mat', {'history':history.losses})


preds = model.evaluate(x = X_test, y = Y_test)

### END CODE HERE ##
print()
print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))

imgID=30
theta = np.linspace(0., 180., no_angles, endpoint=True)

img_result=model.predict(X_test)
img_result_test = np.squeeze(img_result[imgID,0,:,:])
ori_test = np.squeeze(Y_test[imgID,0,:,:])
sino_test= np.squeeze(X_test[imgID,0,:,:])
reconstruction_fbp = iradon(sino_test, theta=theta, circle=True)

img_result_tr=model.predict(X_train)
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


