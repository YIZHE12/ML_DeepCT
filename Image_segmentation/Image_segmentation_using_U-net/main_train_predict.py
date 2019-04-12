from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from data import *
from model import *
from image_function import *
import matplotlib.pyplot as plt

#################################################################################################################
def focal_loss(gamma=2, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        eps = 1e-12
        y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss 
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


#################################################################################################################
data_dir = 'Data\\sample3\\' # image folder

data_folder = data_dir + 'train\\' #put your folder where you stored the annotated data
save_folder = data_dir + 'train\\' + 'aug' #put your folder where you want stored the augmentation data

data_gen_args = dict(rotation_range=1,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGenerator = trainGenerator(10,data_folder,'image','mask',data_gen_args,save_to_dir = save_folder)

test_folder = data_dir + 'test\\' #put your folder where you stored the annotated data
test_save_folder = data_dir + 'test\\' + 'aug' #put your folder where you want stored the augmentation data

testGenerator = trainGenerator(10,test_folder,'image','mask',data_gen_args,save_to_dir = test_save_folder)

#################################################################################################################
model = unet()
focal_loss_fixed = focal_loss()
model.compile(optimizer = Adam(lr = 1e-4), loss = focal_loss_fixed, metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model_unet.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model.fit_generator(myGenerator,steps_per_epoch=500,epochs=50, \
	validation_data = testGenerator, validation_steps=50, callbacks=[es, mc])

#################################################################################################################
data, fs = load_images_from_folder(data_dir + 'Validate',  resize = True) # load validation data
data = np.expand_dims (data, axis = 3) # keras used channel last by default
np.shape(data)

results = model.predict(data)
plt.imshow(results[230,:,:,0], cmap = 'gray')
plt.show()

plt.figure(figsize = (20,10))
plt.subplot(121)
plt.imshow(results[230,:,:,0]>0.001, cmap = 'gray')
plt.subplot(122)
plt.imshow(data[230,:,:,0], cmap = 'gray')
plt.show()