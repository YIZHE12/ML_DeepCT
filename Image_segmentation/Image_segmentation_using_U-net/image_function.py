import numpy as np
import cv2
import os
import imageio
from math import sqrt
import re
import skimage.transform as trans

def ensure_dir(file_path): # create a folder if the folder is not already exist
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_images_from_folder(folder, mode = 'grey', resize = False, target_size = 256): # load all image from a folder with its natural order
    images = []
    fs = []
    for filename in os.listdir(folder):
        f = os.path.join(folder,filename)
       	if mode == 'color':
            img =cv2.imread(f, cv2.COLOR_GRAY2RGB)
        elif mode == 'grey':
            img =cv2.imread(f, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            if resize:
                img = trans.resize(img,[target_size, target_size])
            images.append(img)
            fs.append((f[-7:-4])) # save its file name in the order that it was read
         
    images = np.asarray(images)
    return images, fs

def make_border(data, target_size, path): # make a border to shape image to ideal shape before cropping
    # calculate the padding size 
    top = int((target_size-np.shape(data)[0])/2) 
    bottom = top
    left = int((target_size-np.shape(data)[1])/2)
    right = left
    
    for i in range(0, np.shape(data)[2]):
        image = cv2.copyMakeBorder(data[:,:,i], top, bottom, left, right, borderType = 0)
        imageio.imwrite(path+"\\"+str(i)+'.jpg', image)   
        
def create_filename(N_slices, N_image_start, N_image_stop): 
    # input: 
    # N_slices -  number of slice per images
    # N_image_start - index of the first image to read
    # N_image_stop - index of the last image to read
    
    N_slices = int(sqrt(N_slices))
    filename = []
    for i in range(N_image_start, N_image_stop+1):
        for j in range(0, N_slices):
            for k in range(0, N_slices):                  
                f = str(i)+'_'+str(j)+'_'+str(k)+'.jpg'
                                   
                filename.append(f)
    #print(filename)
    return (filename)

def read_crop_write(read_path, write_path, N_pieces): 
    # read images from folders and cropped them to N_pieces
    #images = []
    for filename in os.listdir(read_path):
        f = os.path.join(read_path,filename)
        img =cv2.imread(f, cv2.IMREAD_GRAYSCALE)
 
        I_crop_v  =  np.asarray(np.split(img, sqrt(N_pieces), axis = 0)) 
        # when split in this axis can be recovered by using I_crop_recover = I_crop_v.reshape(1024,1024)
        I_crop_vv = np.asarray(np.split(I_crop_v, sqrt(N_pieces), axis = 2))
        
        for i in range(0,int(sqrt(N_pieces))):
            for j in range (0,int(sqrt(N_pieces))):
                prefix= re.sub("\D", "", filename)
                filename_new = write_path + '\\'+ prefix + '_'+ str(i)+'_'+str(j) +'.jpg'
                cv2.imwrite(filename_new, I_crop_vv[j, i, :, :])
    return (len(os.listdir(read_path)))
                
def stitch_image (images, slices):
    block_row = int(sqrt(slices))
    for j in range(0,block_row):
        output = images[j*block_row,:,:]
        for i in range(1,block_row):
            output = np.block([output, images[j*block_row+i,:,:]])
        if j == 0:
            output_whole=output
        else:
            output_whole = np.block([[output_whole],[output]])
    return (output_whole)