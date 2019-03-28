'''
create input data with flexible size
'''
from __future__ import print_function, division
import numpy as np
import cv2
import os
import matplotlib
from matplotlib import pyplot as plt
import random

from skimage import data_dir
from skimage.io import imread
from skimage.transform import radon, iradon, rescale, resize
from scipy.ndimage.interpolation import shift


def im_rotate(img, angle):
    """ Rotates an image by angle degrees
    :param img: input image
    :param angle: angle by which the image is rotated, in degrees
    :return: rotated image
    """
    rows, cols = img.shape
    rotM = cv2.getRotationMatrix2D((cols/2-0.5, rows/2-0.5), angle, 1)
    imrotated = cv2.warpAffine(img, rotM, (cols, rows))

    return imrotated

def createCircularMask(h, w, center=None, radius=None):
    #draw a circular mask
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def crop_circle(image, crop_half_width, crop_half_height, mask):
    #crop the center of an image and draw a circular mask '   
    h=image.shape[1]
    w=image.shape[0]
    center = [int(w/2), int(h/2)]
    masked_img = image [int(center[0]-crop_half_width):int(center[0]+crop_half_width),int(center[1]-crop_half_height):int(center[1]+crop_half_height)].copy()

    masked_img[~mask] = 0

    return (masked_img)

def create_sino(image,no_angles):
    theta = np.linspace(0., 180., no_angles, endpoint=True)
    sinogram = radon(image, theta=theta, circle=True)
    #print ('sinogram shape:',np.shape(sinogram))
    return (sinogram, theta)

def create_sino_fft(image,no_angles):
    theta = np.linspace(0., 180., no_angles, endpoint=True)
    sinogram_ori = radon(image, theta=theta, circle=True)
    sinogram_fft = np.fft.fft2(sinogram_ori)
    sinogram_fft = np.fft.fftshift(sinogram_fft)
    sinogram = np.concatenate((sinogram_fft.real, sinogram_fft.imag), axis = 1)
    #print ('sinogram shape:',np.shape(sinogram))
    return (sinogram, theta)

def create_sino_shift(image,no_angles):
    theta = np.linspace(0., 180., no_angles, endpoint=True)
    sinogram = radon(image, theta=theta, circle=True)
    
    for t in range (0, no_angles, 2):
        #random.seed(30) #if no need then the shift will be different each dataset
        step = random.randint(1, 10)
        sinogram[:, t] = np.roll(sinogram [:, t], step)
    #print ('sinogram shape:',np.shape(sinogram))
    return (sinogram, theta)



def load_images(folder, n_im, img_size, no_angles, imrotate=False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """
    assert(img_size % 2 == 0) # image size must be even number 
    # Initialize the arrays:
    if imrotate:  # number of images is 4 * n_im
        bigy = np.empty((n_im * 4, 1, img_size, img_size))
        bigx = np.empty((n_im * 4, 1, img_size, no_angles))
    else:
        bigy = np.empty((n_im, 1, img_size, img_size))
        bigx = np.empty((n_im, 1, img_size, no_angles))
        
    # create circular mask    
    mask = createCircularMask(img_size, img_size, radius=img_size/2)
    print ('mask shape:', np.shape(mask))
    im = 0  # image counter
    #print (os.listdir(folder))
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
            #print ('current image shape:', np.shape(bigy_temp))    
            # normalized to 0 and 1
            bigy_temp=(cv2.normalize(bigy_temp, bigy_temp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)-0.5)*2
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1) # if the image is too small, upscale
                
            #print('max X:',np.amax(bigy_temp))
            # crop to a circle with a mask, also make it to 64 x 64
            #print ('current image shape:', np.shape(bigy_temp))
            
            bigy_temp=crop_circle(bigy_temp, img_size/2, img_size/2, mask)
            
            bigy[im, :, :, :] = np.expand_dims(bigy_temp, axis = 0);
            sinogram, _ = create_sino(bigy_temp, no_angles)
            #print ('sinogram shape:', np.shape(sinogram))
            
            bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
            im += 1
            if imrotate:
                for angle in [90, 180, 270]:
                    bigy_rot = im_rotate(bigy_temp, angle)
                   
                    sinogram, _ = create_sino(bigy_rot,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_rot, axis = 0);
                    
                    im += 1

        if imrotate:
            if im > (n_im * 4 - 1):  # how many images to load
                break
        else:
            if im > (n_im - 1):  # how many images to load
                break

    

    return bigx, bigy

def load_images_0to1(folder, n_im, img_size, no_angles, imrotate=False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """
    assert(img_size % 2 == 0) # image size must be even number 
    # Initialize the arrays:
    if imrotate:  # number of images is 4 * n_im
        bigy = np.empty((n_im * 4, 1, img_size, img_size))
        bigx = np.empty((n_im * 4, 1, img_size, no_angles))
    else:
        bigy = np.empty((n_im, 1, img_size, img_size))
        bigx = np.empty((n_im, 1, img_size, no_angles))
        
    # create circular mask    
    mask = createCircularMask(img_size, img_size, radius=img_size/2)
    print ('mask shape:', np.shape(mask))
    im = 0  # image counter
    #print (os.listdir(folder))
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
            #print ('current image shape:', np.shape(bigy_temp))    
            # normalized to 0 and 1
            bigy_temp=cv2.normalize(bigy_temp, bigy_temp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)-0.5
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1) # if the image is too small, upscale
                
            #print('max X:',np.amax(bigy_temp))
            # crop to a circle with a mask, also make it to 64 x 64
            #print ('current image shape:', np.shape(bigy_temp))
            
            bigy_temp=crop_circle(bigy_temp, img_size/2, img_size/2, mask)
            
            bigy[im, :, :, :] = np.expand_dims(bigy_temp, axis = 0);
            sinogram, _ = create_sino(bigy_temp, no_angles)
            #print ('sinogram shape:', np.shape(sinogram))
            
            bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
            im += 1
            if imrotate:
                for angle in [90, 180, 270]:
                    bigy_rot = im_rotate(bigy_temp, angle)
                   
                    sinogram, _ = create_sino(bigy_rot,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_rot, axis = 0);
                    
                    im += 1

        if imrotate:
            if im > (n_im * 4 - 1):  # how many images to load
                break
        else:
            if im > (n_im - 1):  # how many images to load
                break

    
    bigx=cv2.normalize(bigx, bigx, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)-0.5
    return bigx, bigy

def load_images_normalize_shift_rotate(folder, n_img, img_size, no_angles, shift = False, imrotate = False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """
    assert(img_size % 2 == 0) # image size must be even number 
    # Initialize the arrays:
    n_im = n_img
    if shift:
        n_im = 4*n_img
    if imrotate:
        n_im = 4*n_img        
    if shift and imrotate:  
        n_im = 7*n_img

        
    print ('number of image:', n_im)
    bigy = np.empty((n_im, 1, img_size, img_size))
    bigx = np.empty((n_im, 1, img_size, no_angles))
             
    # create circular mask    
    mask = createCircularMask(img_size, img_size, radius=img_size/2)
    print ('mask shape:', np.shape(mask))
    im = 0  # image counter
    #print (os.listdir(folder))
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
   
            # normalized to -0.5 to 0.5
            bigy_temp=cv2.normalize(bigy_temp, bigy_temp, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            bigy_temp = rescale(bigy_temp,0.2,order=1) 
            # if the image is too small, upscale
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,5,order=1) 
                
            # crop a circle in the middle of the image to simulate CT        
            bigy_temp_crop=crop_circle(bigy_temp, img_size/2, img_size/2, mask)       
            
            # expand dimension to make it channel first 
            bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);

            # generate sinogram
            sinogram, _ = create_sino(bigy_temp_crop, no_angles)
            bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
            im += 1            
            
            
            if shift:
                i = 0
                for step in [-10, -10, 10]:
                    
                    bigy_shift = np.roll(bigy_temp, step, axis = i)
                    bigy_temp_crop=crop_circle(bigy_shift, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino(bigy_temp_crop,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    i = 1 # change axis
                    im += 1
                    
                    
            if imrotate:
                for angle in [90, 180, 270]:
                    
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigy_temp_crop=crop_circle(bigy_rot, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino(bigy_temp_crop,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    
                    im += 1
                    

        if im > (n_im - 1):  # how many images to load
            break
    
    #bigy=cv2.normalize(bigy, bigy, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bigx=cv2.normalize(bigx, bigx, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return bigx, bigy

def load_images_normalize_shift_rotate_random(folder, n_img, img_size, no_angles, shift = False, imrotate = False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """
    assert(img_size % 2 == 0) # image size must be even number 
    # Initialize the arrays:
    n_im = n_img
    if shift:
        n_im = 4*n_img
    if imrotate:
        n_im = 4*n_img        
    if shift and imrotate:  
        n_im = 7*n_img

        
    print ('number of image:', n_im)
    bigy = np.empty((n_im, 1, img_size, img_size))
    bigx = np.empty((n_im, 1, img_size, no_angles))
             
    # create circular mask    
    mask = createCircularMask(img_size, img_size, radius=img_size/2)
    print ('mask shape:', np.shape(mask))
    im = 0  # image counter
    #print (os.listdir(folder))
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
   
            # normalized to -0.5 to 0.5
            # normalized to -0.5 to 0.5
            bigy_temp=cv2.normalize(bigy_temp, bigy_temp, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            bigy_temp = rescale(bigy_temp,0.2,order=1) 
            # if the image is too small, upscale
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1) 
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)   
            # if the image is too small, upscale
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,10,order=1) 
            #print('image shape:',np.shape(bigy_temp))    
            # crop a circle in the middle of the image to simulate CT        
            bigy_temp_crop=crop_circle(bigy_temp, img_size/2, img_size/2, mask)       
            
            # expand dimension to make it channel first 
            bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);

            # generate sinogram
            sinogram, _ = create_sino_shift(bigy_temp_crop, no_angles)
            bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
            im += 1            
            
            
            if shift:
                i = 0
                for step in [-10, -10, 10]:
                    
                    bigy_shift = np.roll(bigy_temp, step, axis = i)
                    bigy_temp_crop=crop_circle(bigy_shift, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino_shift(bigy_temp_crop,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    i = 1 # change axis
                    im += 1
                    
                    
            if imrotate:
                for angle in [90, 180, 270]:
                    
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigy_temp_crop=crop_circle(bigy_rot, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino_shift(bigy_temp_crop,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    
                    im += 1
                    

        if im > (n_im - 1):  # how many images to load
            break
    
    #bigy=cv2.normalize(bigy, bigy, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bigx=cv2.normalize(bigx, bigx, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return bigx, bigy
def load_images_normalize_shift_rotate_random_fs(folder, n_img, img_size, no_angles, shift = False, imrotate = False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """
    assert(img_size % 2 == 0) # image size must be even number 
    # Initialize the arrays:
    n_im = n_img
    if shift:
        n_im = 4*n_img
    if imrotate:
        n_im = 4*n_img        
    if shift and imrotate:  
        n_im = 7*n_img

        
    print ('number of image:', n_im)
    bigy = np.empty((n_im, 1, img_size, img_size))
    bigx = np.empty((n_im, 1, img_size, no_angles))
             
    # create circular mask    
    mask = createCircularMask(img_size, img_size, radius=img_size/2)
    print ('mask shape:', np.shape(mask))
    im = 0  # image counter
    #print (os.listdir(folder))
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
   
            # normalized to -0.5 to 0.5
            # normalized to -0.5 to 0.5
            bigy_temp=cv2.normalize(bigy_temp, bigy_temp, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            #print('image shape:',np.shape(bigy_temp))    
            # crop a circle in the middle of the image to simulate CT        
            bigy_temp_crop=crop_circle(bigy_temp, img_size/2, img_size/2, mask)       
            
            # expand dimension to make it channel first 
            bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);

            # generate sinogram
            sinogram, _ = create_sino_shift(bigy_temp_crop, no_angles)
            bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
            im += 1            
            
            
            if shift:
                i = 0
                for step in [-10, -10, 10]:
                    
                    bigy_shift = np.roll(bigy_temp, step, axis = i)
                    bigy_temp_crop=crop_circle(bigy_shift, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino_shift(bigy_temp_crop,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    i = 1 # change axis
                    im += 1
                    
                    
            if imrotate:
                for angle in [90, 180, 270]:
                    
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigy_temp_crop=crop_circle(bigy_rot, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino_shift(bigy_temp_crop,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    
                    im += 1
                    

        if im > (n_im - 1):  # how many images to load
            break
    
    #bigy=cv2.normalize(bigy, bigy, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bigx=cv2.normalize(bigx, bigx, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return bigx, bigy

def load_images_y(folder, n_img, img_size, min_output, max_output, shift = False, imrotate = False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """
    assert(img_size % 2 == 0) # image size must be even number 
    # Initialize the arrays:
    n_im = n_img
    if shift:
        n_im = 4*n_img
    if imrotate:
        n_im = 4*n_img        
    if shift and imrotate:  
        n_im = 7*n_img

        
    print ('number of image:', n_im)
    bigy = np.empty((n_im, 1, img_size, img_size))
    #bigx = np.empty((n_im, 1, img_size, no_angles))
             
    # create circular mask    
    mask = createCircularMask(img_size, img_size, radius=img_size/2)
    print ('mask shape:', np.shape(mask))
    im = 0  # image counter
    #print (os.listdir(folder))
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
   
            # normalized to -0.5 to 0.5
            bigy_temp = rescale(bigy_temp,0.2,order=1)
            bigy_temp=cv2.normalize(bigy_temp, bigy_temp, alpha=min_output, beta=max_output, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            # if the image is too small, upscale
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,5,order=1) 
                
            # crop a circle in the middle of the image to simulate CT        
            bigy_temp_crop=crop_circle(bigy_temp, img_size/2, img_size/2, mask)       
            
            # expand dimension to make it channel first 
            bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0)

            # generate sinogram
            #sinogram, _ = create_sino(bigy_temp_crop, no_angles)
            #bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0)
            im += 1            
            
            
            if shift:
                i = 0
                for step in [-10, -10, 10]:
                    
                    bigy_shift = np.roll(bigy_temp, step, axis = i)
                    bigy_temp_crop=crop_circle(bigy_shift, img_size/2, img_size/2, mask) 
                    #sinogram, _ = create_sino(bigy_temp_crop,no_angles)
                    #bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0)
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0)
                    i = 1 # change axis
                    im += 1
                    
                    
            if imrotate:
                for angle in [90, 180, 270]:
                    
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigy_temp_crop=crop_circle(bigy_rot, img_size/2, img_size/2, mask) 
                    #sinogram, _ = create_sino(bigy_temp_crop,no_angles)
                    #bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0)
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0)
                    
                    im += 1
                    

        if im > (n_im - 1):  # how many images to load
            break
    
    #bigy=cv2.normalize(bigy, bigy, alpha=min_output, beta=max_output, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #bigx=cv2.normalize(bigx, bigx, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return bigy

def create_sino_X (input_img_Y, no_angles, min_output, max_output):
    n_im = input_img_Y.shape[0]
    img_size = input_img_Y.shape[2]
    bigx = np.empty((n_im, 1, img_size, no_angles))
    for im in range(0, n_im-1):
        img_temp = np.squeeze(input_img_Y[im, :, :, :])
        sinogram, _ = create_sino(img_temp,no_angles)
        
        bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0)
        
    bigx=cv2.normalize(bigx, bigx, alpha=-min_output, beta=max_output, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return bigx




def binary_mask(img_size, no_angles, image):
    np.random.seed(seed=10)
    bi_mask = np.random.randint(9, size=(img_size, no_angles))
    bi_mask = np.clip(bi_mask, 0, 1)
    image = np.multiply(image,bi_mask/10)
    return (image)

def load_images_normalize_shift_rotate_mask(folder, n_img, img_size, no_angles, shift = False, imrotate = False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """
    assert(img_size % 2 == 0) # image size must be even number 
    # Initialize the arrays:
    n_im = n_img
    if shift:
        n_im = 4*n_img
    if imrotate:
        n_im = 4*n_img        
    if shift and imrotate:  
        n_im = 7*n_img

        
    print ('number of image:', n_im)
    bigy = np.empty((n_im, 1, img_size, img_size))
    bigx = np.empty((n_im, 1, img_size, no_angles))
             
    # create circular mask    
    mask = createCircularMask(img_size, img_size, radius=img_size/2)
    print ('mask shape:', np.shape(mask))
    im = 0  # image counter
    #print (os.listdir(folder))
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
   
            # normalized to -0.5 to 0.5
            bigy_temp=cv2.normalize(bigy_temp, bigy_temp, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            bigy_temp = rescale(bigy_temp,0.2,order=1) 
            # if the image is too small, upscale
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)            
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
                
            # crop a circle in the middle of the image to simulate CT        
            bigy_temp_crop=crop_circle(bigy_temp, img_size/2, img_size/2, mask)       
            
            # expand dimension to make it channel first 
            bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);

            # generate sinogram
            sinogram, _ = create_sino(bigy_temp_crop, no_angles)
            sinogram = binary_mask(img_size, no_angles, sinogram)
            bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
            im += 1            
            
            
            if shift:
                i = 0
                for step in [-10, -10, 10]:
                    
                    bigy_shift = np.roll(bigy_temp, step, axis = i)
                    bigy_temp_crop=crop_circle(bigy_shift, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino(bigy_temp_crop,no_angles)
                    sinogram = binary_mask(img_size, no_angles, sinogram)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    i = 1 # change axis
                    im += 1
                    
                    
            if imrotate:
                for angle in [90, 180, 270]:
                    
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigy_temp_crop=crop_circle(bigy_rot, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino(bigy_temp_crop,no_angles)
                    sinogram = binary_mask(img_size, no_angles, sinogram)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    
                    im += 1
                    

        if im > (n_im - 1):  # how many images to load
            break
    
    #bigy=cv2.normalize(bigy, bigy, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bigx=cv2.normalize(bigx, bigx, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return bigx, bigy

def load_images_normalize_shift_rotate_large(folder, n_img, img_size, no_angles, shift = False, imrotate = False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """
    assert(img_size % 2 == 0) # image size must be even number 
    # Initialize the arrays:
    n_im = n_img
    if shift:
        n_im = 4*n_img
    if imrotate:
        n_im = 4*n_img        
    if shift and imrotate:  
        n_im = 7*n_img

        
    print ('number of image:', n_im)
    bigy = np.empty((n_im, 1, img_size, img_size))
    bigx = np.empty((n_im, 1, img_size, no_angles))
             
    # create circular mask    
    mask = createCircularMask(img_size, img_size, radius=img_size/2)
    print ('mask shape:', np.shape(mask))
    im = 0  # image counter
    #print (os.listdir(folder))
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
   
            # normalized to -0.5 to 0.5
            bigy_temp=cv2.normalize(bigy_temp, bigy_temp, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            bigy_temp = rescale(bigy_temp,0.2,order=1) 
            # if the image is too small, upscale
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1) 
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)
            if (np.shape(bigy_temp)[0]  <=img_size) or (np.shape(bigy_temp)[1] <=img_size):
                bigy_temp = np.squeeze(bigy_temp)
                bigy_temp = rescale(bigy_temp,2,order=1)                 
            # crop a circle in the middle of the image to simulate CT        
            bigy_temp_crop=crop_circle(bigy_temp, img_size/2, img_size/2, mask)       
            
            # expand dimension to make it channel first 
            bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);

            # generate sinogram
            sinogram, _ = create_sino(bigy_temp_crop, no_angles)
            bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
            im += 1            
            
            
            if shift:
                i = 0
                for step in [-10, -10, 10]:

                    bigy_shift = np.roll(bigy_temp, step, axis = i)
                    bigy_temp_crop=crop_circle(bigy_shift, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino(bigy_temp_crop,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    i = 1 # change axis
                    im += 1
                    
                    
            if imrotate:
                for angle in [90, 180, 270]:
                    
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigy_temp_crop=crop_circle(bigy_rot, img_size/2, img_size/2, mask) 
                    sinogram, _ = create_sino(bigy_temp_crop,no_angles)
                    bigx[im, :, :, :] = np.expand_dims(sinogram, axis = 0);
                    bigy[im, :, :, :] = np.expand_dims(bigy_temp_crop, axis = 0);
                    
                    im += 1
                    

        if im > (n_im - 1):  # how many images to load
            break
    
    #bigy=cv2.normalize(bigy, bigy, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bigx=cv2.normalize(bigx, bigx, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return bigx, bigy

if __name__ == "__main__":
    dir_temp=r'Data\ImageNet'
    #X, Y = load_images(dir_temp, 10, 128, 180, imrotate=True)
    n_im = 100
    img_size = 128
    no_angles = 128
    min_output = -0.5
    max_output = 0.5 
    X, Y = load_images_normalize_shift_rotate_large(dir_temp, n_im, img_size, no_angles, shift = True, imrotate = True)
    theta = np.linspace(0., 180., no_angles, endpoint=True)

    #Y = load_images_y(dir_temp, n_im, img_size,  min_output, max_output, shift = True, imrotate = True)
    print (np.shape(Y))
    #X = create_sino_X (Y, no_angles,min_output, max_output, )
    
    print('max X:',np.amax(X))
    print('min X:',np.amin(X))
    print('max Y:',np.amax(Y))
    print('min Y:',np.amin(Y))
    
    k=7*5
    
    plt.subplot(331), plt.imshow(np.squeeze(Y[k, :, :, :]), cmap='gray')

    plt.title('Y_ori'), plt.xticks([]), plt.yticks([])

    plt.subplot(334), plt.imshow(np.squeeze(Y[k+1, :, :, :]), cmap='gray')

    plt.title('Y_shift 10 vertical up'), plt.xticks([]), plt.yticks([])

    plt.subplot(335), plt.imshow(np.squeeze(Y[k+2, :, :, :]), cmap='gray')

    plt.title('Y_shift 10 horizontal left'), plt.xticks([]), plt.yticks([])

    plt.subplot(336), plt.imshow(np.squeeze(Y[k+3, :, :, :]), cmap='gray')

    plt.title('Y_shift 10 horizontal right'), plt.xticks([]), plt.yticks([])
    


    plt.subplot(337), plt.imshow(np.squeeze(Y[k+4, :, :, :]), cmap='gray')

    plt.title('Y_rot90'), plt.xticks([]), plt.yticks([])

    plt.subplot(338), plt.imshow(np.squeeze(Y[k+5, :, :, :]), cmap='gray')

    plt.title('Y_rot180'), plt.xticks([]), plt.yticks([])

    plt.subplot(339), plt.imshow(np.squeeze(Y[k+6, :, :, :]), cmap='gray')

    plt.title('Y_rot270'), plt.xticks([]), plt.yticks([])

    plt.show()



    plt.subplot(331), plt.imshow(np.squeeze(X[k, :, :, :]), cmap='gray')

    plt.title('X_ori'), plt.xticks([]), plt.yticks([])

    plt.subplot(334), plt.imshow(np.squeeze(X[k+1, :, :, :]), cmap='gray')

    plt.title('X_shift 10 vertical up'), plt.xticks([]), plt.yticks([])

    plt.subplot(335), plt.imshow(np.squeeze(X[k+2, :, :, :]), cmap='gray')

    plt.title('X_shift 10 horizontal left'), plt.xticks([]), plt.yticks([])

    plt.subplot(336), plt.imshow(np.squeeze(X[k+3, :, :, :]), cmap='gray')

    plt.title('X_shift 10 horizontal right'), plt.xticks([]), plt.yticks([])
    


    plt.subplot(337), plt.imshow(np.squeeze(X[k+4, :, :, :]), cmap='gray')

    plt.title('X_rot90'), plt.xticks([]), plt.yticks([])

    plt.subplot(338), plt.imshow(np.squeeze(X[k+5, :, :, :]), cmap='gray')

    plt.title('X_rot180'), plt.xticks([]), plt.yticks([])

    plt.subplot(339), plt.imshow(np.squeeze(X[k+6, :, :, :]), cmap='gray')

    plt.title('X_rot270'), plt.xticks([]), plt.yticks([])

    plt.show()

    reconstruction_fbp = iradon(np.squeeze(X[k+6, :, :, :]), theta=theta, circle=True)
    plt.imshow(reconstruction_fbp, cmap='gray')
    plt.show()

    
    print (np.shape(X))
