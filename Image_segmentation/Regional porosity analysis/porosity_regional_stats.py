#!/usr/bin/env python
# coding: utf-8




import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.pyplot import savefig

from skimage.measure import regionprops
from skimage.measure import label


# # Read segmentaiton image from folder


def read_img_folder(read_path): 
    images = []
    for filename in range(0,52):
        f = read_path + '\\' + str (filename) + '.jpg'
        #print (f)
        img =cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            
    images = np.asarray(images)        
    return images

def crop_3D(image_3D, N_pieces, mode = 'row', channel_first = True):
    
    if channel_first == False:
        image_3D=np.moveaxis(image_3D, -1, 0) # move the last axis to the first
        
    s = np.shape(image_3D)
    
    if mode == 'row': # crop to several rows
        axis = int(1)
    elif mode == 'col':
        axis = int(2)
    else:
        axis = int(0)
        
    extra_lines = s[axis]%N_pieces 
    image_3D = np.delete(image_3D, np.s_[-extra_lines:], axis) # remove extra lines to make it be able to divide evenly
    
    I_crop = np.asarray(np.split(image_3D, N_pieces, axis = axis)) 
    
    if channel_first == False:
            I_crop=np.moveaxis(I_crop, 0, -1) # change back to channel last

    return(I_crop)
    
def cal_diameter(img): # caclulate the equivalent diameter for each segmented regions
    label_img = label(img,background=0)
    regions  = regionprops(label_img)
    return ([props.equivalent_diameter  for props in regions])

def cal_diameter_regions(img_block):
    N_block = np.shape(img_block)[0]
    sns.set(font_scale=5)
    fig, axes = plt.subplots(N_block, 1, figsize=(40, 40), sharex='col', sharey='row')

    for i in range(0, N_block):
        D = np.asarray(cal_diameter(img_block[i]))
        filtered = D[(D >= 0) & (D < 30)]
        #print (D)

        sns.distplot(filtered.ravel(),ax=axes[i])
        
# Calculate the porosity of each sub-divided regions
def porosity_image(image_stack):
    poro = []
    for i in range(0, np.shape(image_stack)[0]):
        img = image_stack[i].ravel()
        p = np.sum(img)/len(img)
        poro.append(p*100)
        
    return(poro)

seg = read_img_folder(r'') # put your folder of the segmentation results
k = 15 # remove the top and bottom
seg = np.delete(seg, np.s_[0:k], 1)
seg = np.delete(seg, np.s_[-k:], 1)
plt.imshow(seg[10, :,:], cmap = 'gray')

new_seg = np.zeros(np.shape(seg))
new_seg[seg == 0] = 1


plt.imshow(new_seg[10, :,:], cmap = 'gray')
np.shape(new_seg)



image_3D = np.copy(new_seg)
np.shape(image_3D)
## by row
N_pieces = 6
image_3D_crop = crop_3D(image_3D, N_pieces)
print (np.shape(image_3D_crop))

plt.imshow(image_3D_crop[2][25,:,:], cmap = 'gray')

poro=porosity_image(image_3D_crop)

print ('average porosity:',"{0:0.2f}".format(np.mean(poro)),'+/-',"{0:0.2f}".format(np.std(poro)),)
print (["%0.2f%%" % i for i in poro])

cal_diameter_regions(image_3D_crop)
savefig('d_row.png')

## by column 
image_3D_crop = crop_3D(image_3D, N_pieces,  mode = 'col')
print (np.shape(image_3D_crop))

plt.imshow(image_3D_crop[2][25,:,:], cmap = 'gray')

poro=porosity_image(image_3D_crop)
print ('average porosity:',"{0:0.2f}".format(np.mean(poro)),'+/-',"{0:0.2f}".format(np.std(poro)),)
print (["%0.2f%%" % i for i in poro])

cal_diameter_regions(image_3D_crop)
savefig('d_col.png')
## by depth
image_3D_crop = crop_3D(image_3D, 3,  mode = 'depth')
print (np.shape(image_3D_crop))

poro=porosity_image(image_3D_crop)
print ('average porosity:',"{0:0.2f}".format(np.mean(poro)),'+/-',"{0:0.2f}".format(np.std(poro)),)
print (["%0.2f%%" % i for i in poro])

cal_diameter_regions(image_3D_crop)
savefig('d_depth.png')

