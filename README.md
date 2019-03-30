# X-ray and CT image processing

## [Image reconstruction](Image_reconstruction)

### Astra 
Astra is a CUDA based GPU toolkit for X-ray CT image reconsturction using algebraic reconstruction techniques. 
[In this example](Image_reconstruction/Astra), I show how to combine Astra with novel resampling method to increase the time resolution of CT by 8-fold.

<img src=Image_reconstruction/Astra/example.png height = 300>
Astra tool box (https://www.astra-toolbox.com) is developed and maintained by  iMinds-Vision Lab, University of Antwerp http://visielab.uantwerpen.be/ and 2014-2016, CWI, Amsterdam 

### Automap
Automap is a deep learning method with convolutional neural network. Originally developed for MRI image reconstruction. Here, I showed that it can also be used for [CT image reconsturction](Image_reconstruction/Deep_learning), which has a different sampling scheme as MRI: MRI acquisition scheme is in k-space with complex number while CT data acquisition scheme is a sinogram. Here, I showed that with deep learning method, we can not only reconstruct the CT image with little number of projections, but also reconstruct it even when there is large distrotion due to missing data (bad pixels on detector) or random shift (patient movement).

<img src=Image_reconstruction/Deep_learning/automap.jpg height = 300>

## [Image registration and stitching](pre_processing) 

Often, industrial CT imaging is limited by its field of view when we are aiming at a high spatial resolution. Here, I demonstrated how to use to use machine learning methods for CT image registrion and to stitch two CT volumn to increase the field of view.

<img src=pre_processing/example_ransac.png height = 400>
feature matching using RANSAC removing outlier


## [Image segementation and porosity analysis]() 
### 1.[Image annotation](Image_segmentation/Label_annotation_with_mouse) 
using mouse and opencv to create label data for image segmentation
### 2.[Image prepration for segmentation: framing, slicing, resizing]()
### 3.[Image Segmentation]()
Markov random field and layer detection image segmentation toolkit originated from http://qim.compute.dtu.dk/tools/
The implementation of U-net for image segmentation comes largely from https://github.com/zhixuhao/unet
### 4.[Regional porosity analysis]()



## Other functions: 


[Nikon CT file export](vgi_to_tiff)
