# X-ray and CT image processing using machine learning and deep learning

___
## [Image reconstruction](Image_reconstruction)

### Traditional method - filter back projection

<img src=/images/ctsim_a60.gif height = 300> (http://xrayphysics.com/ctsim.html)

#### Backprojection
The standard method of reconstructing CT slices is backprojection. This involves "smearing back" the projection across the image at the angle it was acquired. By smearing back all of the projections, you reconstruct an image. This image looks similar to the real picture but is blurry - we smeared bright pixels across the entire image instead of putting them exactly where they belonged. You can see this effect in the simulator on the right-most panel.

In order to reconstruct an image, you need 180 degrees of data (* actually 180 + fan beam angle). Why? The remaining 180 degrees are simply a mirror image of the first (because it does not matter which way a photon travels through tissue, it will be attenuated the same amount). (Because of the fan beam geometry, you need to measure an extra amount - equal to the fan angle - to actually get all of the data you need, but the concept is the same.)

In a fan-beam geometry, the angle of the fan determines how much of the object is included in the reconstructible field of view. A point must be included in all 180 degrees of projections in order to be reconstructed correctly.

#### Filtered Backprojection
As you may have noticed, backprojection smears or blurs the final image. In order to fix the blurring problem created by standard backprojection, we use filtered backprojection. Filtering refers to altering the projection data before we do the back-projections. The particular type of filter needed is a high-pass filter, or a sharpening filter. This type of filter picks up sharp edges within the projection (and thus, in the underlying slice) and tends to ignore flat areas. Because the highpass filter actually creates negative pixels at the edges, it subtracts out the extra smearing caused by backprojection. Thus, you end up with the correct reconstruction (see the simulator panel labeled "Filtered BP Reconstruction").
___
### [Iterative method](/Image_reconstruction/Astra)

<img src=/images/IT_CT_recon.png height = 500>
(https://pubs.rsna.org/doi/pdf/10.1148/radiol.2015132766)

Schematic representation of the principle steps of iterative image algorithms: following the CT
acquisition process (measured projections), a first image estimate is generated. An x-ray beam is simulated
via forward projection to obtain simulated projection data, which are then compared with the measured
projection data. In case of discrepancy, the first image estimate is updated based on the characteristics of
the underlying algorithm. This correction of image and projection data is repeated until a condition predefined
by the algorithm is satisfied and the final image is generated.

Compared to the traditional filtered back projection, iteratvie CT reconstruction can recover the signal from much more nosiy data. Therefore, we can dramatically reduce the exposure time for each CT projection (angle) and/or the total number of projections (angles, leading to less patient radiation exposure or higher time resolution for dynamic CT.

Combining with a smart sampling scheme in [my paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10391/103910M/Micro-CT-in-situ-study-of-carbonate-rock-microstructural-evolution/10.1117/12.2273877.full?SSO=1), I show that with iterative reconstruction method, it is possible to increase the time resolution of CT by 8-fold.

<img src=Image_reconstruction/Astra/example.png height = 300>

___

### [Deep learning method](/Image_reconstruction/Deep_learning)

Automated transform by manifold approximation (Automap) is a deep learning method with convolutional neural network. Originally developed for MRI image reconstruction. 
<img src=images/automap_ori.jpg height = 400>

I further developed Automap for [CT image reconsturction](Image_reconstruction/Deep_learning). CT has a different sampling scheme as MRI: MRI acquisition scheme is in k-space with complex number while CT data acquisition scheme is a sinogram. 

I also added dropout layers into the original network design and show that in this way, we can not only reconstruct the CT image with little number of projections, but also reconstruct it even when there is large distrotion due to missing data (bad pixels on detector) or random shift (patient movement).

<img src=Image_reconstruction/Deep_learning/automap.jpg height = 500>

___

## [Image registration and stitching](Image_registration_n_stitching) 

Often, industrial CT imaging is limited by its field of view when we are aiming at a high spatial resolution. Here, I demonstrated how to use to use machine learning methods for CT image registrion and to stitch two CT volumn to increase the field of view. I first used Oriented FAST and rotated BRIEF feature detector (ORB) to detect features in the two images, then used RANSAC to register the two images using the detected features. 

Orb algorithm uses a multiscale image pyramid. Each level in the pyramid contains the downsampled version of the image than the previous level. Once orb has created a pyramid it uses the fast algorithm to detect keypoints in the image. By detecting keypoints at each level orb is effectively locating key points at a different scale. 

<img src = images/orb2.png>

Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates. Therefore, it also can be interpreted as an outlier detection method.

<img src = images/ransac.png>

Combining ORB and RANSAC results in a robust image registration techniques that can be used in many different applications, including face recognition]. 

<img src = images/ORB_RANSAC.png> [(Vinay A et al, Procedia Computer Science, 2015)](https://www.sciencedirect.com/science/article/pii/S1877050915032329)



___

## [Image segementation](https://github.com/YIZHE12/ML_DeepCT/tree/master/Image_segmentation/Image_segmentation_using_U-net) 
### 1.Image annotation

[using mouse as paint brash for labelling images for segmentation](Image_segmentation/Image_segmentation_using_U-net/Pipeline_step1_image_annotation.ipynb) to create label data for image segmentation
### 2.Image data manipulation: framing, slicing, resizing

[preparation for CNN](Image_segmentation/Image_segmentation_using_U-net/crop_predict_stitch.ipynb)
### 3.Image Segmentation

[Unet](image_segmentation/Image_segmentation_using_U-net/unet.jpg) 

<img src=Image_segmentation/Image_segmentation_using_U-net/unet.jpg height = 400>

## [Image analysis](https://github.com/YIZHE12/ML_DeepCT/tree/master/Image_analysis)
[Regional porosity analysis](Image_analysis/porosity_regional_stats.py)

<img src=Image_analysis/pore_size_distribution.png height = 400>

___

## Other functions: 


[Nikon CT file export](vgi_to_tiff)
