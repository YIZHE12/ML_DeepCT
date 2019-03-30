### Reasons for data preparation 
1. When using U-net for segmentation, we need to make sure the input image has the same size as the required input of the model. Therefore, it requires we resize the images.
2. As most of the convolutional network, one side effect of using pulling layer is that it lose some spatial resolutions. Although U-net has skip connection to reserve some features in the higher resoltuion. In practice, we still find some resolution loss. Therefore, here, I upsampling the data before the image segmentation.
3. I don't want to strech the image when we resizing it. Therefore, I first pad zero to the image to make the image size (column and row) = 2^n. 
### Training pipeline
1. use mouse to draw labels on images for training
2. prepare the label data to correct size (same as prediciton procedure)
3. data augmentation to increase training data
4. define and train the model
### Prediction pipeline
Pad zero to the image making column and row = 2^n

Slice the image to 2^m smaller pieces, which will be upsampled to the size of the network in data generator before feeding to the network

Stitch small patches to recover the original image, this will be used for the prediction results (segmented images)

Note: the first two steps are also required for the training data 
