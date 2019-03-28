function [images_3D, rect]=read3DVol(path,scale,gap,st,last,line)
%% change folder
cd (path)

%% select ROI
Imgfile=dir('*.tiff');
% NF = round(length(Imgfile)/2)+327; % 180 degree  + cone beam
NF=length(Imgfile); % read 360 degree
% NF=827;
% % select ROI
% downscale image
%ScaleFactor=0.2; 
%B = imresize(A,ScaleFactor)

image_ex=imread(Imgfile(1).name);

%scale=1; % downsize data
image_ex= imresize(image_ex,scale);

figure
imshow(image_ex,[])
% size_detector=size(image_ex);
title('select ROI')
rect =uint16(getrect); 
if line==1
rect(4)=1; % only read one line
end

% figure
% imshow(image_ex(rect(2):rect(2)+rect(4)-1,rect(1):rect(1)+rect(3)-1),[]);
% title('selected ROI')
%% read all image
%gap=1; % setting parameters for taking gap in projections
%st=1; % started number of frames
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if last<=NF
Index_image=st : gap: last;
number_proj=length(Index_image);
% images = cell(number_proj,1);
images_3D=zeros(rect(4),rect(3),number_proj);

  for k = 1 : number_proj
  A = imread(Imgfile(Index_image(k)).name);
   A= imresize(A,scale);
  %images{k}=A(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
  images_3D(:,:,k)=A(rect(2):rect(2)+rect(4)-1,rect(1):rect(1)+rect(3)-1);
  end

else 
    disp('selected angle outside range')
end

end