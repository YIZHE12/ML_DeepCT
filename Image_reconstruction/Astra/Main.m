close all
clear all
clc

path='..';% here put your data path
savepath = '..'% here put where you want to store the reconstruciton data
%% read files with selected ROI
gap=1; % sampleing gap
st=1; % starting frame
last=1200; % ending frame
scale=1; % downscale factor, between 0 to 1, 1 means no scale
line=1; % whether to just read one line
[images_3D, rect]=read3DVol(path,scale,gap,st,last,line);
% rect is selected ROI
%% sampling based on selected parameters
st=1; % starting frame
last=827; % ending frame
plus=500;
st=st+plus;
last=last+plus;
gap=1;
images_3D_sub=images_3D(:,:,st:gap:last);
%% setting up parameters
size_detector=[1015 1015];
Ori_pixel=13.5*10^-3; % original detector pixel size 14 micron
bin=2;% binning of the detector system
number_proj=1601;
Ori_pixel=Ori_pixel*bin;
voxel_size=3.1985*10^-3;%17.5425*10^-3;
%voxel_size=voxel_size/scale;
MF1=Ori_pixel/voxel_size; % check the intrinsict magnification factor by looking at the pixel size
%% Create projection geometry
real_scale = 1.0 /voxel_size; %0.14050885988; % pixel/pixel size
det_spacing_x=1.0; %pixel size: distance between the centers of two horizontally adjacent detector pixels
det_spacing_y=1.0; %pixel size: distance between the centers of two vertically adjacent detector pixels
det_row_count=rect(4); %number of detector rows in a single projection
det_col_count=rect(3); %number of detector columns
source_origin=61.7614;%41.0576;  %distance between the source and the center of rotation
origin_det=68.0137;%117.6249;     %distance between the center of rotation and the detector array
angles = linspace2(0, 2*pi, number_proj); % start, end , number
angles=angles (st:gap:last);
Optic=4.0173; % 4X optic
MF=(origin_det+source_origin)*Optic/source_origin; % Magnification factor
Det=(MF-1)*source_origin;
DetectorPixelsX=rect(3);
%geom_dets_y=transpose(Ori_pixel*((-(DetectorPixelsX-1)/2):((DetectorPixelsX-1)/2)));
det_width =Ori_pixel/voxel_size;


%% find rotation center
% 
%for i=-5:0
i=-2.207;

PixelShift=i; % center of rotation from the software


Center_detector=double(size_detector(1)/2); % center of detector in row direction
Center_ROI=double(rect(1)+rect(3)/2);

center_shift=Center_ROI-Center_detector+PixelShift;

%histogram(images_3D(:,:,1))
images_3D_rec=imtranslate((100-images_3D_sub),[center_shift,0,0],'OutputView','same');
crop=abs(round(center_shift))+1;
images_3D_crop=single(images_3D_rec);
images_3D_crop(:,1:crop,:)=[];
images_3D_crop(:,end-crop+1:end,:,:)=[];

det_col_count=det_col_count-crop*2;
%% Create projection data
%proj_data = permute(images_3D_rec,[2,3,1]); % up needs to be in z direction
proj_geom = astra_create_proj_geom('cone',  det_width, det_width, det_row_count, det_col_count, angles, source_origin*real_scale, Det*real_scale);
% proj_geom_vec = astra_geom_2vec(proj_geom); % Get the vector form

proj_data = permute( images_3D_crop,[2,3,1]); 
proj_id = astra_mex_data3d('create', '-proj3d', proj_geom);


% create 3D sinogram
% figure, imshow(squeeze(proj_data(:,1,:))',[]) % projection
% title ('projection')
figure
x0=10;
y0=10;
width=550;
height=400;
set(gcf,'units','points','position',[x0,y0,width,height]) 
imshow(squeeze(proj_data(:,:,1))',[]) % sinogram
str = sprintf('pixelshift=%d',i);
title(str)
%% create a volume for reconstuction
vol_geom = astra_create_vol_geom(rect(3), rect(3), rect(4)); % number of rows, number of columns, number of slices
rec_id = astra_mex_data3d('create', '-vol', vol_geom);% allocate space to store reconstructed data
proj_geom = astra_create_proj_geom('cone',  det_width, det_width, det_row_count, det_col_count, angles, source_origin*real_scale, Det*real_scale);

%% reconstruction

tic
% Create a data object for the reconstruction


astra_mex_data3d('set',proj_id,proj_data);
% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct('SIRT3D_CUDA');
% cfg = astra_struct('FDK_CUDA'); 
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = proj_id;
% cfg.option.MinConstraint=0;
% cfg.option.MaxConstraint = 256;
% Create the algorithm object from the configuration structure
alg_id = astra_mex_algorithm('create', cfg);

% Run 150 iterations of the algorithm
% Note that this requires about 750MB of GPU memory, and has a runtime
% in the order of 10 seconds.
astra_mex_algorithm('iterate', alg_id, 150);

% Get the result
rec = astra_mex_data3d('get', rec_id);


% save rec.mat
%
%Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
astra_mex_algorithm('delete', alg_id);
astra_mex_data3d('delete', rec_id);
astra_mex_data3d('delete', proj_id);

toc

%

% write image

s=size(rec);
nImage=s(1);
mImage=s(2);
lImage=s(3);

center=1;
C=(rec(:,:,center)+2)*1000;
figure, imshow(squeeze(C),[1996 2212]);
str = sprintf('pixelshift=%d',i);
title(str)
%% remove the ring
vol = (rec+1)*10000;
vol2 = uint16(vol);

% create a mask
rad=double(rect(3))/2;
c = -rad+0.5:rad-0.5;
[x, y] = meshgrid(-rad+0.5:rad-0.5,-rad+0.5:rad-0.5);
mask = uint16((x.^2 + y.^2 < (rad-15)^2));
% figure(1); imshow(mask, []);
%%
FileStem=savepath;
fid = rec;
colormap(gray)
%
for i=1:lImage
    
    NewFileName=sprintf('%sGap%d_st%d_fin%d_%04d.tif',FileStem,gap,st,last,i);
    imwrite(vol2(:,:,i).*mask,NewFileName,'tif')
%    i
end
