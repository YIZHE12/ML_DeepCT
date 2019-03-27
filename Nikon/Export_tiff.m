% convert vgi to uint16 tif
% This code will:
% Find the VGI file
% Create a new folder in the current directory 
% Open the VGI file in a text editor and find the image size 
% the imagesize is given by nImage,mImage,lImage (size in .vgi file)
% Read the VOL file based on the given image size and output tif file to
% the new folder 'Recon_tif + sample name'
% Created: 17/6-2015 by Carsten Gundlach (cagu@fysik.dtu.dk)
% Edited: 4/10-2018 by Yi Zheng (yizhe@fysik.dtu.dk)
clear all
close all
clc

%% find the VOL size from the VGI file
fileList = dir('*.vgi');
file_name_vgi=fileList.name;
filetext  = fileread(file_name_vgi);
expr = '[^\n]*Size[^\n]*';
matches = regexp(filetext,expr,'match');
disp(matches{1})
expression = '\ ';
splitStr = regexp(matches,expression,'split');
nImage=str2num(cell2mat(splitStr{1,1}(3)));
mImage=str2num(cell2mat(splitStr{1,1}(4)));
lImage=str2num(cell2mat(splitStr{1,1}(5)));
%% create a new folder
Foldername = strcat('Recon_tif_',fileList.name(1:end-4));
mkdir (Foldername)
Foldername = strcat(Foldername,'\',fileList.name(1:end-4));
%%
tic
% FileStem='H:\Hay\Compact\Hay_120kV_11W_air_01\Hay_120kV_11W_air'
% FileImportName=[FileStem '.vol']
% size=[780 765 982];
% nImage=size(1);
% mImage=size(2);
% mImage=size(3);
fileList = dir('*.vol');
file_name_vol=fileList.name;
fid = fopen(file_name_vol);
colormap(gray)
for i=1:lImage
    A = fread(fid, nImage*mImage*1, 'float');
    vol = reshape(A, [nImage, mImage, 1]);
%     figure
%     imshow(vol,[])
    
    vol2 = uint16(vol*100);
    NewFileName=sprintf('%s_convert_%04d.tif',Foldername,i);
    imwrite(vol2(:,:,1),NewFileName,'tif')
%    i
end
fclose(fid)
toc
%%