clear all
close all
clc
addpath D:\work\matlab_code\Retinex-Based-Multiphase-Algorithm-for-Low-Light-Image-Enhancement-main\
addpath D:\work\matlab_code\benchmark\VIFutils\matlabPyrTools-master\TUTORIALS\
addpath D:\work\matlab_code\Brain-like-Retinex-main\
addpath D:\work\matlab_code\image-enhancement-about-Retinex-master\
addpath D:\work\matlab_code\LIME-master\
addpath D:\work\matlab_code\Low-light-image-enhancement-master\
addpath D:\work\matlab_code\MSRetinex\
addpath D:\work\matlab_code\STAR-TIP2020-master\
addpath D:\work\matlab_code\Function\
addpath D:\work\matlab_code\Brain-like-Retinex-main\image\
addpath D:\work\matlab_code\benchmark\
addpath D:\work\matlab_code\benchmark\VIFutils\
addpath D:\work\matlab_code\benchmark\VIFutils\matlabPyrTools-master\
addpath D:\work\matlab_code\SSD-master\
addpath D:\work\matlab_code\Infrared-image-enhancement-using-adaptive-histogram-partition-and-brightness-correction-master\
addpath D:\work\dataset\BSDS500\BSDS500\data\images\train\
addpath D:\work\dataset\BSDS500\BSDS500\data\images\test\
addpath D:\work\dataset\BSDS500\BSDS500\data\images\val\
addpath D:\work\dataset\LOLdataset\our485\low\
addpath D:\work\dataset\LOLdataset\eval15\low\
addpath D:\work\matlab_code\WVM\
addpath D:\work\matlab_code\BIMEF-master\
addpath D:\work\dataset\MEF\
addpath D:\work\dataset\LIME\
addpath D:\work\dataset\ExDark\
addpath D:\work\dataset\NPE\
addpath D:\work\dataset\ExDark\Bicycle\
addpath D:\work\dataset\ExDark\Cat\
addpath D:\work\dataset\ExDark\Boat\
addpath D:\work\dataset\ExDark\Bottle\
addpath D:\work\dataset\ExDark\Bus\
addpath D:\work\dataset\ExDark\Dog\
addpath D:\work\dataset\ExDark\Table\
addpath D:\work\dataset\LSRW\Huawei\low\
addpath D:\work\dataset\LSRW\Nikon\low\
addpath D:\work\matlab_code\LR3M-Method-master\code\
addpath D:\work\matlab_code\LR3M-Method-master\code\lowRank\
addpath D:\work\matlab_code\ourFSTAR\
%% 
dataset_path1='D:\work\matlab_code\thesis2\yds\';
save_path='D:\work\matlab_code\thesis2\yds\output\';
file_list1 = dir(dataset_path1);

file_list=[file_list1(3:end)];
for i =1:length(file_list)
    img_or=imread([file_list(i).folder,'\',file_list(i).name]);
    img_or=imresize(img_or,[664,1000]);
    img=img_or; %
    img_hsv=rgb2hsv(img);
    %% 
    [~, img_name, ~] = fileparts(file_list(i).name);
    % noise_type='salt';
    % noise_type='gaussian';
    % noise_type='speckle';
    noise_type='nonoise';
    noise_level='0';
    algorithm="HistRetinex";
    %% proposed
    [L6,L5] = gradient_LR(img);
    [~,~,~,result2]=fun_for_hrm(img,0.5,L6,L5,1,0.1,0.1);
%     time8=toc;
    new_V=histeq(img_hsv(:,:,3),result2);
    new_image=img_hsv;
    new_image(:,:,3)=new_V;
    new_image_rgb=hsv2rgb(new_image);
    new_image_rgb=new_image_rgb*255;
    imwrite(uint8(new_image_rgb),[save_path,img_name,char(algorithm),'.png']);
end