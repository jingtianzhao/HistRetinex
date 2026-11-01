clear all
close all
clc

addpath D:\work\matlab_code\Function\
addpath D:\work\matlab_code\benchmark\
addpath D:\work\dataset\LOLdataset\eval15\low\
addpath D:\work\dataset\LSRW\Huawei\low\
addpath D:\work\python_code\LatentDark-main\
addpath D:\work\python_code\MWFFnet-main\
addpath D:\work\python_code\HVI-CIDNet-master\
addpath D:\work\python_code\Ghillie-main\


img_or = imread('House.png');
% img_or = imresize(img_or,[1000,1000]);
img = img_or;
img_hsv = rgb2hsv(img);

img_name = '9';

noise_type = 'nonoise';
noise_level = '0';
algorithm = ["original","Ghillie","CIDNet","MWFFnet","LatentDark","proposed"];

%% proposed, MEX accelerated version
[L6,L5] = gradient_LR(img);

maxIter = 20;

tic
[~,~,~,result2,iter] = fun_for_hrm_mex(img,2.5,L6,L5,1,0.1,0.1,maxIter);
time8 = toc;

fprintf('MEX version time: %.4f s, iter = %d\n', time8, iter);

new_V = histeq(img_hsv(:,:,3),result2);
new_image = img_hsv;
new_image(:,:,3) = new_V;
new_image_rgb = hsv2rgb(new_image);


figure
imshow(new_image_rgb)

