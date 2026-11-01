function [L6,L5] = gradient_LR(img)
img_hsv=rgb2hsv(img);
V=img_hsv(:,:,3);

N_row = 3;
sigma = 1;
gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
blur1 = imfilter(V, gausFilter, 'symmetric');

N_row = 40;
sigma = 8;
gausFilter = fspecial('gaussian', [N_row, N_row], sigma);
blur2 = imfilter(V, gausFilter, 'symmetric');
blur2(blur2 < 0.000000001) = 0.000000001;

L = blur2;
% R = 0.75 * (blur1 ./ blur2);
R =  (blur1 ./ blur2);

% h = fspecial('average', [7 7]);
% R_1=imfilter(R, h);
% % figure
% % imshow(L)
% % figure
% % imshow(R)
% R=R_1;

% imwrite(R,'ideal_R.png')
[L5,~]=imhist(L);
[L6,~]=imhist(R);
end

