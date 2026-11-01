function [L_new,new_L,R,result2,iter] = fun_for_hrm_mex(img,gamma,L6,L5,P,alpha,beta,maxIter,tolL,tolR)
%FUN_FOR_HRM_MEX MEX accelerated version of fun_for_hrm.
%
% Original interface:
%   [L_new,new_L,R,result2] = fun_for_hrm(img,gamma,L6,L5,P,alpha,beta)
%
% Accelerated interface:
%   [L_new,new_L,R,result2] = fun_for_hrm_mex(img,gamma,L6,L5,P,alpha,beta)
%   [L_new,new_L,R,result2,iter] = fun_for_hrm_mex(img,gamma,L6,L5,P,alpha,beta,maxIter)
%   [L_new,new_L,R,result2,iter] = fun_for_hrm_mex(img,gamma,L6,L5,P,alpha,beta,maxIter,tolL,tolR)
%
% Notes:
%   1. L6 and L5 must be 256x1 or contain exactly 256 elements.
%   2. maxIter is the external loop-count interface. The original code used iter > 3000.
%   3. tolL and tolR correspond to the original convergence thresholds 0.1 and 0.1.
%   4. mystand.m must be in the MATLAB path, because this wrapper keeps the same preprocessing logic.

    if nargin < 8 || isempty(maxIter)
        maxIter = 3000;
    end
    if nargin < 9 || isempty(tolL)
        tolL = 0.1;
    end
    if nargin < 10 || isempty(tolR)
        tolR = 0.1;
    end

    L6 = double(L6(:));
    L5 = double(L5(:));
    if numel(L6) ~= 256
        error('fun_for_hrm_mex:L6Size','L6 must be 256x1 or contain exactly 256 elements.');
    end
    if numel(L5) ~= 256
        error('fun_for_hrm_mex:L5Size','L5 must be 256x1 or contain exactly 256 elements.');
    end

    img_hsv = rgb2hsv(img);
    [N1,C1] = imhist(img_hsv(:,:,3));
    L_init = imhist(mystand(img_hsv(:,:,3)));

    N1 = double(N1(:));
    C1 = double(C1(:));
    L_init = double(L_init(:));

    [L_new,new_L,R,result2,iter] = fun_for_hrm_core_mex( ...
        N1,C1,L_init,L6,L5,double(gamma),double(P),double(alpha),double(beta), ...
        double(maxIter),double(tolL),double(tolR));
end
