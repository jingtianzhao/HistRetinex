function [L_new,new_L,R,result2]=fun_for_hrm(img,gamma,L6,L5,P,alpha,beta)

[m,n,~]=size(img);
N=m*n;
img_hsv=rgb2hsv(img);

%% Compute the initial image histogram
[N1,C1]=imhist(img_hsv(:,:,3));

%% Construct the histogram-position matrix of the original image
E_C=C1.^P;
B=E_C*E_C';
index=zeros(256,256); % For inspection

for i=1:256
    for j=1:256
        dist=inf;
        for k=1:256
            if abs(B(i,j)-E_C(k))<=dist && N1(k)~=0
                dist=abs(B(i,j)-E_C(k));
                index(i,j)=k;
            end
        end
    end
end

%% Initialize the weight matrix
weight_matrix=zeros(256,256);
for i=1:256
    for j=1:256
        weight_matrix(i,j)=1/(sum(sum(index==index(i,j)))+eps);
    end
end

%% Initialize the reference distributions
temp=L6;
temp2=L5;

%% Initialize the L and R variables
L=imhist(mystand(img_hsv(:,:,3)));
R=zeros(256,1);
R_old=zeros(256,1);

%% Iterative optimization
iter=1;
while true
    if iter~=1
        L=L_new;
        R_old=R;
    end

    % Update R
    de=alpha+sum(L.^2)/(N^2);
    mo=zeros(256,1);
    for i=1:256
        mo(i)=sum(weight_matrix(i,:).*L(:)'.*N1(index(i,:))')/N;
    end
    R=(mo+alpha*temp)/de;

    % Update L
    de=beta+sum(R.^2)/(N^2);
    mo=zeros(256,1);
    for i=1:256
        mo(i)=sum(weight_matrix(:,i)'.*R(:)'.*N1(index(:,i))')/N;
    end
    L_new=(mo+beta*temp2)/de;

    % Compute the contribution matrix
    S1=R.*L_new'/(N);
    new_hist=zeros(256,1);
    for i=1:256
        for j=1:256
            new_hist(index(i,j))=S1(i,j)+new_hist(index(i,j));
        end
    end

    % Normalize the contributions within each histogram bin
    for i=1:256
        for j=1:256
            weight_matrix(i,j)=S1(i,j)/(new_hist(index(i,j))+eps);
        end
    end

    % Aggregate the contributions according to the nearest-bin assignments
    if sum(abs(L_new-L))<0.1 && sum(abs(R_old-R))<0.1 || iter>10
        break;
    end
    iter=iter+1;
end

%% Normalize the estimated distributions
L_new=L_new*N/sum(L_new);
R=R*N/sum(R);

%% Apply contrast stretching
C2=C1.^(1/gamma);
ind=zeros(256,1);
new_L=zeros(256,1);

for i=1:256
    dist=inf;
    for j=1:256
        if abs(C2(i)-C1(j))<dist
            dist=abs(C2(i)-C1(j));
            ind(i)=j;
        end
    end
end

for i=1:256
    new_L(ind(i))=new_L(ind(i))+L_new(i);
end

%% Merge the matrices
result2=zeros(256,1);
A=R*new_L'; % Represents the pixel counts

index_2=zeros(256,256); % For inspection
C3=C1*C1'; % Construct the histogram-position matrix

for i=1:256
    for j=1:256
        dist=inf;
        for k=1:256
            if abs(C3(i,j)-C1(k))<=dist
                dist=abs(C3(i,j)-C1(k));
                index_2(i,j)=k;
            end
        end
    end
end

%% Accumulate the contributions into the output histogram
for i=1:256
    for j=1:256
        result2(index_2(i,j))=A(i,j)+result2(index_2(i,j));
    end
end

%% Normalize the output histogram
result2=result2/N;