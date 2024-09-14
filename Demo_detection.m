clc;clear all;close all;
%% 读入数据
load('.\Data\HYDICE_data.mat');
load('.\Data\initial_det.mat');
%% 加载GCN的feature
[row,col,bands]=size(data);
A = ones(row,col);
load('.\Features\gcnfeature.mat');
features = hyperNormalize(features);
%% GCN检测网络输出
X = reshape(data,8000,162);
X = X';
N = size(X,2);
D = size(X,1);
groundtruth = map;
imgH = size(groundtruth,1);
imgW = size(groundtruth,2);
SNR = 30;
for i = 1:size(X,2)
       X(:,i) = awgn(X(:,i), SNR);
end
lambda = 200;
epsilon = 1e-6;
Weight = ones(1,N);
y_old = ones(1,N);
max_it = 100;
Energy = [];
for T = 1:max_it
     for pxlID = 1:N
         X(:,pxlID) = X(:,pxlID).*Weight(pxlID);
     end
     R = X*X'/N;
     w = inv(R+0.0001*eye(D)) * d' / (d*inv(R+0.0001*eye(D)) *d');
     y = w' * X;
     Weight = 1 - 2.71828.^(-lambda*y);
     Weight(Weight<0) = 0;     
     res = norm(y_old)^2/N - norm(y)^2/N;
     Energy = [Energy, norm(y)^2/N];
     y_old = y;     
     if (abs(res)<epsilon)
         break;
     end     
     outputs = reshape(mat2gray(y),[imgH,imgW]);
end
TR = im2bw(outputs,1e-16);
TE = im2bw(outputs,1e-16);
TE2d = hyperConvert2d(TE);
x = find(TE2d>0);
Pred_TE = zeros(size(TE2d));
Pred_TE = max(features', [], 1);
Pred_TE3d = hyperConvert3d(Pred_TE,row,col);
Pred_CM = Pred_TE3d;
Pred_CM = hyperNormalize(Pred_CM);
%% 生成对抗网络检测输出
num_set = [1:199];
[r,c,b] = size(data);
a_Result = zeros(200,2);
for i = 99%1:99
    test = load(strcat('.\Train\Results\Test_out\output_',num2str(num_set(i)),'.mat'));
    t_0 = struct2array(test);
    t = hyperNormalize(t_0);
    t = double(t);
    t_1=reshape(t,col,row,bands);
    t_3=permute(t_1,[2,1,3]);
    X = reshape(t_3,col*row,bands);
    X = X';
    [N M] = size(X);
    X_mean = mean(X,2);
    X = X - repmat(X_mean, [1 M]);
    Sigma = (X * X')/(M-1);
    Sigma_inv = inv(Sigma);
    D=zeros(1,M);
    for m = 1:M
        D(m) = X(:, m)' * Sigma_inv * X(:, m);
    end
    result = reshape(D,r,c);
    result = hyperNormalize(result);
%     a3_Result(i,1) = auc3;
%     a3_Result(i,2) = axujing3;
end
%% 背景抑制及融合
for k = 1:19%100
    for i = 1:row
        for j = 1:col
            omega(i,j) = 1-exp(-k*result(i,j));
        end
    end
    result_3 = omega.*result;
    output_result = hyperNormalize(result_3);
%     a3_Result(k,1) = auc3;
%     a3_Result(k,2) = axujing3;
end
% a3_maxresult = max(a3_Result);
% a3_pos = find(a3_Result==max(a3_Result));
for k = 1%1:100
    for i = 1:row
        for j = 1:col
            omega(i,j) = 1-exp(-k*initial_det(i,j));
        end
    end
    result_i = output_result.*omega;
%     a7_Result(k,1) = auc7;
%     a7_Result(k,2) = axujing7;
end
% auc7 = max(a7_Result);
result = A./(A+exp(-result_i.*Pred_CM));
figure,imagesc(result),axis image,axis off;
% save('.Results\result.mat','result');