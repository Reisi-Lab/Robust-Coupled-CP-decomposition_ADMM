function [X,Y] = generate_coupled_data(coupled_size,outlier_ratio,r,outlier_variance,noise_variance)
%generate_coupled_data function generates coupled high-order tensors and
% returns the generated data tensor X, Y.

%    [X,Y] = generate_coupled_data(coupled_size,outlier_ratio,r,outlier_variance,noise_variance) 
%    coupled_size: For example, if we want to generate a 3-order tensor of size 50 by 40
%    by 30 and a matrix of size 50 by 20 coupled in the first mode, then
%    coupled_size will be [50 40 30 20]. Another example: to generate a 50*40*30 tensor coupled with a 50*20*10 tensor
%    coupled in the first mode, then the coupled_size will be [50 40 30 20 10].

%    -- r is the tensor rank
%    -- outlier_ratio is the ratio of outliers in both tensor X and Y
%    -- outlier_variance is the variance of outliers in both tensor X and Y
%    -- noise_variance is the variance of small Gaussian noise in both tensor X and Y

%   X: generated tensor
%   Y: generated tensor

% rng(0);          % random seed
%%  Generate a higher-order tensor coupled with a matrix/higher-order tensor on their first mode
if length(coupled_size) == 4
    % generate coupled matrix and 3-order tensor
    sizeX = coupled_size(1:end-1);
    sizeY = [coupled_size(1),coupled_size(end)];
elseif length(coupled_size) == 5
    % generate coupled 3-order tensor and tensor
    sizeX = coupled_size(1:end-2);
    sizeY = [coupled_size(1),coupled_size(end-1:end)];
end

% Generate factor matrices
for i = 1:length(sizeX)
    U{i} = randn(sizeX(i),r);
end
% Generate tensor(tensorlab toolbox)
L1 = cpdgen(U);
% Generate a small Gaussian noisy tensor adding to tensor L1
epsilon1 = sqrt(noise_variance).*randn(sizeX);
% Generate a sparse outlier S1 with larger variance
S1 = sqrt(outlier_variance).*randn(sizeX); 
for i = 1:sizeX(end)
    % make S1 sparse
    S1(:,:,i) = S1(:,:,i).* double(sprand(sizeX(1),sizeX(2),outlier_ratio)~=0);
end
% Generated tensor X
X = L1 + epsilon1 + S1;

%% % Generate a tensor Y coupled the first mode with X
V = randn(sizeY(2),r);
L2 = U{1}*V';
% Generate a small Gaussian noisy tensor adding to tensor L2
epsilon2 = sqrt(noise_variance).*randn(sizeY);
% Generate a outlier S2 to L2
S2 = sqrt(outlier_variance).*randn(sizeY);
if length(coupled_size) == 4   
    S2 = S2.*double(sprand(sizeY(1),sizeY(2),outlier_ratio)~=0);
elseif length(coupled_size) == 5
    for i = 1:sizeY(end)
        S2(:,:,i) = S2(:,:,i).* double(sprand(sizeY(1),sizeY(2),outlier_ratio)~=0);
    end    
end  
% Generated tensor Y
Y = L2 + epsilon2 + S2;
end