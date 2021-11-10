function CMTF_TRS = compare_cmtf(X,Y,r)
%compare_cmtf function can simultaneously decompose coupled tensor and
% matrix and returns the result of TRS.

% [CMTF_TRS] = compare_cmtf(X,Y) 
%    input: 
%       X: generated tensor
%       Y: generated matrix 

%    output:
%       CMTF_TRS: is calculated by dividing the ||X - L^||^2 by ||X||^2

% Refered to Article on CMTF: 
%    E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled Matrix 
%   and Tensor Factorizations, KDD Workshop on Mining and Learning with Graphs, 2011 (arXiv:1105.3422v1)

% CMTF to simultaneously decompose coupled tensor and matrix 
sizeX = size(X);
sizeY = size(Y);
coupled_size = [sizeX,sizeY(end)];
modes = {[1 2 3],[1 4]};

T{1} = tensor(X)/norm(tensor(X));
T{2} = tensor(Y)/norm(tensor(Y));

P = length(T);
for p = 1:P
    Z.object{p} = T{p};
    Z.object{p} = Z.object{p}/norm(Z.object{p});
end
Z.modes = modes;
Z.size = coupled_size;

options = ncg('defaults');
options.MaxFuncEvals = 1000;
options.MaxIters = 500;
options.RelFuncTol = 1e-8;
options.StopTol = 1e-8;

Fac = cmtf_opt(Z,r,'alg','ncg','alg_options',options,'init','nvecs');
cmtf_X = cpdgen(Fac.U(1:3));
CMTF_TRS = frob(double(T{1}) - cmtf_X)/frob(double(T{1}));
end