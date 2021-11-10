function [cmtf_TRS] = compare_cmtf(T,Y,r)
%compare_cmtf function can simultaneously decompose two coupled tensors 
% and returns the result of decomposing score TRS.

% [cmtf_FIT] = compare_cmtf(T,Y) 
%    input: 
%       T: input tensor
%       Y: input matrix coupled with tensor T

%    output:
%       cmtf_TRS: is the test score calculated by dividing the ||T - T^||^2
%       by ||T||^2

% Refered to Article on CMTF: 
%    E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled Matrix 
%   and Tensor Factorizations, KDD Workshop on Mining and Learning with Graphs, 2011 (arXiv:1105.3422v1)

%% % CMTF to simultaneously decompose coupled tensors 
T_size = size(T);
Y_size = size(Y);
coupled_size = [T_size,Y_size(end)];
modes = {[1 2 3],[1 4]};

X{1} = tensor(T)/norm(tensor(T));
X{2} = tensor(Y)/norm(tensor(Y));

P = length(X);
for p = 1:P
    Z.object{p} = X{p};
    Z.object{p} = Z.object{p}/norm(Z.object{p});
end
Z.modes = modes;
Z.size = coupled_size;

options = ncg('defaults');
options.MaxFuncEvals = 1000;
options.MaxIters = 100;
options.RelFuncTol = 1e-8;
options.StopTol = 1e-8;

[Fac,~,~] = cmtf_opt(Z,r,'alg','ncg','alg_options',options,'init','nvecs');
cmtf_X = cpdgen(Fac.U(1:3));
cmtf_TRS = frob(double(X{1}) - cmtf_X)/frob(double(X{1}));

end