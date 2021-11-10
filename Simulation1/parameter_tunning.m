function TRS_X = parameter_tunning(x,r,X,Y)
% Parameter_tunning function is to tune some important parameters
% to optimize the objective value.

%% parameter setting
alpha = x.alpha;
beta = x.beta;
opts.alpha = alpha;
opts.beta = beta;

opts.max_iter = 300;
opts.pho = 0.001;
opts.r = r;
opts.k = 1.2;
opts.sigma = 1e-6;

% Execute ADMM algorithm
[A,~,~,S1,~,~,~,~,~] = admm(X,Y,opts);
L1 = cpdgen(A);

% Reproting results
TRS_X = frob(X-L1-S1) / frob(X);



