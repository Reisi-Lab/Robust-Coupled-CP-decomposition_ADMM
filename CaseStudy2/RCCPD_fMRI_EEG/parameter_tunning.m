function TRS_T = parameter_tunning(x,X,Y,rank)
% Parameter_tunning function is used to adjust some important parameters
% to optimize the objective value.

%% parameter setting
alpha = x.alpha;
beta = x.beta;

opts.max_iter = 500;
opts.alpha = alpha;
opts.beta = beta;
opts.pho = 0.001;
opts.rank = rank;
opts.k = 1.2;
opts.sigma = 1e-6;

% Execute ADMM algorithm
[A,~,~,S1,~,~,~,~,~,~] = admm(X,Y,opts);
L1 = cpdgen(A);

% Reporting results
TRS_T = frob(X-L1-S1) / frob(X);


