function CTTF_TRS_XY = CTTF_parameter_tunning(x,r,X,Y)
% Parameter_tunning function is used to adjust some important parameters
% to optimize the objective value.

%% parameter setting
alpha = x.alpha;
beta = x.beta;

opts.max_iter = 500;
opts.alpha = alpha;
opts.beta = beta;
opts.pho = 0.001;
opts.r = r;
opts.k = 1.2;
opts.sigma = 1e-6;

% Execute ADMM algorithm
[A,V,~,S1,~,S2,~,~,~] = admm_remove_outlier(X,Y,opts);
L1 = cpdgen(A);
L2 = cpdgen(V);

% Reporting results
global CTTF_TRS_X
global CTTF_TRS_Y

CTTF_TRS_X = frob(X-L1-S1) / frob(X);  
CTTF_TRS_Y = frob(Y-L2-S2) / frob(Y);
CTTF_TRS_XY = CTTF_TRS_X + CTTF_TRS_Y;
% fprintf('CTTF_TRS_X =%.4f \n  CTTF_TRS_Y=%.4f  \n  CTTF_TRS_XY=%.4f  \n',...
%     TRS_X,TRS_Y,TRS_XY)



