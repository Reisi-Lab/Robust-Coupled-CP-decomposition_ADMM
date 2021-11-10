function TRS_XY = parameter_tunning(x,r,X,Y)
% Parameter_tunning function is used to adjust some important parameters
% to optimize the objective value.

%% parameter setting
alpha = x.alpha;
beta = x.beta;
opts.alpha = alpha;
opts.beta = beta;
opts.max_iter = 300;

opts.pho = 0.001;
opts.r = r;
opts.k = 1.5;
opts.sigma = 1e-6;

% Execute ADMM algorithm
[A,V,~,S1_,~,S2_,~,~,~] = admm(X,Y,opts);
L1 = cpdgen(A);
L2 = cpdgen(V);

% Reporting results

%  我想记录这2个值TRS_X TRS_Y

global RCCPD_TRS_X
global RCCPD_TRS_Y
RCCPD_TRS_X = frob(X-L1-S1_) / frob(X);
RCCPD_TRS_Y = frob(Y-L2-S2_) / frob(Y);
TRS_XY = RCCPD_TRS_X + RCCPD_TRS_Y;
% fprintf('RCCPD_TRS_X = %.4f \n RCCPD_TRS_Y = %.4f \n  RCCPD_TRS_XY = %.4f \n',...
%     TRS_X,TRS_Y,TRS_XY)





