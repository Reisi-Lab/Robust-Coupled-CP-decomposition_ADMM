function [Objective_value] = lagrange_function(X,Y,A,V,F,S1,M,S2,D1,D2,opts)
%This is the corresponding Augmentaed Lagrngian function
%  [Objective_value,Outp1,Outp2] = lagrange_function(X,Y,A,V,F,S1,M,S2,D1,D2,opts)
%
%      Input:
%          X,Y,A,V,F,S1,M,S2,D1,D2,opts
%
%      Output:
%          Objective_value: The objective value of original function
%          Outp1 is the term5 + term6 in the objective function
%          Outp2 is the term7 + term8 in the objective function
%
%% Prepare the parameters
alpha  = opts.alpha;                   
beta   = opts.beta;       
pho    = opts.pho;      
%% Precompute fixed variables for the objective
normF = frob(tens2mat(F,1));
normM = frob(tens2mat(M,1));
L1 = cpdgen(A);
L2 = cpdgen(V);
%% Calculate each term in augmented Lagrangian
% Calculate term1
term1 = (1/2)*frob((X-S1-L1),'squared');
% Calculate term2
term2 = alpha*normF;  
% Calculate term3
term3 = (1/2)*frob((Y-S2-L2),'squared');
% Calculate term4
term4 = beta*normM;
% Calculate term5
term5 = trace(tens2mat(D1,1)'*tens2mat((F-S1),1));
% Calculate term6
term6 = (1/2)*pho*frob((F-S1),'squared');
% Calculate term7
term7 = trace(tens2mat(D2,1)'*tens2mat((M-S2),1));
% Calculate term8
term8 = (1/2)*pho*frob((M-S2),'squared');

% Calculate the objective function
Objective_value = term1 + term2 + term3 + term4;
% Outp1 = term5 + term6; 
% Outp2 = term7 + term8; 
end

