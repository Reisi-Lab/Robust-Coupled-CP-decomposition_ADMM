function [A,V,F,S1,M,S2,D1,D2,history] = admm(X,Y,opts)
%Solve a robust coupled tensor CP decomposition problem via ADMM
%   [A,V,F,S1,M,S2,D1,D2,history] = admm(X,Y,opts)
%   Input:
%       X        -    I1*I2*...*IN  tensor  
%       Y        -    I1*J2*...*JM  tensor    
%       opts     -    structure value in Matlab, the fields are
%           opts.max_iter    -    maximum number of iterations
%           opts.alpha       -    >0, a scalar parameter
%           opts.beta        -    >0, a scalar parameter
%           opts.pho         -    >0, a augmented term parameter
%           opts.k           -    >1, scaling parameter for increasing mu
%           opts.r           -    scalar, the rank of tensor X
%           opts.sigma       -    a stopping condition parameter
%
%   Output:
%       A        -    the factor matrices of tensor X
%       V        -    the factor matrices of tensor Y
%       S1       -    the noisy tensor added to tensor X
%       S2       -    the noisy tensor added to tensor Y
%       F        -    the substitude of S1
%       M        -    the substitude of S2
%       D1       -    the lagrangian multiplier, a tensor
%       D2       -    the lagrangian multiplier, a tensor
%   history is a structure that contains the objective values and
%   the number of iteration and so on.

% QUIET   = 1;
%% Set parameters 
max_iter = opts.max_iter;
alpha    = opts.alpha;                     
beta     = opts.beta;   
pho      = opts.pho;
r        = opts.r;        
k        = opts.k;  
sigma    = opts.sigma;         

%% Extract number of dimensions of tensor X
dimX = size(X); 
dimY = size(Y); 
nX = length(dimX);
mY = length(dimY);
A = cell(nX,1);
V = cell(mY,1);

%% Initialization
% Initialization of A
for i = 1:nX
    A{i} = normrnd(0,1,[dimX(i),r]); 
end
% Initialization of V
for idx = 1:mY
    V{idx} = normrnd(0,1,[dimY(idx),r]); 
end
V{1} = A{1};
% Initialization of F,S1,D1,M,S2,D2
F  = zeros(dimX);  
S1 = F;        
D1 = S1;                          
M = zeros(dimY);                     
S2 = M;                             
D2 = S2;   

%% ADMM solver for robust coupled CPD
for iter = 1:max_iter    
    % Update A and V
    H1 = 1;
    for idx = 2:mY
        H1 = H1.*(V{idx}'*V{idx});          
    end
    
    for i = 1:nX
        if i==1
            W{i}=1; 
        else
            W{i}= A{1}'*A{1};
        end
        for j = 2:nX
            if j ~= i
                W{i} = W{i}.*(A{j}'*A{j});  
            end
        end 
        t1 = W{1} + H1;                      % calculate similar sign T(-1) + A(-1)
        Khat_A_N = khatrirao(A{[nX:-1:i+1,i-1:-1:1]}); % calculate (odot A(-n))
        Khat_V_M = khatrirao(V{[mY:-1:2]});  % calculate (odot V(-1))
        if i == 1 
            A{i} = ((tens2mat(X,i) - tens2mat(S1,i))*Khat_A_N + (tens2mat(Y,i) - tens2mat(S2,i))*Khat_V_M)*(t1'/(t1*t1'));
        else    
            A{i} = (tens2mat(X,i) - tens2mat(S1,i))*Khat_A_N*(W{i}'/(W{i}*W{i}')); 
        end   
    end
   
    % Update V   
    for idx = 1:mY
        if idx == 1
            H{idx} = 1;
        else
            H{idx} = A{1}'*A{1};
        end
        for index = 2:mY
            if index ~= idx
                H{idx} = H{idx}.*(V{index}'*V{index});
            end
        end     
        
        Khat_V_M = khatrirao(V{[mY:-1:idx+1,idx-1:-1:1]});  
        if idx == 1
            V{idx} = A{1};
        else  
            V{idx} = (tens2mat(Y,idx) - tens2mat(S2,idx))*Khat_V_M*(H{idx}'/(H{idx}*H{idx}'));
        end
    end

    % Update F
    F = soft_thresholding(alpha/pho,S1 - D1/pho);
    % Update S1
    L1 = cpdgen(A);
    S1 = (X - L1 + D1 + pho*F)/(1+pho); 
    % Update M
    M = soft_thresholding(beta/pho,S2 - D2/pho);
    % Update S2
    L2 = cpdgen(V);
    S2 = (Y - L2 + D2 + pho*M)/(1 + pho); 
    % Update D1
    D1 = D1 + pho*(F - S1); 
    % Update D2
    D2 = D2 + pho*(M - S2);
    % Inceasing mu
    pho = pho*k;  
    % Result reporting and termination checks
    [history.Obj_val(iter),history.Outp1(iter),history.Outp2(iter)] = lagrange_function(X,Y,A,V,F,S1,M,S2,D1,D2,opts);
    history.normXLS(iter) = frob(X-L1-S1);
    history.normX(iter) = frob(X);
    history.kktX(iter) = history.normXLS(iter) / history.normX(iter);
    
    history.normYLS(iter) = frob(Y-L2-S2);
    history.normY(iter) = frob(Y);
    history.kktY(iter) = history.normYLS(iter)/history.normY(iter);
    
%     if ~QUIET
%         fprintf('%3d\t   %5.4f\t   %5.4f\t  %5.2f\t   %5.2f\t   %5.2f\n', iter, ...
%             history.Objective_value(iter),history.kktX(iter),history.kktY(iter),history.Outp1(iter),history.Outp2(iter));
%     end

    if (history.kktX(iter) < sigma) && (history.kktY(iter)< sigma)
        break;
    end
     
end
end






