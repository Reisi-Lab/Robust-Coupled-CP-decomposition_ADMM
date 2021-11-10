function [A,V,F,S1,M,S2,D1,D2,lambda1,lambda2] = admm_remove_outliers(X,Y,opts)
%Solve a robust coupled tensor CP decomposition problem via ADMM
%   [A,V,F,S1,M,S2,D1,D2,lambda1,lambda2] = ADMM(X,Y,opts)
%   Input:
%       X        -    I1*I2*...*IN  tensor  
%       Y        -    I1*J2*...*JM  tensor    
%       opts     -    structure value in Matlab, the fields are
%           opts.max_iter    -    maximum number of iterations
%           opts.alpha       -    >0, a scalar parameter
%           opts.lambda      -    >0, a scalar parameter
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
%% Set parameters from input or by using defaults 
max_iter = opts.max_iter;
alpha    = opts.alpha;   
beta     = opts.beta;   
pho      = opts.pho;
rank     = opts.rank; 
k        = opts.k;  
sigma    = opts.sigma;         

%% Extract number of dimensions of tensor 
dimX = size(X); 
dimY = size(Y); 
nX = length(dimX);
mY = length(dimY);
A = cell(nX,1);
V = cell(mY,1);

%% Initialization
% Initialization of A
for i = 1:nX
    A{i} = normrnd(0,1,[dimX(i),rank]); 
end
% Initialization of V
for idx = 1:mY
    V{idx} = normrnd(0,1,[dimY(idx),rank]); 
end
V{1} = A{1};
% Initialization of F,S1,y1,M,S2,y2
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
        H1 = H1.*(V{idx}'*V{idx});          % calculate similarsign A(-1)
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
        
        Khat_A_N = khatrirao(A{[nX:-1:i+1,i-1:-1:1]}); 
        if i == 1    
            Khat_V_M = khatrirao(V{[mY:-1:2]});  
            t1 = W{1} + H1;                     
            A{i} = ((tens2mat(X,i) - tens2mat(S1,i))*Khat_A_N + (tens2mat(Y,i) - tens2mat(S2,i))*Khat_V_M)*(t1'/(t1*t1'));
            num_col_A = size(A{i},2);
            for col = 1:num_col_A
               lambda1 = sqrt(sum(A{i}(:,col).^2,1))';
               A{i}(:,col) = A{i}(:,col)/lambda1;
            end
        else   
            A{i} = (tens2mat(X,i) - tens2mat(S1,i))*Khat_A_N*(W{i}'/(W{i}*W{i}'));
            num_col_A = size(A{i},2);
            for col = 1:num_col_A
                lambda1 = sqrt(sum(A{i}(:,col).^2,1))';
                A{i}(:,col) = A{i}(:,col)/lambda1;
            end           
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
          
        if idx == 1 
            V{idx} = A{1};
        else  
            Khat_V_M = khatrirao(V{[mY:-1:idx+1,idx-1:-1:1]});
            V{idx} = (tens2mat(Y,idx) - tens2mat(S2,idx))*Khat_V_M*(H{idx}'/(H{idx}*H{idx}'));
            num_col_V = size(V{idx},2);
            for col = 1:num_col_V
                lambda2 = sqrt(sum(V{idx}(:,col).^2,1))';
                V{idx}(:,col) = V{idx}(:,col)/lambda2;
            end
        end
    end

    % Update F
    F = soft_thresholding(alpha/pho,S1 - D1/pho);
    % Update S1
    L1 = cpdgen(A);
    S1 = zeros(dimX);
    % Update M
    M = soft_thresholding(beta/pho,S2 - D2/pho);
    % Update S2
    L2 = cpdgen(V);
    S2 = zeros(dimY);
    % Update D1
    D1 = D1 + pho*(F - S1); 
    % Update D2
    D2 = D2 + pho*(M - S2);
    % Inceasing mu
    pho = pho*k; 
    
    % Result reporting and termination checks
    %[history.Objective_value(iter),history.Outp1(iter),history.Outp2(iter)] = lagrange_function(X,Y,A,V,F,S1,M,S2,D1,D2,opts);
    history.normXLS(iter) = frob(X-L1-S1);
    history.normX(iter) = frob(X);
    history.kktX(iter) = history.normXLS(iter)/history.normX(iter);
    
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






