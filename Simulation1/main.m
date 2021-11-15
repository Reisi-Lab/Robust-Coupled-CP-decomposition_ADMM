clear; close all; clc

addpath('D:\Matlab_Toolbox\tensorlab_2016-03-28');
addpath('D:\Matlab_Toolbox\tensor_toolbox\tensor_toolbox');
addpath('D:\Matlab_Toolbox\CMTF_Toolbox_v1_1');
addpath('D:\Matlab_Toolbox\CMTF_Toolbox_v1_1\poblano_toolbox-main');

%% Main function: return TRSs of the proposed method and compared methods
% By running main function, average TRSs with their standard deviation for 
% three methods (RCCPD, CP,CMTF) are obtained.

coupled_size = [10 10 10 20];     % coupled tensor and matrix size
outlier_ratio = 0.05;             % set the outlier ratio as 5%, 10%, 20% respectively
r = 4;                            % tensor rank

K = 50;                           % number of experiments:30
i = 0; 
for outlier_variance = [1 4 9]  
    i = i + 1;
    j = 0;
    for noise_variance = [0.04 0.16 0.25 0.36]
        j = j + 1;
        for k = 1:K
            [X,Y] = generate_coupled_data(coupled_size,outlier_ratio,r,outlier_variance,noise_variance);
            
            % optimize parameters 'alpha' and 'beta' for the proposed RCCPD method
            alpha = optimizableVariable('alpha',[0.1,1]);
            beta = optimizableVariable('beta',[0.1,1]);
            fun = @(x)parameter_tunning(x,r,X,Y);
            results = bayesopt(fun,[alpha,beta],'IsObjectiveDeterministic',true,...
                'NumCoupledConstraints',0,'PlotFcn',{},'AcquisitionFunctionName',...
                'expected-improvement-plus','Verbose',0,'UseParallel',false);
            zbest = bestPoint(results);
            % Results of the proposed RCCPD
            RCCPD_TRS(i,j,k) = results.MinObjective;  
            
            % Compare with CPD
            CPD_X = cpdgen(cpd(X,r));
            CPD_TRS(i,j,k) = frob(X - CPD_X)/frob(X); 
             
            % Compare with CMTF 
            CMTF_TRS(i,j,k) = compare_cmtf(X,Y,r);
            
        end        
    end
end

ave_RCCPD_TRS = sum(RCCPD_TRS(:,:,(1:K)),3)/K;
std_RCCPD_TRS = std(RCCPD_TRS,0,3);

ave_CMTF_TRS = sum(CMTF_TRS(:,:,(1:K)),3)/K;
std_CMTF_TRS = std(CMTF_TRS,0,3);

ave_CPD_TRS = sum(CPD_TRS(:,:,(1:K)),3)/K;
std_CPD_TRS = std(CPD_TRS,0,3);

% Results saving
save('ave_RCCPD_TRS_5.mat','ave_RCCPD_TRS')
save('ave_CMTF_TRS_5.mat','ave_CMTF_TRS')
save('ave_CPD_TRS_5.mat','ave_CPD_TRS')

save('std_RCCPD_TRS_5.mat','std_RCCPD_TRS')
save('std_CPD_TRS_5.mat','std_CPD_TRS')
save('std_CMTF_TRS_5.mat','std_CMTF_TRS')