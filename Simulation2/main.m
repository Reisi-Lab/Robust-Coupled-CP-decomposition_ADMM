clear; close all; clc

addpath('D:\Matlab_Toolbox\tensor_toolbox');
addpath('D:\Matlab_Toolbox\tensorlab_2016-03-28');
addpath('D:\Matlab_Toolbox\CMTF_Toolbox_v1_1');
addpath('D:\Matlab_Toolbox\CMTF_Toolbox_v1_1\poblano_toolbox-main');
%% This is the main function: return the TRSs of the proposed method and compared methods
%  The compared method are CTTF (implenmented via setting updates of S as zeros) and 
%  CPD (decompose two tensors seperately)  
%% Coupled size and parameters setting
global RCCPD_TRS_X 
global RCCPD_TRS_Y
global CTTF_TRS_X
global CTTF_TRS_Y

coupled_size = [10 20 30 20 10];    % tensor [10 20 30] and [10 20 10]                          
outlier_ratio = 0.1;                % set 5% 10% 20% respectively
r = 4;

K = 50;                             % repeat times of experiments
i = 0; 
for outlier_variance = [1 4 9]    
    i = i + 1;
    j = 0;
    for noise_variance = [0.04 0.16 0.25 0.36]   
        j = j + 1;
        for k = 1:K
            [X,Y] = generate_coupled_data(coupled_size,outlier_ratio,r,outlier_variance,noise_variance);
            % optimize parameters 'alpha' and 'beta' for the proposed method (RCCPD)
            alpha = optimizableVariable('alpha',[0.1,1]);  
            beta = optimizableVariable('beta',[0.1,1]);     
            fun = @(x)parameter_tunning(x,r,X,Y);
            results = bayesopt(fun,[alpha,beta],'IsObjectiveDeterministic',true,...
                'NumCoupledConstraints',0,'PlotFcn',{},'AcquisitionFunctionName',...
                'expected-improvement-plus','Verbose',0,'UseParallel',false,'MaxObjectiveEvaluations',1); %,'MaxObjectiveEvaluations',5
            
            zbest = bestPoint(results);
            RCCPD1_TRS(i,j,k) = RCCPD_TRS_X;
            RCCPD2_TRS(i,j,k) = RCCPD_TRS_Y;
%             RCCPD_TRS(i,j,k) = results.MinObjective;   
            
            % optimize parameters 'alpha' and 'beta' for CTTF
            alpha = optimizableVariable('alpha',[0.1,1]);  
            beta = optimizableVariable('beta',[0.1,1]);        
            fun = @(x)CTTF_parameter_tunning(x,r,X,Y);
            results = bayesopt(fun,[alpha,beta],'IsObjectiveDeterministic',true,...
                'NumCoupledConstraints',0,'PlotFcn',{},'AcquisitionFunctionName',...
                'expected-improvement-plus','Verbose',0,'UseParallel',false,'MaxObjectiveEvaluations',1);
            zbest = bestPoint(results);
            CTTF1_TRS(i,j,k) = CTTF_TRS_X;
            CTTF2_TRS(i,j,k) = CTTF_TRS_Y;
%             CTTF_TRS(i,j,k) = results.MinObjective;  
            %  Compare with CP decomposition for Tensor X
            CPD1_TRS(i,j,k) = frob(X - cpdgen(cpd(X,r)))/frob(X);              
            %  Compare with CP decomposition for Tensor Y
            CPD2_TRS(i,j,k) = frob(Y - cpdgen(cpd(Y,r)))/frob(Y);            
        end 
    end
end

Ave_RCCPD1_TRS = sum(RCCPD1_TRS(:,:,(1:K)),3)/K;
std_RCCPD1_TRS = std(RCCPD1_TRS,0,3);
Ave_RCCPD2_TRS = sum(RCCPD2_TRS(:,:,(1:K)),3)/K;
std_RCCPD2_TRS = std(RCCPD2_TRS,0,3);

Ave_CTTF1_TRS = sum(CTTF1_TRS(:,:,(1:K)),3)/K;
std_CTTF1_TRS = std(CTTF1_TRS,0,3);
Ave_CTTF2_TRS = sum(CTTF2_TRS(:,:,(1:K)),3)/K;
std_CTTF2_TRS = std(CTTF2_TRS,0,3);

Ave_CPD1_TRS = sum(CPD1_TRS(:,:,(1:K)),3)/K;
std_CPD1_TRS = std(CPD1_TRS,0,3);
Ave_CPD2_TRS = sum(CPD2_TRS(:,:,(1:K)),3)/K;
std_CPD2_TRS = std(CPD2_TRS,0,3);

% std_CPD_TRS = (std_CPD1_TRS + std_CPD2_TRS)/2; 
% Ave_CPD_TRS = Ave_CPD1_TRS + Ave_CPD2_TRS;


save(['Ave_RCCPD1_TRS_',num2str(outlier_ratio), '.mat'],'Ave_RCCPD1_TRS')
save(['std_RCCPD1_TRS_',num2str(outlier_ratio), '.mat'],'std_RCCPD1_TRS')
save(['Ave_RCCPD2_TRS_',num2str(outlier_ratio), '.mat'],'Ave_RCCPD2_TRS')
save(['std_RCCPD2_TRS_',num2str(outlier_ratio), '.mat'],'std_RCCPD2_TRS')

save(['Ave_CMTF1_TRS_',num2str(outlier_ratio), '.mat'],'Ave_CTTF1_TRS')
save(['std_CMTF1_TRS_',num2str(outlier_ratio), '.mat'],'std_CTTF1_TRS')
save(['Ave_CMTF2_TRS_',num2str(outlier_ratio), '.mat'],'Ave_CTTF2_TRS')
save(['std_CMTF2_TRS_',num2str(outlier_ratio), '.mat'],'std_CTTF2_TRS')

save(['Ave_CPD1_TRS_',num2str(outlier_ratio), '.mat'],'Ave_CPD1_TRS')
save(['std_CPD1_TRS_',num2str(outlier_ratio), '.mat'],'std_CPD1_TRS')
save(['Ave_CPD2_TRS_',num2str(outlier_ratio), '.mat'],'Ave_CPD2_TRS')
save(['std_CPD2_TRS_',num2str(outlier_ratio), '.mat'],'std_CPD2_TRS')


