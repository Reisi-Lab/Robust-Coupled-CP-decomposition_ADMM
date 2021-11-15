clear; close all; clc

addpath('D:\Matlab_Toolbox\tensor_toolbox');
addpath('D:\Matlab_Toolbox\tensorlab_2016-03-28');
%Main function extract EEG and fMRI features (lambda) as input of
%classification and returns the result of accuracy.

% Read EEG (tensor) and fMRI (matrix) data  
EEG_target = load('EEG_target_trials.mat');
EEG_target = EEG_target.EEG_target_allsubs;             
EEG_standard = load('EEG_standard_trials.mat');
EEG_standard = EEG_standard.EEG_standard_allsubs;           

fMRI_target = load('fMRI_target_trials.mat');
fMRI_target = fMRI_target.fMRI_target_allsubs;       
fMRI_standard = load('fMRI_standard_trials.mat');
fMRI_standard = fMRI_standard.fMRI_standard_allsubs;                     

% Optimize parameters 'alpha' and 'beta' for proposed method (RCCPD)
% Find one set of parameters used for all coupled tensors and matrices
rank  = 16; % rank = 4,8,10...
alpha = optimizableVariable('alpha',[0.1,0.9]);  
beta  = optimizableVariable('beta',[0.1,0.9]);     
   
fun = @(x)parameter_tunning(x,EEG_target(:,:,:,1),fMRI_target(:,:,1),rank);
results = bayesopt(fun,[alpha,beta],'IsObjectiveDeterministic',true,...
   'NumCoupledConstraints',0,'PlotFcn',{},'AcquisitionFunctionName',...
       'expected-improvement-plus','Verbose',0,'UseParallel',false,'MaxObjectiveEvaluations',30);
zbest = table2array(bestPoint(results));
RCCPD_TRS = results.MinObjective;                        
       
opts.max_iter = 300;
opts.alpha    = zbest(1);
opts.beta     = zbest(2);
opts.pho      = 0.01;
opts.rank     = rank;
opts.k        = 1.5;
opts.sigma    = 1e-6;

% K-fold is used to select train data for RCCPD
K = 3;  
num_sample = size(EEG_target,4); 
Label = ones(num_sample,1);
indices = crossvalind('Kfold',Label,K);  
size_fold = sum(indices == 1);

accuracy_list = zeros(1,K);
std_accuracy_list = zeros(1,K);
precision_list = zeros(1,K);
recall_list = zeros(1,K);
f1_list = zeros(1,K);

lam_EEG_target = zeros(num_sample - size_fold,rank);
lam_EEG_standard = zeros(num_sample - size_fold,rank);
lam_fMRI_target = zeros(num_sample - size_fold,rank);
lam_fMRI_standard = zeros(num_sample - size_fold,rank);

for k = 1:K
    % generate train data and test data
    test = (indices == k);  % size_fold
    train = ~test;
    base_data_T = EEG_target(:,:,:,test);
    base_data_Y = fMRI_target(:,:,test);
    
    rng default
    [A,V,~,~,~,~,~,~,~,~] = admm(base_data_T,base_data_Y,opts); % RCCPD
%     [A,V,~,~,~,~,~,~,~,~] = admm_remove_outliers(base_data_T,base_data_Y,opts); % CMTF/CTTF
    HA = khatrirao(A{3},A{2},A{1});
    HHA = (HA'*HA)\HA';  
    HV = khatrirao(V{2},V{1});
    HHV = (HV'*HV)\HV';

    for idx = 1:num_sample-size_fold
        EEG_target_train = EEG_target(:,:,:,train);   
        EEG_target_new = EEG_target_train(:,:,:,idx);
        lam_EEG_target(idx,:) = (HHA * EEG_target_new(:))'; 
        
        EEG_standard_new = EEG_standard(:,:,:,idx);
        lam_EEG_standard(idx,:) = (HHA * EEG_standard_new(:))'; 
        
        fMRI_target_train = fMRI_target(:,:,train);
        fMRI_target_new = fMRI_target_train(:,:,idx);
        lam_fMRI_target(idx,:) = (HHV * fMRI_target_new(:))'; 
        
        fMRI_standard_new = fMRI_standard(:,:,idx);
        lam_fMRI_standard(idx,:) = (HHV * fMRI_standard_new(:))'; 
    end  
    lambdas = [lam_EEG_target,lam_fMRI_target;lam_EEG_standard,lam_fMRI_standard]; 
    labels = [ones(num_sample-size_fold,1); 2*ones(num_sample-size_fold,1)];
    [accuracy_list(k),std_accuracy_list(k),precision_list(k),recall_list(k),f1_list(k)] = svm_classifier(lambdas,labels);
end
accuracy = mean(accuracy_list);
precision = mean(precision_list);
recall = mean(recall_list);
f1 = mean(f1_list);
std_accuracy = std(std_accuracy_list);

fprintf('Resullts of RCCPD on coupled tensor and matrix are: \n')
fprintf('accuracy=%.4f std_acc=%.4f  precision=%.4f  recall=%.4f  f1=%.4f \n',...
    accuracy, std_accuracy, precision, recall, f1)

% RCCPD:
%accuracy=0.9028 std_acc=0.0029  precision=0.8910  recall=0.9167  f1=0.9033 

% CMTF:
%accuracy=0.8190 std_acc=0.0095  precision=0.8365  recall=0.8053  f1=0.8123 
