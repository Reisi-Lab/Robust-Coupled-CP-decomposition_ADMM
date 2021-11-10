function [accuracy,std_accuracy,precision,recall,f1] = CPD_SVM(target_allsubs, standard_allsubs, K)
% CPD_SVM classify trials based on lambda values extracted from CP decomposition. This function 
% first use parts of target data as CPD trianing and then use the obtained factor matrices on the 
% rest target data and same number of standard data to calculate new inputs for SVM classifier.
% [accuracy,std_accuracy,precision,recall,f1] = CPD_SVM(target_allsubs, standard_allsubs, K)
% Input:
%    target_allsubs     - type: 4-tensor, the 4th order is number of samples
%    standard_allsubs   - type: 4-tensor, the 4th order is number of samples
%    K                  - K-fold, the number of fold
%
% Output:
%    accuracy       - the accuracy result is (TP+TN)/(TP+FP+TN+FN)
%    std_accuracy   - the standard deviation of accuracy on repeat time
%    precision      - the precision is TP/(TP+FP)
%    recall         - the recall is TP/(TP+FN)
%    f1             - the f1 score is 2*(precision*recall)/(precision+recall)


% Select tensor rank for CPD
rank = 16;  %rank = 4,8,10,...
num_sample = size(target_allsubs,4); 
Label = ones(num_sample,1);
indices = crossvalind('Kfold',Label,K);  
size_fold = sum(indices == 1);

lambda_EEG_target = zeros(num_sample-size_fold, rank);
lambda_EEG_standard = zeros(num_sample-size_fold, rank);

accuracy_list = zeros(1,K);
std_accuracy_list = zeros(1,K);
precision_list = zeros(1,K);
recall_list = zeros(1,K);
f1_list = zeros(1,K);

for k = 1:K
    % generate train data and test data
    test = (indices == k);  % size_fold
    train = ~test;
    base_data = tensor(target_allsubs(:,:,:,test));
    rng default
    cpdecom = parafac_als(base_data,rank);
    H = khatrirao(cpdecom.U{3},cpdecom.U{2},cpdecom.U{1});
    HH = (H'*H)\H';

    for idx = 1:num_sample-size_fold
        target_train = target_allsubs(:,:,:,train);   
        target_new = target_train(:,:,:,idx);
        lambda_EEG_target(idx,:) = (HH * target_new(:))'; 
        
        standard_train = standard_allsubs(:,:,:,train);
        standard_new = standard_train(:,:,:,idx);
        lambda_EEG_standard(idx,:) = (HH * standard_new(:))';        
    end 
    
    lambdas = [lambda_EEG_target;lambda_EEG_standard]; 
    labels = [ones(num_sample-size_fold,1); 2*ones(num_sample-size_fold,1)];
    [accuracy_list(k),std_accuracy_list(k),precision_list(k),recall_list(k),f1_list(k)] = svm_classifier(lambdas, labels);
end
accuracy = mean(accuracy_list);
precision = mean(precision_list);
recall = mean(recall_list);
f1 = mean(f1_list);
std_accuracy = std(std_accuracy_list);
end
    

    

