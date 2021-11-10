function [accuracy,std_accuracy,precision,recall,f1] = svm_classifier(data,label)
%svm_classifier is a two-class classification solver in our work used to classify 
%the target trials (label 1) and standad trials (label 2).
% [accuracy,standard_deviation,precision,sensitivity,f1] = svm_classifier(Data, Label)
% Input:
%    data           - input data for classification, Nsample*Ndimensions
%    label          - data lables, Nsample*label
%
% Output:
%    accuracy       - the accuracy result is (TP+TN)/(TP+FP+TN+FN)
%    std_accuracy   - the standard deviation of accuracy on repeat time
%    precision      - the precision is TP/(TP+FP)
%    recall         - the recall is TP/(TP+FN)
%    f1             - the f1 score is 2*(precision*recall)/(precision+recall)

K = 200;
accuracy_list = zeros(1,K);
precision_list = zeros(1,K);
recall_list = zeros(1,K);
f1_list = zeros(1,K);

for k = 1:K
    size_target = size(data,1);
    num_train = round(0.7*size_target);
    % randomly split train data and test data
    n = randperm(size_target);
    train_data = data(n(1:num_train),:);
    train_label = label(n(1:num_train),:);
    test_data = data(n(num_train+1:end),:);
    test_label = label(n(num_train+1:end),:);
    
    % fit a SVM model
    rng default
    SVMModel = fitcsvm(train_data,train_label,'Standardize',true,'KernelFunction','RBF',...
        'KernelScale','auto');
    % predict by the SVM model
    predict_label = predict(SVMModel,test_data);
    % calculate the confusion matrix (TP,FP; TN,FN)
    TP = sum((predict_label == 1) & ( test_label == 1));
    FP = sum((predict_label == 1) & ( test_label == 2));
    TN = sum((predict_label == 2) & ( test_label == 2));
    FN = sum((predict_label == 2) & ( test_label == 1));
     
    accuracy_list(k) = (TP+TN)/(TP+FP+TN+FN);
    precision_list(k) = TP/(TP+FP);
    recall_list(k) = TP/(TP+FN);
    f1_list(k) = 2*precision_list(k)*recall_list(k)/(precision_list(k)+recall_list(k));
end

% report the average results
accuracy = mean(accuracy_list);
precision = mean(precision_list);
recall = mean(recall_list);
f1 = mean(f1_list);
std_accuracy = std(accuracy_list);

end