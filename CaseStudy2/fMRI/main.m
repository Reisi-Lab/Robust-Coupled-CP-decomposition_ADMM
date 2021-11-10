clear; close all; clc

addpath('D:\Matlab_Toolbox\tensor_toolbox');
addpath('D:\Matlab_Toolbox\tensorlab_2016-03-28');

%Main function split extracted fMRI ROI data (acquired by pre-processing) into target trials 
% and standard trials for classification. We use 16 subjects 3 runs of auditory task data to 
% construct fMRI matrices where the modes represent: subject * voxels. Under each subject, 
% there is the ROI data file, totally 197 voxels for each scan are extracted. For each run, 
% 20 samples are saved. Finally, 60 target and 60 standard trials are saved for classification.

%% fMRI trials split based on pre-processing ROI data
% number of subjects:16 (remove subject 14)
subjects = [01 02 03 05 06 07 08 09 10 11 12 13 14 15 16 17];

% variables initialization 
fMRI_roi_3runs = cell(1,3);
fMRI_label_3runs = cell(1,3);
target_label_3runs = cell(1,3);
standard_label_3runs = cell(1,3);
target_fMRI_data = cell(1,3);
standard_fMRI_data = cell(1,3);
fMRI_target_allsubs = zeros(16,197,60);
fMRI_standard_allsubs = zeros(16,197,60);

% read data of each run for each subject
idx = 0;
for sub = subjects   
    subject = num2str(sub, '%02d');
    for n_runs = 1:3
        n_run = num2str(n_runs, '%03d');
        m_run = num2str(n_runs, '%02d'); 
        % read fMRI ROI data and stimulus labels for each run 
        fMRI_roi = load(['D:\RCCPD_Code and Data\CaseStudy2\fMRI\sub-' subject '\run' n_run '\Mats_MNI_con4mats.mat']);
        fMRI_roi_3runs{n_runs} = cell2mat(struct2cell(fMRI_roi.Mats_MNI_con4mat)); 
        fMRI_label = load(['D:\RCCPD_Code and Data\CaseStudy2\fMRI\sub-' subject '\run' m_run '_label.mat']);
        fMRI_label_3runs{n_runs} = fMRI_label.labels;
        
        % Remove labels without stimulus -- label 0       
        num_no_stimulus = find(fMRI_label_3runs{n_runs}(:,1) == 0);  
        fMRI_label_3runs{n_runs}(num_no_stimulus,:) = []; 
        fMRI_roi_3runs{n_runs}(num_no_stimulus,:) = []; 
        
        % split label 1(target) and label 2 (standard)
        num_target = find(fMRI_label_3runs{n_runs}(:,1) == 1);
        target_label_3runs{n_runs} = fMRI_label_3runs{n_runs}(num_target(1:20),:);
        target_fMRI_data{n_runs} = fMRI_roi_3runs{n_runs}(num_target(1:20),:);
        num_standard = find(fMRI_label_3runs{n_runs}(:,1) == 2);
        standard_label_3runs{n_runs} = fMRI_label_3runs{n_runs}(num_standard(1:20),:);
        standard_fMRI_data{n_runs} = fMRI_roi_3runs{n_runs}(num_standard(1:20),:);
    end
    target_data = [target_fMRI_data{1};target_fMRI_data{2};target_fMRI_data{3}];  
    standard_data = [standard_fMRI_data{1};standard_fMRI_data{2};standard_fMRI_data{3}]; 
    
    idx = idx + 1;
    fMRI_target_allsubs(idx,:,:) = target_data';                 
    fMRI_standard_allsubs(idx,:,:) = standard_data';   
end 

save('fMRI_target_trials.mat','fMRI_target_allsubs');
save('fMRI_standard_trials.mat','fMRI_standard_allsubs'); 
save('D:\Code and Data\CaseStudy2\RCCPD_fMRI_EEG\fMRI_target_trials.mat','fMRI_target_allsubs');
save('D:\Code and Data\CaseStudy2\RCCPD_fMRI_EEG\fMRI_standard_trials.mat','fMRI_standard_allsubs');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call CPD_SVM function
K = 3;
[accuracy,std_accuracy,precision,recall,f1] = CPD_SVM(fMRI_target_allsubs, fMRI_standard_allsubs,K);
fprintf('Resullts of CPD_SVM function on fMRI tensors are: \n')
fprintf('accuracy=%.4f std_acc=%.4f  precision=%.4f  recall=%.4f  f1=%.4f \n',...
    accuracy, std_accuracy, precision, recall, f1)

% accuracy=0.6392 std_acc=0.0045  precision=0.7327  recall=0.6126  f1=0.6117
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call svm_classifier function
num_sample = size(fMRI_target_allsubs,3);
Label = ones(num_sample,1);
indices = crossvalind('Kfold',Label,K);
size_fold = sum(indices == 1);

data = cat(3,fMRI_target_allsubs(:,:,size_fold:num_sample), fMRI_standard_allsubs(:,:,size_fold:num_sample));  
[d1,d2,d3] = size(data);
data = (reshape(data,[d1*d2,d3]))'; 
label = [ones(d3/2,1); 2*ones(d3/2,1)]; 
[accuracy,std_accuracy,precision,recall,f1] = svm_classifier(data, label);
fprintf('Resullts of SVM classifier on fMRI tensors are: \n')
fprintf('accuracy=%.4f std_acc=%.4f  precision=%.4f  recall=%.4f  f1=%.4f \n',...
    accuracy, std_accuracy, precision, recall, f1)

% accuracy=0.7994 std_acc=0.0085  precision=0.8992  recall=0.6913  f1=0.7817
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%