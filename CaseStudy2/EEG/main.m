close all; clear; clc;

addpath('D:\Matlab_Toolbox\fieldtrip-master\preproc');
addpath('D:\Matlab_Toolbox\tensor_toolbox');
addpath('D:\Matlab_Toolbox\tensorlab_2016-03-28');

%Main function processes EEG signal first and then split EEG signal into target trials 
% and standard trials for classification. We use 16 subjects 3 runs of auditory task data
% to construct 3-order EEG tensors where the modes represent: subject * channel * time. 
%  
% A version of re-referenced EEG data with 37 channels sampled at 1000HZ are used. 
% Pre-processing steps include removing channels 35,36,37, down-sample(to 200HZ),
% (band-pass filter may be used) and so on. EEG tensor size is: 16 * 34 * 121. For each run, 20 samples 
% are saved. Finally, 60 target and 60 standard trials are saved for classification step. 
%
% References: 
%  - Walz JM, Goldman RI, Carapezza M, Muraskin J, Brown TR, Sajda P (2013) “Simultaneous 
%    EEG-fMRI Reveals Temporal Evolution of Coupling between Supramodal Cortical Attention 
%    Networks and the Brainstem,” J Neurosci 33(49):19212-22.doi: 10.1523/JNEUROSCI.2649-13.2013.
%  - Data source: https: //openneuro.org/datasets/ds000116/versions/00003.

%% EEG Pre-processing and trials split
%  original frequence is 1000 (samples/second)
original_freq = 1000;  
% down-sampled frequence is 200 (samples/second)
down_freq = 200; 
% time window starts 100 ms before the onset
beforeOnSet = 100/1000;   
% time window ends 500 ms after the onset
afterOnSet = 500/1000;  
% number of subjects:16 (remove subject 14)
subjects = [001 002 003 005 006 007 008 009 010 011 012 013 014 015 016 017]; 

% variables initialization 
EEG_3runs = cell(1,3);
target_trials_3runs = cell(1,3);
standard_trials_3runs = cell(1,3);
onset_target_3runs = cell(1,3);
onset_standard_3runs = cell(1,3);
target_subject = zeros(16,37,121,60);
standard_subject = zeros(16,37,121,60);

% read data of each run for each subject
idx = 0;
for sub = subjects
    subject = num2str(sub, '%03d');      
    idx = idx + 1;
    for n_runs = 1:3
        n_run = num2str(n_runs, '%03d');
        % read re-referenced version of EEG data for each run
        EEG_reref = load(['D:\RCCPD_Code and Data\CaseStudy2\EEG\sub' subject '\EEG\task001_run' n_run '\EEG_rereferenced.mat']);        
        data_reref = EEG_reref.data_reref;    % 37*340000
        % down-sample and band-pass filter
        resamped_data = ft_preproc_resample(data_reref,original_freq,down_freq,'resample'); % data: Nchans*Nsamples 37*68000 
        EEG_3runs{n_runs} = resamped_data;

        % read target onset and standard onset data for each run
        onset_target_3runs{n_runs} = readmatrix(['D:\RCCPD_Code and Data\CaseStudy2\EEG\sub' subject '\model\model001\onsets\task001_run' n_run '\cond001.txt']);
        onset_standard_3runs{n_runs} = readmatrix(['D:\RCCPD_Code and Data\CaseStudy2\EEG\sub' subject '\model\model001\onsets\task001_run' n_run '\cond002.txt']);

        % split all target trials from three runs
        onsetTarget = onset_target_3runs{n_runs};
        num_onset_target = length(onsetTarget);
        % time points list for stimulus
        timeStampOnSet = zeros(1,num_onset_target); 
        % time window list before stimulus
        timeStampBeforeOnSet = zeros(1,num_onset_target);  
        % time window list after stimulus
        timeStampAfterOnSet = zeros(1,num_onset_target);           
        indexOnSet = zeros(1,num_onset_target);
        indexBeforeOnSet = zeros(1,num_onset_target);
        indexAfterOnSet = zeros(1,num_onset_target);
        
        for i = 1:num_onset_target
            timeStampOnSet(i) = onsetTarget(i,1);
            timeStampBeforeOnSet(i) = onsetTarget(i,1) - beforeOnSet;  
            timeStampAfterOnSet(i) = onsetTarget(i,1) + afterOnSet;     
            indexOnSet(i) = int64(down_freq*onsetTarget(i,1));
            indexBeforeOnSet(i) = int64(down_freq*(onsetTarget(i,1) - beforeOnSet));
            indexAfterOnSet(i) = int64(down_freq*(onsetTarget(i,1) + afterOnSet));
        end
        % number of target trials of each run
        num_trials_target = size(indexOnSet,2);  
        
        target_trial_1run = zeros(37,121,num_trials_target);
        for k = 1:num_trials_target         
            target_trial_1run(:,:,k) = EEG_3runs{n_runs}(:,indexBeforeOnSet(k):indexAfterOnSet(k));
        end
        % for each run, record the first 20 target trials
        target_trials_3runs{n_runs} = target_trial_1run(:,:,1:20);
        
        % split all standard trails from three runs
        onsetStandard = onset_standard_3runs{n_runs};
        num_onset_standard = length(onsetStandard);
        % time points list for stimulus
        timeStampOnSet = zeros(1,num_onset_target); 
        % time window list before stimulus
        timeStampBeforeOnSet = zeros(1,num_onset_target);
        % time window list after stimulus
        timeStampAfterOnSet = zeros(1,num_onset_target);           
        indexOnSet = zeros(1,num_onset_target);
        indexBeforeOnSet = zeros(1,num_onset_target);
        indexAfterOnSet = zeros(1,num_onset_target);
        
        for i = 1:length(onsetStandard)
            timeStampOnSet(i) = onsetStandard(i,1);
            timeStampBeforeOnSet(i) = onsetStandard(i,1)-beforeOnSet;  
            timeStampAfterOnSet(i) = onsetStandard(i,1)+afterOnSet;    
            indexOnSet(i) = int64(down_freq*onsetStandard(i,1));
            indexBeforeOnSet(i) = int64(down_freq*(onsetStandard(i,1)-beforeOnSet));
            indexAfterOnSet(i) = int64(down_freq*(onsetStandard(i,1)+afterOnSet));
        end
        
        num_trials_standard = size(indexOnSet,2);
        standard_trial_1run = zeros(37,121,num_trials_standard);
        for k = 1:num_trials_standard
            standard_trial_1run(:,:,k) = EEG_3runs{n_runs}(:,indexBeforeOnSet(k):indexAfterOnSet(k));
        end
        % for each run, record the first 20 standard trials
        standard_trials_3runs{n_runs} =  standard_trial_1run(:,:,1:20);       
    end
    target_subject(idx,:,:,:) = cat(3,target_trials_3runs{1},target_trials_3runs{2},target_trials_3runs{3});
    standard_subject(idx,:,:,:) = cat(3,standard_trials_3runs{1},standard_trials_3runs{2},standard_trials_3runs{3});   
end
% save only 1-34 channels of data for classification 
EEG_target_allsubs = target_subject(:,1:34,:,:);        % 16 34 121 60
EEG_standard_allsubs = standard_subject(:,1:34,:,:);    % 16 34 121 60

save('EEG_target_trials.mat','EEG_target_allsubs')       
save('EEG_standard_trials.mat', 'EEG_standard_allsubs')  
save('D:\RCCPD_Code and Data\CaseStudy2\EEG\EEG_target_trials.mat','EEG_target_allsubs')       
save('D:\RCCPD_Code and Data\CaseStudy2\EEG\EEG_standard_trials.mat', 'EEG_standard_allsubs')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call CPD_SVM function
K = 3;
[accuracy,std_accuracy,precision,recall,f1] = CPD_SVM(EEG_target_allsubs, EEG_standard_allsubs,K);

fprintf('Resullts of CPD_SVM function on EEG tensors are: \n' )
fprintf('accuracy=%.4f std_acc=%.4f  precision=%.4f  recall=%.4f  f1=%.4f \n',...
    accuracy, std_accuracy, precision, recall, f1)

% accuracy=0.7354 std_acc=0.0064  precision=0.7364  recall=0.7484  f1=0.7401
% accuracy=0.7358 std_acc=0.0045  precision=0.7354  recall=0.7492  f1=0.7403
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call svm_classifier function
num_sample = size(EEG_target_allsubs,4);
Label = ones(num_sample,1);
indices = crossvalind('Kfold',Label,K);
size_fold = sum(indices == 1);

data = cat(4,EEG_target_allsubs(:,:,:,size_fold:num_sample), EEG_standard_allsubs(:,:,:,size_fold:num_sample)); 
[d1,d2,d3,d4] = size(data);
data = (reshape(data,[d1*d2*d3,d4]))'; 
label = [ones(d4/2,1); 2*ones(d4/2,1)]; 

[accuracy,std_accuracy,precision,recall,f1] = svm_classifier(data,label);
fprintf('Resullts of SVM classifier on EEG tensors are:  \n')
fprintf('accuracy=%.4f std_acc=%.4f  precision=%.4f  recall=%.4f  f1=%.4f \n',...
    accuracy, std_accuracy, precision, recall, f1)

% accuracy=0.8398 std_acc=0.0028  precision=0.8460  recall=0.8457  f1=0.8458
% accuracy=0.8396 std_acc=0.0040  precision=0.8459  recall=0.8452  f1=0.8455
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%