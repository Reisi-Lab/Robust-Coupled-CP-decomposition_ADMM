%% Plot one target and standard trial sample
close all; clear; clc

% standard
target_EEG_trial = load('EEG_target_trials.mat');       % 16 34 121 60
target_EEG_trial = target_EEG_trial.EEG_target_allsubs;      
standard_EEG_trial = load('EEG_standard_trials.mat');   % 16 34 121 60
standard_EEG_trial = standard_EEG_trial.EEG_standard_allsubs;
s = reshape(standard_EEG_trial(1,:,:,1),[34,121]);
t = reshape(target_EEG_trial(1,:,:,1),[34,121]);

figure('color',[1 1 1]);
subplot(2,1,1);
x = 1:121;
plot(x,s);
title('Standard Stimulus')
% ylabel(ax1,'s')
% xlabel(ax1,'Time Point')
line([20 20],[-30 30],'Color','r','LineWidth',3);
xlim([1,121])
% ylim([-1,1])
text(16,-85,'onset')

subplot(2,1,2);
plot(x,t);
title('Target Stimulus')
% ylabel(ax2,'t')
% xlabel(ax2,'Time Point')
xlim([1,121])
line([20 20],[-60 60],'Color','r','LineWidth',3);
text(16,-200,'onset')
% ylim([-1,1])