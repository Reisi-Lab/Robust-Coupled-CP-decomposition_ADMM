% close all; clear; clc

% parameter setting
size = [10 10 10 20];
outlier_variance = [1,4,9];
noise_variance = [0.04,0.16,0.25,0.36];

% read data

RCCPD = load('ave_RCCPD_TRS_5.mat');
RCCPD = RCCPD.ave_RCCPD_TRS;
CMTF = load('ave_CMTF_TRS_5.mat');
CMTF = CMTF.ave_CMTF_TRS;
CPD = load('ave_CPD_TRS_5.mat');
CPD = CPD.ave_CPD_TRS;

title_list = {'\sigma_1^2 = 1,  q = 5%','\sigma_1^2 = 4,  q = 5%','\sigma_1^2 = 9,  q = 5%'};
figure
for i = 1:3
    subplot(3,1,i)
    plot(noise_variance,RCCPD(i,:),'-pr','linewidth',1); hold on
    plot(noise_variance,CMTF(i,:),'-dg','linewidth',1); hold on
    plot(noise_variance,CPD(i,:),'-vb','linewidth',1)
    title(title_list{i},'fontsize',12)
    ylim([0.1 0.7])
    ylabel('TRS','fontsize',12)
    xticks([0.04 0.16 0.25 0.36])
    xlim([0.02 0.38])
end
legend({'RCCPD','CMTF','CPD'},'Location','northeast')


