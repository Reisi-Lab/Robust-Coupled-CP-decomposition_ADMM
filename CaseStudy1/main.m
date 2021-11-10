clear; close all; clc

addpath('D:\Matlab_Toolbox\tensor_toolbox');
addpath('D:\Matlab_Toolbox\tensorlab_2016-03-28');
addpath('D:\Matlab_Toolbox\CMTF_Toolbox_v1_1');
addpath('D:\Matlab_Toolbox\CMTF_Toolbox_v1_1\poblano_toolbox-main');
%% This is the code for Case study 1
%  load real data
data = load('surrogate_engin_data.mat');   
L1 = reshape(cell2mat(data.x_new),[200,203,5]);  
L2 = data.y_new; 
r = 5;

outlier_variance = [400,60,100,500,100,0.001];
outlier_ratio = 0.01; 

% Generate sparse outlier S1 to X
sizeL = size(L1(:,:,1));
S1 = zeros(200,203,5);
for i = 1:5
    S1(:,:,i) = sqrt(outlier_variance(i)).*randn(sizeL);
    S1(:,:,i) = S1(:,:,i).* double(sprand(sizeL(1),sizeL(2),outlier_ratio)~=0);
end
X = L1 + S1;
% Generate outlier S2 to Y
sizeY = size(L2);
S2 = sqrt(outlier_variance(6)).*randn(sizeY);
S2 = S2.*double(sprand(sizeY(1),sizeY(2),outlier_ratio)~=0);
Y = L2 + S2;

%% Plot the noisy tensor and matrix data
figure(1)
y = {'Actual air mass (mg/s)','Engine rotational speed(rpm)','Injection quantity (mg/s)',...
    'Boost pressure actual value (mbar)','Inner torque (Nm)', 'Lambda value US NCS'};
x = {'Tensor slice X1','Tensor slice X2','Tensor slice X3','Tensor slice X4','Tensor slice X5','The matrix Y'};
min = [400, 1550, 0, 100, 140, 0.85];
max = [700, 1590, 100, 400, 220, 1];

num_example = 30;
for i = 1:5
    plotX = X(:,:,i);
    for j = 1:num_example
        subplot(2,3,i)
        plot(plotX(j,:));
        title(x{i})
        xlabel('Time(s)')
        ylabel(y{i})
        ylim([min(i) max(i)]);
        xticklabels({'0','1','2'});
        hold on
    end
end
subplot(2,3,6)
plot(Y(1:num_example,:)')
xticklabels({'0','1','2'});
xlabel('Time(s)')
ylim([min(6) max(6)]);
ylabel(y{6})
title(x{6})

%% Execute the proposed method and benchmarks
alpha = optimizableVariable('alpha',[0.01,0.05]);
beta = optimizableVariable('beta',[0.001,0.05]);

fun = @(x)parameter_tunning(x,r,X,Y);
results = bayesopt(fun,[alpha,beta],'IsObjectiveDeterministic',true,...
    'NumCoupledConstraints',0,'PlotFcn',{},...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',1,...
    'UseParallel',false,'MaxObjectiveEvaluations',1);
%     'UseParallel',true,...
% @plotMinObjective,@plotObjective,@plotObjectiveModel,@plotAcquisitionFunction
zbest = bestPoint(results);
RCCPD_TRS = results.MinObjective;  
% Compare with CP decomposition
CPD_X = cpdgen(cpd(X,r));
CP_TRS = frob(X - CPD_X) / frob(X);
% disp('CP_TRS = ');
% disp(CP_TRS)
% Compare with CMTF
CMTF_TRS = compare_cmtf(X,Y,r);
% disp('CMTF_TRS = ')
% disp(CMTF_TRS)
% 
