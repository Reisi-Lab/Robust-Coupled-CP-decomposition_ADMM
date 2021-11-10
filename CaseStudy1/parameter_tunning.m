function TRS_RCCPD = parameter_tunning(x,r,X,Y)
%Parameter_tunning function is used to select parameters
% to optimize the objective value.

%% parameter setting
alpha = x.alpha;
beta = x.beta;
opts.alpha = alpha;
opts.beta = beta;

opts.max_iter = 100;
opts.r = r;
opts.k = 1.2;
opts.sigma = 1e-6;
opts.pho = 0.0001;

% Execute ADMM algorithm
[A,V,~,S1_,~,~,~,~,~] = admm(X,Y,opts);
L1 = cpdgen(A);
L2 = cpdgen(V);

%% Plot the reconstructed tensor and matrix data
figure(2)
y = {'Actual air mass (mg/s)','Engine rotational speed(rpm)','Injection quantity (mg/s)',...
    'Boost pressure actual value (mbar)','Inner torque (Nm)', 'Lambda value US NCS'};
x = {'Tensor slice X1','Tensor slice X2','Tensor slice X3','Tensor slice X4',...
    'Tensor slice X5','The matrix Y'};
min = [400, 1550, 0, 100, 140, 0.85];
max = [700, 1590, 60, 400, 220, 1];

num_example = 30;
for i = 1:5
    plotX = L1(:,:,i);
    for j = 1:num_example
        subplot(2,3,i)
        plot(plotX(j,:));
        xlabel('Time(s)')
        title(x{i})
        ylabel(y{i})
        ylim([min(i) max(i)]);
        xticklabels({'0','1','2'});
        
        hold on
    end
end

subplot(2,3,6)
plot(L2(1:num_example,:)')
xticklabels({'0','1','2'});
xlabel('Time(s)')
ylim([min(6) max(6)]);
ylabel(y{6})
title(x{6})

% Reproting results
TRS_RCCPD = frob(X-L1-S1_) / frob(X); 
disp('TRS_RCCPD = ')
disp(TRS_RCCPD)
