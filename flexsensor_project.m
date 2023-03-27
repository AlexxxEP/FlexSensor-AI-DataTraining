clear all;
clc;
% -------------------- Data reception --------------------
Ve = readmatrix('measures/Book4.csv'); %Data2.csv is your sensing dataset
N = length(Ve)-1; % The number of samples
angle = Ve(1,:);  % The  angle
L = length(angle);% The number of angle
Vs_clean = zeros(N,L); % The data without the valor of angle
for i = 2:N
    Vs_clean(i,:) = Ve(i,:);
end
angleSample = zeros(N,L); % The sample of angle we study
for i=1:N
    angleSample(i,:) = angle;
end


% ------------------ Elimination of erroneous values ----------------

k = 1.96; % The factor for a confidence level at 95%
standarDev = std(Vs_clean);
confidence_interval = standarDev*k/sqrt(N); % 

real_value = mean(Vs_clean); 

Vs_process = Vs_clean; % Remove data that isn't in the confidence interval
for i=1:N
    for j=1:L
        if (Vs_process(i,j)<real_value(j)-k*confidence_interval(j))||(real_value(j)+k*confidence_interval(j)<Vs_process(i,j))
            Vs_process(i,j)=real_value(j);
        end
    end
end

% -------------------- Simulating data acquisition --------------------
 
 
standarDev = 0.01;
error = standarDev*randn(N,L); % The matrice of error for each angle
 
Vs_sim = zeros(N,L); % The acquisition simulated
for i=1:N
    Vs_sim(i,:) = real_value+error(i,:);
end
 
% -------------- Training the NN Model with Vs_clean --------------
 
H = 15;
net = feedforwardnet(H);
net = train(net,Vs_clean,angleSample);
anglePred = net(Vs_clean); % The output of angle with the data
anglePred_process = net(Vs_process); % The output of angle with the processed data
anglePred_sim = net(Vs_sim); % The output of angle with the simulated data
clear net;
 
figure(1)
subplot(3,1,1);plot(anglePred)
title("Training with the data")
xlabel('sample'); ylabel('Vs data')
subplot(3,1,2);plot(anglePred_process)
xlabel('sample'); ylabel('Vs data processed')
subplot(3,1,3);plot(anglePred_sim)
xlabel('sample'); ylabel('Vs data simulated')
 
% -------------- Training the NN Model with Vs_process --------------
 
net = feedforwardnet(H);
net = train(net,Vs_process,angleSample);
anglePred = net(Vs_clean); % The output of angle with the data
anglePred_process = net(Vs_process); % The output of angle with the processed data
anglePred_sim = net(Vs_sim); % The output of angle with the simulated data
clear net;
 
figure(2)
subplot(3,1,1);plot(anglePred)
title("Training with the data processed")
xlabel('sample'); ylabel('Vs data')
subplot(3,1,2);plot(anglePred_process)
xlabel('sample'); ylabel('Vs data processed')
subplot(3,1,3);plot(anglePred_sim)
xlabel('sample'); ylabel('Vs data simulated')
 
% -------------- Training the NN Model with Vs_sim --------------
 
net = feedforwardnet(H);
net = train(net,Vs_sim,angleSample);
anglePred = net(Vs_clean); % The output of angle with the data
anglePred_process = net(Vs_process); % The output of angle with the processed data
anglePred_sim = net(Vs_sim); % The output of angle with the simulated data
clear net;
 
figure(3)
subplot(3,1,1);plot(anglePred)
title("Training with the data simulated")
xlabel('sample'); ylabel('Vs data')
subplot(3,1,2);plot(anglePred_process)
xlabel('sample'); ylabel('Vs data processed')
subplot(3,1,3);plot(anglePred_sim)
xlabel('sample'); ylabel('Vs data simulated')
 
mean(anglePred)
std(anglePred)
mean(anglePred_process)
std(anglePred_process)
mean(anglePred_sim)
std(anglePred_sim)
