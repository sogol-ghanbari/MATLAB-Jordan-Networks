% Jordan Recurrent Neural Network (JRNN) Implementation
% Application: Non-linear System Identification
clc; close all; clear all;

%% Data Generation
N = 1003;
u = rand(1, N) - 0.5;  
yp = zeros(1, N);

for k = 3:N-1
    x1 = yp(k);
    x2 = yp(k-1);
    x3 = yp(k-2);
    x4 = u(k);
    x5 = u(k-1);
    yp(k+1) = (x1*x2*x3*x5*(x3-1) + x4) / (1 + x2^2 + x3^3);
end

x = [yp(3:1002); yp(2:1001); yp(1:1000); u(3:1002); u(2:1001)];
y = yp(4:1003);

%% Network Setup
ns = 700;  % Number of training samples
nt = 200;  % Number of testing samples
nc = 100;  % Number of checking samples

ni = 5;    % Input layer neurons
no = 1;    % Output layer neurons
nh1 = 20;  % Hidden layer neurons

epochNum = 100;
eta = 0.01; % Learning rate

mseTrain = [];
Tr = []; Te = []; Tc = [];

D = y(1:ns);
p = x(:, 1:ns);
pt = x(:, ns+1:ns+nt);
Dt = y(ns+1:ns+nt);
pc = x(:, ns+nt+1:end);
Dc = y(ns+nt+1:end);

w1 = rand(nh1, ni) * 2 - 1;
w2 = rand(no, nh1) * 2 - 1;
wr = rand(nh1, no) * 2 - 1;  % Jordan feedback: output -> hidden
b1 = zeros(nh1, 1);
b2 = zeros(no, 1);

%% Training Loop
for epoch = 1:epochNum
    error = [];
    
    % Reset context for Jordan network
    outputPrev  = zeros(no, 1);   % o^2(t-1)
    outputPrev2 = zeros(no, 1);   % o^2(t-2)
    df1Prev     = zeros(nh1, 1);  % f'(t-1)
    w2Prev      = zeros(no, nh1); % Weight at t-1
    
    for i = 1:ns
        % Forward pass
        out0 = p(:, i);
        net1 = w1*out0 + wr*outputPrev + b1; 
        out1 = tansig(net1);
        net2 = w2*out1 + b2;
        out2 = purelin(net2);
        Y(i) = out2;
        
        w2_t = w2;   
        
        e = D(i) - out2;
        error = [error, e];
        
        % Backpropagation
        % Output Layer
        d_f2 = 1; 
        S2 = d_f2 * e;
        
        % Hidden Layer
        d_f1 = 1 - out1.^2; 
        S1 = d_f1 .* (w2' * S2);
        
        % Update Weights
        deltaw2 = eta * S2 * out1';
        w2 = w2 + deltaw2;
        deltabias2 = eta * S2;
        b2 = b2 + deltabias2;
        
        deltaw1 = eta * S1 * out0';
        w1 = w1 + deltaw1;
        deltabias1 = eta * S1;
        b1 = b1 + deltabias1;
        
        % Jordan Recurrent Weight Update
        term1 = outputPrev;
        term2 = wr .* (df1Prev .* (w2Prev' * outputPrev2));    
        dnet1_dwr = term1' + term2;                              
        deltawr = eta * S1 .* dnet1_dwr;
        wr = wr + deltawr;
        
        % Update Context
        outputPrev2 = outputPrev;
        outputPrev  = out2;
        df1Prev     = d_f1;
        w2Prev      = w2_t;   
    end
    
    mseTrain = [mseTrain, mse(error)];
    
    %% Test phase
    testError = [];
    Yt = [];
    outputPrevTest = zeros(no, 1); 
    
    for i = 1:length(pt)
        out0 = pt(:, i);
        net1 = w1*out0 + wr*outputPrevTest + b1;
        out1 = tansig(net1);
        net2 = w2*out1 + b2;
        out2 = purelin(net2);
        testError = [testError, Dt(i) - out2];
        Yt = [Yt, out2];
        outputPrevTest = out2;
    end
    mseTest(epoch) = mse(testError);
    
    %% Plotting
    subplot(2,1,1)
    plot(mseTrain)
    title('MSE error for Training')
    subplot(2,1,2)
    plot(mseTest)
    title('MSE error for Testing')
    drawnow
end

%% Final Results
mse_final = mse(error)
test_err_final = mse(testError)

%% Check phase
c_error = [];
Yc = [];
outputPrevCheck = zeros(no, 1);

for i = 1:length(pc)
    out0 = pc(:, i);
    net1 = w1*out0 + wr*outputPrevCheck + b1;
    out1 = tansig(net1);
    net2 = w2*out1 + b2;
    out2 = purelin(net2);
    c_error = [c_error, Dc(i) - out2];
    Yc = [Yc, out2];
    outputPrevCheck = out2;
end
check_error = mse(c_error)

figure; plot(Y); hold on; plot(D,'r'); title('Training Output vs Target'); legend('Output', 'Target');
figure; plot(Yt); hold on; plot(Dt,'r'); title('Testing Output vs Target'); legend('Output', 'Target');
figure; plot(Yc); hold on; plot(Dc,'r'); title('Checking Output vs Target'); legend('Output', 'Target');
