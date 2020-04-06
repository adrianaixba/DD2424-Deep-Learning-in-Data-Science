% input sizes:
% x: dx1
% p: Kx1
% W_1: mxd
% W_2: Kxm
% b_1: mx1
% b_2: Kx1

% train and test k-layer network
addpath Datasets/cifar-10-matlab/;

% exercise 1: compute gradients, d = 10, lambda = 0, start w 2-layer, then
% 3-layer, then 4-layer

% [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
% [ValX, ValY, Valy] = LoadBatch('data_batch_2.mat');
% [testX, testY, testy] = LoadBatch('test_batch.mat');
% m = [10];
% fprintf("relative error: 2 layers\n");
% ExerciseOne(2, m, trainX, trainY, trainy);
% m = [10, 10];
% fprintf("relative error: 3 layers\n");
% ExerciseOne(3, m, trainX, trainY, trainy);
% m = [10, 10, 10];
% fprintf("relative error: 4 layers\n");
% ExerciseOne(4, m, trainX, trainY, trainy);

%ExerciseTwo();
% val_size_limit = 5000;
layers = 2;
m =[50,50];
train_data = 'data_batch_1.mat';
val_data = 'data_batch_2.mat';
test_data = 'test_batch.mat';

% load data
[X_train, Y_train, y_train] = LoadBatch(train_data);
[X_val, Y_val, y_val] = LoadBatch(val_data);
[X_test, Y_test, y_test] = LoadBatch(test_data);
%BatchNormTest(layers, m, X_train, Y_train, y_train, X_test, y_test);

% fprintf("Starting sig test with %f\n", 1e-1);
% SigTests(1e-1);
% fprintf("Starting sig test with %f\n", 1e-3);
% SigTests(1e-3);
fprintf("Starting sig test with %f\n", 1e-4);
SigTests(1e-4);



function [X, Y, y] = LoadBatch(filename)
    batch = load(filename);
    X = double(batch.data);
    X = transpose(X);    
    y = double(batch.labels) + 1;
    
    % create one hot encoding
    N = size(X, 2);
    for i = 1 : N
        Y(y(i), i) = 1;
    end
    
    % normalize data
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
    
end

% initialize parameters
% m = an array that contains number of nodes in each hidden layer
% K = number of samples
% d = dimensionality of image
function [W, b] = InitializeParameters(K, d, m, k)
    rng(400);
    W = {};
    b = {};
    % from lecture 4
    % Gaussian distribution w mean 0, std 1/sqrt(d)
    W{1} = randn(m(1),d) * 1.0/sqrt(d);
    b{1} = randn(m(1),1) * 1.0/sqrt(d);
    i = 2;
    while i <= size(m,2)
        W{i} = 1.0/sqrt(m(i-1)) * randn(m(i),m(i-1)); 
        b{i} = 1.0/sqrt(m(i-1)) * randn(m(i),1);
        i = i + 1;
    end
    W{i} = 1.0/sqrt(m(end)) * randn(K,m(end));
    b{i} = 1.0/sqrt(m(end)) * randn(K,1);
end

function acc = ComputeAccuracy(X, y, W, b)
    [p, ~] = EvaluateClassifier(X, W, b);
    num_correct = 0;
    num_img = size(X, 2);
    [argval, argmax] = max(p);
    for img = 1:num_img
        [~, argmaxy] = max(y(:,img));
        if argmax(img) == argmaxy
            num_correct = num_correct + 1;
        end         
    end
    acc = double(num_correct)/num_img * 100;
end

function acc = ComputeAccuracyBN(X, Y, NetParams)
    P = EvaluateClassifierBN(X, NetParams);
    num_correct = 0;
    num_img = size(X, 2);
    [argval, argmax] = max(P{end,6});
    for img = 1:num_img
        [~, argmaxy] = max(Y(:,img));
        if argmax(img) == argmaxy
            num_correct = num_correct + 1;
        end         
    end
    acc = double(num_correct)/num_img * 100;
end

function [P, H] = EvaluateClassifier(X, W, b)
    s = {};
    h = {};
    
    s{1} = W{1}*X + b{1}*ones(1, size(X, 2));
    h{1} = max(s{1}, 0);
    for i=2:numel(W)
        % finding s, s = Wx+b
        s{i} = W{i}*h{i-1} + b{i}*ones(1, size(X, 2));

        % need the last s, but not the last activation
        if i ~= numel(W)
            % Activation
            h{i} = max(s{i}, 0);
        end
    end
    % do softmax(s) = exp(s)/((1^T)(exp(s)))
    exponential = exp(s{end});
    % 1 means the 1st dimension (AKA rows)
    D = ones(size(exponential, 1), 1)*sum(exponential, 1);
    P = exponential./D;
    H = h;
end


function [J, loss] = ComputeCost(X, Y, NetParams, lambda)
    W = NetParams.W;
    b = NetParams.b;
    [p, ~] = EvaluateClassifier(X, W, b);
    loss_func = -log(sum(Y .* p,1));

    % want to iterate through all the W values 
    reg_temp = 0.0;
    for i=1:length(W)
        reg_temp = reg_temp + sum(sum(W{i} .* W{i},'double'),'double');
    end
   
    % regularization term, summation of W^2 multiplied with lambda
    loss = mean(loss_func);
    J = loss + lambda*reg_temp;
end

% function that computes the gradients of the cost function for a
% mini-batch of data given the values computed from the forward pass
function [dl_dW, dl_db] = ComputeGradients(X, Y, P, NetParams, lambda, h)
    W = NetParams.W;
    N = size(X, 2);
    g_batch = -(Y - P);
   
    dl_dW = {};
    dl_db = {};
    for i=numel(W): -1:2
        dl_dWi = (1/N)*g_batch*transpose(h{i-1}) + 2*lambda*W{i};
        dl_dbi = (1/N)*g_batch*ones(size(X, 2), 1);
        g_batch = transpose(W{i})*g_batch;
        g_batch = g_batch .* (h{i-1} > 0);
        dl_dW{i} = dl_dWi;
        dl_db{i} = dl_dbi;
    end
    dl_dW1 = (1/N)*g_batch*transpose(X) + 2*lambda*W{1};
    dl_db1 = (1/N)*g_batch*ones(size(X, 2), 1);
    dl_dW{1} = dl_dW1;
    dl_db{1} = dl_db1;
       
end

function [grad_b, grad_W, grad_gam, grad_beta] = ComputeGradsNumSlow (X, Y, NetParams, lambda, h)
    Grads.W = cell(numel(NetParams.W), 1);
    Grads.b = cell(numel(NetParams.b), 1);
    if NetParams.use_bn
        Grads.gammas = cell(numel(NetParams.gammas), 1);
        Grads.betas = cell(numel(NetParams.betas), 1);
    end

    for j=1:length(NetParams.b)
        Grads.b{j} = zeros(size(NetParams.b{j}));
        NetTry = NetParams;
        for i=1:length(NetParams.b{j})
            b_try = NetParams.b;
            b_try{j}(i) = b_try{j}(i) - h;
            NetTry.b = b_try;
            c1 = ComputeCost(X, Y, NetTry, lambda);        

            b_try = NetParams.b;
            b_try{j}(i) = b_try{j}(i) + h;
            NetTry.b = b_try;        
            c2 = ComputeCost(X, Y,  NetTry, lambda);

            Grads.b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(NetParams.W)
        Grads.W{j} = zeros(size(NetParams.W{j}));
            NetTry = NetParams;
        for i=1:numel(NetParams.W{j})

            W_try = NetParams.W;
            W_try{j}(i) = W_try{j}(i) - h;
            NetTry.W = W_try;        
            c1 = ComputeCost(X, Y,  NetTry, lambda);

            W_try = NetParams.W;
            W_try{j}(i) = W_try{j}(i) + h;
            NetTry.W = W_try;        
            c2 = ComputeCost(X, Y,  NetTry, lambda);

            Grads.W{j}(i) = (c2-c1) / (2*h);
        end
    end

    if NetParams.use_bn
        for j=1:length(NetParams.gammas)
            Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
            NetTry = NetParams;
            for i=1:numel(NetParams.gammas{j})

                gammas_try = NetParams.gammas;
                gammas_try{j}(i) = gammas_try{j}(i) - h;
                NetTry.gammas = gammas_try;        
                c1 = ComputeCost(X, Y,  NetTry, lambda);

                gammas_try = NetParams.gammas;
                gammas_try{j}(i) = gammas_try{j}(i) + h;
                NetTry.gammas = gammas_try;        
                c2 = ComputeCost(X, Y,  NetTry, lambda);

                Grads.gammas{j}(i) = (c2-c1) / (2*h);
            end
        end

        for j=1:length(NetParams.betas)
            Grads.betas{j} = zeros(size(NetParams.betas{j}));
            NetTry = NetParams;
            for i=1:numel(NetParams.betas{j})

                betas_try = NetParams.betas;
                betas_try{j}(i) = betas_try{j}(i) - h;
                NetTry.betas = betas_try;        
                c1 = ComputeCost(X, Y,  NetTry, lambda);

                betas_try = NetParams.betas;
                betas_try{j}(i) = betas_try{j}(i) + h;
                NetTry.betas = betas_try;        
                c2 = ComputeCost(X, Y,  NetTry, lambda);

                Grads.betas{j}(i) = (c2-c1) / (2*h);
            end
        end   
    end
    grad_W = Grads.W;
    grad_b = Grads.b;
    grad_gam = Grads.gammas;
    grad_beta = Grads.betas;
end

% train network with cyclical learning rates
function [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, NetParams, lambda)
    W = NetParams.W;
    b = NetParams.b;
    n_batches = GDparams.n_batches;
    n_s = GDparams.n_s;
    cycles = GDparams.cycles;
    
    plot_idx = 1;
    
    eta_min = 1e-5;
    eta_max = 1e-1;
    step = (eta_max - eta_min)/n_s;
    tot_update_step = 2*cycles*n_s;
    eta = eta_min;
    
    N = size(X_train, 2);
    
    % at the start, eta is increasing by the step
    update = step; 
    batch_start = 1;
        
    % cost and accuracy matrices
%     costs = zeros(tot_update_step/100, 2);
%     accs = zeros(tot_update_step/100, 2);
    losses = zeros(tot_update_step/100, 2);
    
    % do all epocs, runs through all imgs 
    for t=1:tot_update_step
        % "Sample a batch of training data"  
         if batch_start >= N
            batch_start = 1;
        end
        %get indexes of the batch data
        idx = batch_start : min(batch_start + n_batches -1, N);
        
        Xbatch = X_train(:, idx);
        Ybatch =  Y_train(:, idx);
        
        %update starting index
        batch_start = batch_start + n_batches;
        
                        
        % complete forward pass - "calculate loss/cost"
        [Pbatch, H_batch] = EvaluateClassifier(Xbatch, W, b);
            
        % complete backwards pass - "to calculate gradients"
        [gradW, gradb] = ComputeGradients(Xbatch, Ybatch, Pbatch, NetParams, lambda, H_batch);
             
        % "update the parameters using the gradient"
        for i=1:numel(W)
            W{i} = W{i} - eta*gradW{i};
            b{i} = b{i} - eta*gradb{i};
        end     
        % Need to update NetParams
        NetParams.W = W;
        NetParams.b = b;
        
        % save the cost and accuracy after each epoch
        if mod(t,100)==0
            [cost_train, loss_train] = ComputeCost(X_train, Y_train, NetParams, lambda);
            [cost_val, loss_val] = ComputeCost(XVal, YVal, NetParams, lambda);
%             
%             costs(plot_idx, 1) = cost_train;
%             costs(plot_idx, 2) = cost_val;
%             
            losses(plot_idx, 1) = loss_train;
            losses(plot_idx, 2) = loss_val;
%           
%             accs(plot_idx, 1) = ComputeAccuracy(X_train, Y_train, W, b);
%             accs(plot_idx, 2) = ComputeAccuracy(XVal, YVal, W, b);
%             
            plot_idx = plot_idx + 1;
        end

        % update eta
        eta = eta + update;
        if eta >= eta_max
            % when we reach max, need to linearly decrease eta
            % if we go over max, we must start at eta_max, not an eta >
            % eta_max
            eta = eta_max;
            update = -step;
        elseif eta <= eta_min
            % when we reach min, need to linearly increase eta
            eta = eta_min;
            update = step;
        end
    end
    % plotting costs and accuracies
    x = 1 : plot_idx - 1;
%     plot(x*100, costs(:, 1), x*100, costs(:, 2));
%     title('Cost plot');
%     xlabel('update step')
%     ylabel('cost')
%     figure();
    % losses plot
    plot(100*x, losses(:, 1), 100*x, losses(:, 2));
    title('Loss plot');
    xlabel('update step')
    ylabel('loss')
    figure();
    % the train accuracies are blue, the validation accuracies are red
%     plot(100*x, accs(:, 1), 100*x, accs(:, 2));
%     title('Accuracy plot');
%     xlabel('update step')
%     ylabel('accuracy')

    Wstar = W;
    bstar = b;
end

% initializes parameters for batch norm
function [W, b, gammas, betas] = InitializeParametersBN(K, m, d)
    rng(400);
    W = {};
    b = {};
    gammas = {};
    betas = {};
    
    i = 1;
    while i <= size(m,2)
        %input check
        if i ==1
           W{i} = 1.0/sqrt(d) * randn(m(i),d);
           b{i} = 1.0/sqrt(d) * randn(m(i),1);
        else
           W{i} = 1.0/sqrt(m(i-1)) * randn(m(i),m(i-1)); 
           b{i} = 1.0/sqrt(m(i-1)) * randn(m(i),1);
        end
        gammas{i} = 1.0/sqrt(m(i)) * randn(m(i),1); 
        betas{i} = 1.0/sqrt(m(i)) * randn(m(i),1);
        i = i + 1;
    end
    W{i} = 1.0/sqrt(m(end)) * randn(K,m(end));
    b{i} = 1.0/sqrt(m(end)) * randn(K,1);
end

% evaluate classifier for batch norm
function P = EvaluateClassifierBN(X, NetParams, means, var)
    if nargin < 3
       P = no_comp(X, NetParams);
    else   
       P = with_comp(X, NetParams, means, var);
    end 
end

% evaluate classifier helper function
function P = no_comp(X, NetParams)
    % evaluate linear part
    eps = 1e-9;
    layers = numel(NetParams.W);
    meta = {};

    % 1 - x batch
    % 2 - s_batch
    % 3 - s_norm_batches
    % 4 - mean
    % 5 - variances
    % 6 - probabilities (softmax)
    
    for l = 1 : layers
        if l == 1
            s = NetParams.W{l} * X + NetParams.b{l} *  ones(1,size(X,2));
        else
            s = NetParams.W{l} * meta{l-1,1} + NetParams.b{l} *  ones(1,size(meta{l-1,3},2));
        end
        
        if l ~= layers
            % x_batches
            meta{l,2} = s;

            % mean
            meta{l,4} = mean(s,2);

            % variance
            meta{l,5} = (sum(((meta{l,2} - meta{l,4}).^2),2))./size(s,2);

            % now we need to normalize
            meta{l,3} = BatchNormalize(s, meta{l,4}, meta{l,5}, eps);
            
            % apply gamma and beta       
            s = (NetParams.gammas{l}* ones(1,size(s,2))) .* meta{l,3};
            s = s + NetParams.betas{l} *  ones(1,size(s,2));
                        
            %apply relu
            meta{l,1} = max(0,s);
        end
    end
    
    %last layer
    meta{layers,2} = s;
    % compute the softmax:
    % - numerators of the softmax:
    E = exp(s);
    % - denominators of the softmax:
    D = ones(size(E,1),1) * sum(E,1);
    
    meta{layers,6} = E./D;
    % Divide each column by their sum
    % to have the softmax
    P = meta;
end

%
% (l,1) -> Wx+b              s
% (l,2) -> batch_norm(Wx+b)  s_hat
% (l,3) -> ReLU(...)         X
% (l,4) -> means
% (l,5) -> variances
% (l,6) -> gamma s^ + beta   s_tilde

function s_norm = BatchNormalize(s, means, var, eps)
    %s_norm = s - mean * ones(1,size(s,2));
    s_norm = s - means;
    s_norm = s_norm ./ (sqrt((var + eps)* ones(1,size(s,2))));
end

% evaluate classifier helper function
function P = with_comp(X, NetParams, means, var)
    % evaluate linear part
    eps = 1e-9;
    layers = numel(NetParams.W);
    meta = {};
    
    for l = 1 : layers
        if l == 1
            s = NetParams.W{l} * X + NetParams.b{l} *  ones(1,size(X,2));
        else
            s = NetParams.W{l} * meta{l-1,1} + NetParams.b{l} *  ones(1,size(meta{l-1,3},2));
        end
        
        if l ~= layers
            meta{l,2} = s;
            % batch normalize
            meta{l,4} = means{l};
            
            meta{l,5} = var{l};
            meta{l,3} = BatchNormalize(s, meta{l,4},meta{l,5}, eps);

            % apply gamma and beta
            s = (NetParams.gammas{l}* ones(1,size(s,2))) .* s;
            s = s + NetParams.betas{l} *  ones(1,size(s,2));

            %apply relu
            meta{l,1} = max(0,s);
        end
    end
    
    %last layer
    meta{layers,2} = s;
    % compute the softmax:
    % - numerators of the softmax:
    E = exp(s);
    % - denominators of the softmax:
    D = ones(size(E,1),1) * sum(E,1);
    
    meta{layers,6} = E./D;
    % Divide each column by their sum
    % to have the softmax
    P = meta;
end



% compute gradients for batch norm
function [grad_W, grad_b, grad_gammas, grad_betas] = ComputeGradientsBN(X, Y, P, NetParams, lambda)
    % compute useful constant
    N = double(size(X,2));
    
    W = NetParams.W;
    b = NetParams.b;

    l = numel(W);
    grad_b = {};
    grad_W = {};
    grad_gammas = {};
    grad_betas = {};
    
    g_batch = -(Y-P{l,6});
    
    % gradients of J wrt bias vector b and W
    x_batch = P{l-1,1};
    grad_b{l} = 1.0/ N * g_batch * ones(size(g_batch, 2), 1);
    grad_W{l} = 1.0/ N * (g_batch * transpose(x_batch)) + 2 * lambda * W{l};

    % propogate g_batch to the previous layer
    g_batch = (transpose(W{l}) * g_batch);
    g_batch(x_batch==0) = 0;
    l = l - 1;

    while l >= 1
        
%         grad_gammas{l} = mean(g_batch .* P{l,3},2);
%         grad_betas{l} = mean(g_batch,2);
        grad_gammas{l} = 1/N * (g_batch .* P{l,3}) * ones(N, 1);
        grad_betas{l} = 1/N * g_batch * ones(N, 1);
        
        g_batch = g_batch .* (NetParams.gammas{l} * ones(1,N));
        
        % BatchNormBackPass
        g_batch = BatchNormBackPass(g_batch, P{l,2}, P{l,4}, P{l,5}, N);
        
        % end batch norm back pass
        if l == 1
            x_batch = X;
        else
            x_batch = P{l-1,1};
        end

        grad_b{l} = 1.0/ N * g_batch * ones(size(g_batch, 2), 1);
        grad_W{l} = 1.0/ N * (g_batch * transpose(x_batch)) + 2 * lambda * W{l};

        if l > 1
            %update gradient
            g_batch = (transpose(W{l}) * g_batch);
            g_batch(x_batch==0) = 0;
        end

        l = l - 1;
    end

end

function g_batch_norm = BatchNormBackPass(g_batch, s, mean_l, var, N)
    sig_1 = var.^(-0.5);
    sig_2 = var.^(-1.5);
    G1 = g_batch .* (sig_1 * ones(1,N));
    G2 = g_batch .* (sig_2 * ones(1, N));
    D = s - (mean_l * ones(1,N));
    c = (G2 .* D) * ones(N,1);
    g_batch_norm = G1 - (1/N) * (G1 * ones(N,1)) - (1/N) * (D .* (c * ones(1,N)));
end

function [J, loss] = ComputeCostBN(X, Y, NetParams, lambda, means, var)
    
    if nargin < 5
    % get the evaluation of the current parameters for the batch
        P = EvaluateClassifierBN(X, NetParams);
    else
        P = EvaluateClassifierBN(X, NetParams, means, var);
    end
   
    %compute the cross-entropy part
    loss = -mean(log(sum(Y .* P{end,6},1)));
    
    J2 = compute_regularization(NetParams.W,lambda);
    
    % add the regularizing term
    J =  loss + lambda*J2;
end

function J2 = compute_regularization(W, lambda)
    J2 = 0;

    if nargin < 2
        lambda = 1;
    end
    
    if lambda == 0
        return;
    end
    
    for k=1:length(W)
        Wi = W{k};
        J2 = J2 + sum(sum(Wi .* Wi,'double'),'double');
    end


end

function NetParamsStar = MiniBatchGDBN(X, Y, GDparams, NetParams, lambda, XVal, YVal, alpha)
    if nargin < 8
       alpha = 0.9;
    end
    
    batch_size = GDparams.n_batches;
    n_s = GDparams.n_s;
    cycles = GDparams.cycles;
    
    plot_idx = 1;
    
    eta_min = 1e-3;
    eta_max = 1e-1;
    step = (eta_max - eta_min)/n_s;
    tot_update_step = 2*cycles*n_s;
    eta = eta_min;
    
    N = size(X, 2);
    
    % at the start, eta is increasing by the step
    update = step; 
    batch_start = 1;
        
    % cost and accuracy matrices
%     costs = zeros(tot_update_step/100, 2);
%     accs = zeros(tot_update_step/100, 2);
    losses = zeros(tot_update_step/100, 2);
    
    means = {};
    var = {};
    
    for t = 1 : tot_update_step
        if batch_start >= N
            batch_start = 1;
        end
        %get indexes of the batch data
        idx = batch_start : min(batch_start + batch_size -1, N);

        % index the actual data
        X_batch = X(:,idx);
        Y_batch = Y(:,idx);

        %update starting index
        batch_start = batch_start + batch_size;
        
        % complete forward pass - "calculate loss/cost"
        P = EvaluateClassifierBN(X_batch, NetParams);

        % complete backwards pass - "to calculate gradients"
        [grad_W, grad_b,grad_gammas, grad_betas] = ComputeGradientsBN(X_batch, Y_batch, P, NetParams, lambda);
        
        % "update the parameters using the gradient"
        for k=1:length(NetParams.W)
            NetParams.W{k} = NetParams.W{k} - eta * grad_W{k};
            NetParams.b{k} = NetParams.b{k} - eta * grad_b{k};
            
            if k < length(NetParams.W)
                NetParams.gammas{k} = NetParams.gammas{k}- eta *grad_gammas{k} ;      
                NetParams.betas{k}= NetParams.betas{k} - eta * grad_betas{k};
            end
        end
        
        % update averages and variances
        if t == 1
            for j = 1 : size(P,1)
                means{j} = P{j,4};
                var{j} = P{j,5};    
            end
        else
            for j = 1 : size(P,1)
                means{j} = alpha * P{j,4}  + (1-alpha) * means{j};
                var{j} = alpha * P{j,5} + (1-alpha) * var{j};    
            end
        end 
        
        % save the cost and accuracy after each epoch
        if mod(t,100)==0
            [cost_train, loss_train] = ComputeCostBN(X, Y, NetParams, lambda);
            [cost_val, loss_val] = ComputeCostBN(XVal, YVal, NetParams, lambda);
%             
%              costs(plot_idx, 1) = cost_train;
%              costs(plot_idx, 2) = cost_val;
%             
            losses(plot_idx, 1) = loss_train;
            losses(plot_idx, 2) = loss_val;
%           
%              accs(plot_idx, 1) = ComputeAccuracyBN(X, Y, NetParams);
%              accs(plot_idx, 2) = ComputeAccuracyBN(XVal, YVal, NetParams);
%             
            plot_idx = plot_idx + 1;
        end
        
        
        % update eta
        eta = eta + update;
        if eta >= eta_max
            % when we reach max, need to linearly decrease eta
            % if we go over max, we must start at eta_max, not an eta >
            % eta_max
            eta = eta_max;
            update = -step;
        elseif eta <= eta_min
            % when we reach min, need to linearly increase eta
            eta = eta_min;
            update = step;
        end
        
    end
    
% plotting costs and accuracies
    x = 1 : plot_idx - 1;
%     plot(x*100, costs(:, 1), x*100, costs(:, 2));
%     title('Cost plot');
%     xlabel('update step')
%     ylabel('cost')
%     figure();
    % losses plot
    plot(100*x, losses(:, 1), 100*x, losses(:, 2));
    title('Loss plot');
    xlabel('update step')
    ylabel('loss')
    figure();
    % the train accuracies are blue, the validation accuracies are red
%     plot(100*x, accs(:, 1), 100*x, accs(:, 2));
%     title('Accuracy plot');
%     xlabel('update step')
%     ylabel('accuracy')
%     figure();
    
    NetParams.means = means;
    NetParams.var = var;
    NetParamsStar = NetParams;
end


% Loads all training data, up to the limit
function [X_train, Y_train, y_train, XVal, YVal, yVal, testX, testY, testy] = LoadAll(val_size_limit)
    [x, Y, y] = LoadBatch('data_batch_1.mat');
    [x2, Y2, y2] = LoadBatch('data_batch_2.mat');
    [x3, Y3, y3] = LoadBatch('data_batch_3.mat');
    [x4, Y4, y4] = LoadBatch('data_batch_4.mat');
    [x5, Y5, y5] = LoadBatch('data_batch_5.mat');
    % add the 45,000 training 
    X_train = [x, x2, x3, x4, x5(:,1:val_size_limit)];
    %size(X_train)
    Y_train = [Y, Y2, Y3, Y4, Y5(:,1:val_size_limit)];
    %size(Y_train)
    y_train = [y; y2; y3; y4];
    y_train = [y_train; y5(1:val_size_limit)];
    %size(y_train)

    % add the validating set, based on limit provided as parameter
    XVal = x5(:, val_size_limit + 1:end);
    %size(XVal)
    YVal = Y5(:, val_size_limit + 1:end);
    %size(YVal)
    yVal = y5(val_size_limit + 1:end);
    %size(yVal)
    [testX, testY, testy] = LoadBatch('test_batch.mat');
end

% coarse search to find lambda
function CoarseSearch(l_min, l_max, cycles, n_batch, GDparams, W, b)
    % 2 cycles
    % n_s =2 floor(n/n_batch)
    % l_min = -5
    % l_max = -1
    lambda = 0;
    val_size_limit = 5000;
    [X_train, Y_train, y_train, XVal, YVal, yVal, testX, testY, testy] = LoadAll(val_size_limit);
    n_s = 2*floor(size(X_train, 2)/n_batch);
    lambdas = [];
    accs = [];
    %fprintf('lambda_res.txt','%6s %12s\r\n','Lambda','Accuracy');
    iterations = 20;
    for i = 1: iterations
        l = l_min + (l_max - l_min)*rand(1, 1);
        lambda = 10^l;
        lambdas = [lambdas, lambda];
        [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, W, b, lambda);
        acc = ComputeAccuracy(testX, testY, Wstar, bstar);
        accs = [accs, acc];
        fprintf('Finished search #%d\n', i);
    end
    %fprintf('lambda_res.txt','%6.2f %12.8f\r\n',res);
    fig = figure;
    axes1 = axes('Parent',fig)
    hold(axes1,'all');
    
    scatter(lambdas, accs);
    
    title('Lambdas and Accuracies');
    xlabel('lambda')
    ylabel('accuracy')

end

function CoarseSearchBN(X_train, Y_train, X_val, Y_val, X_test, Y_test, l_min, l_max, GDParams, NetParams)
    % 2 cycles
    % n_s =2 floor(n/n_batch)
    % l_min = -5
    % l_max = -1
    lambdas = [];
    accs = [];
    iterations = 10;
    fprintf("starting coarse\n");
    for i = 1: iterations
        l = l_min + (l_max - l_min)*rand(1, 1);
        lambda = 10^l;
        lambdas = [lambdas, lambda];
        NetParams_star = MiniBatchGDBN(X_train, Y_train, GDParams, NetParams, lambda, X_val, Y_val);
        acc = ComputeAccuracyBN(X_test, Y_test, NetParams_star);
        accs = [accs, acc];
        fprintf('Finished search #%d\n', i);
    end
    %fprintf('lambda_res.txt','%6.2f %12.8f\r\n',res);
    fig = figure;
    axes1 = axes('Parent',fig)
    hold(axes1,'all');
    
    scatter(lambdas, accs);
    
    title('Lambdas and Accuracies');
    xlabel('lambda')
    ylabel('accuracy')

end

function [W, b] = InitializeParametersSig(K, d, m, sig)
    rng(400);
    W = {};
    b = {};
    % from lecture 4
    % Gaussian distribution w mean 0, std 1/sqrt(d)
    W{1} = sig * randn(m(1),d);
    b{1} = sig * randn(m(1),1);
    i = 2;
    while i <= size(m,2)
        W{i} = sig * randn(m(i),m(i-1)); 
        b{i} = sig * randn(m(i),1);
        i = i + 1;
    end
    W{i} = sig * randn(K,m(end));
    b{i} = sig * randn(K,1);
end

function [W, b, gammas, betas] = InitializeParametersBNSig(K, m, d, sig)
    rng(400);
    W = {};
    b = {};
    gammas = {};
    betas = {};
    
    i = 1;
    while i <= size(m,2)
        %input check
        if i ==1
           W{i} = sig * randn(m(i),d);
           b{i} = sig * randn(m(i),1);
        else
           W{i} = sig * randn(m(i),m(i-1)); 
           b{i} = sig * randn(m(i),1);
        end
        gammas{i} = sig * randn(m(i),1); 
        betas{i} = sig * randn(m(i),1);
        i = i + 1;
    end
    W{i} = sig * randn(K,m(end));
    b{i} = sig * randn(K,1);
end

% Relative error function
function [grad_W_err, grad_b_err] = RelativeError(grad_W, ngrad_W, grad_b, ngrad_b) 
    grad_W_err = {};
    grad_b_err = {};
    eps = 1e-9;
    for i=1:numel(grad_W)
        numerator_W = double(abs(ngrad_W{i} - grad_W{i}));
        denominator_W = max(eps, abs(ngrad_W{i}) + abs(grad_W{i}));
        grad_W_err{i} = double(max(numerator_W ./ denominator_W));
   
    end
    
    for i=1:numel(grad_b)
        numerator_b = double(abs(ngrad_b{i} - grad_b{i}));
        denominator_b = max(eps, abs(ngrad_b{i}) + abs(grad_b{i}));
        grad_b_err{i} = double(max(numerator_b ./ denominator_b));
    end
end

% exercise 1: compute gradients, d = 10, lambda = 0, start w 2-layer, then
% 3-layer, then 4-layer,
function ExerciseOne(layers, m, trainX, trainY, trainy)
    
    lambda = 0;
    
    d = size(trainX(1:10, 1:10), 1);
    K = size(trainY(:, 1:10), 1);
    
    [W, b] = InitializeParameters(K, d, m, layers);
    NetParams.W = W;
    NetParams.b = b;
    NetParams.use_bn = false;
    
    [P, H] = EvaluateClassifier(trainX(1:10, 1:10), W, b);
    [dl_dW, dl_db] = ComputeGradients(trainX(1:10, 1:10),trainY(:, 1:10), P, NetParams, lambda, H);
    
    [grad_b, grad_W] = ComputeGradsNumSlow(trainX(1:10, 1:10), trainY(:, 1:10), NetParams, lambda, 1e-5);
    [grad_W_err, grad_b_err] = RelativeError(dl_dW, grad_W, dl_db, grad_b)
end

% exercise 2: 
% 1) replicate assignment 2 (2 layer, 50 nodes w mini-batch & cyclical)
% 2) train a 3-layer network (m=[50,50], n_batch=100, eta_min=1e-5,
% eta_max=1e-1, lambda=0.005, 2cycles, n_s = 5*45000/n_batch) and use
% Xavier or He initialization
function ExerciseTwo()
    assignment2 = false;
    layer3test = false;
    layer9test = false;
    layer6test = true;
    
    val_size_limit = 5000;
    [X_train, Y_train, y_train, XVal, YVal, yVal, testX, testY, testy] = LoadAll(val_size_limit);
    
    if assignment2
        layers = 2;
        m = 50;
        lambda = 0.01;
        n_s = 800;
        batch_size = 100;
        cycles = 3;

        d = size(X_train, 1);
        K = size(Y_train, 1);

        [W, b] = InitializeParameters(K, d, m, layers);
        NetParams.W = W;
        NetParams.b = b;
        NetParams.use_bn = false;

        GDparams.n_s = n_s;
        GDparams.n_batches = batch_size;
        GDparams.cycles = cycles;
        [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, NetParams, lambda);
        acc = ComputeAccuracy(testX, testY, Wstar, bstar);
        fprintf("Accuracy: %.2f%\n", acc);
    end
    
    if layer3test
        layers = 3;
        m = [50, 50];
        cycles = 2;
        batch_size=100;
        lambda = .005;
        n_s = 5*45000/batch_size;
        
        d = size(X_train, 1);
        K = size(Y_train, 1);
        
        [W, b] = InitializeParameters(K, d, m, layers);
        NetParams.W = W;
        NetParams.b = b;
        NetParams.use_bn = false;

        GDparams.n_s = n_s;
        GDparams.n_batches = batch_size;
        GDparams.cycles = cycles;
        [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, NetParams, lambda);
        acc = ComputeAccuracy(testX, testY, Wstar, bstar);
        fprintf("Accuracy: %.2f%%\n", acc);
    end
    
    if layer9test
        layers = 9;
        m = [50, 30, 20, 20, 10, 10, 10, 10];
        cycles = 2;
        batch_size=100;
        lambda = .005;
        n_s = 5*45000/batch_size;
        
        d = size(X_train, 1);
        K = size(Y_train, 1);
        
        [W, b] = InitializeParameters(K, d, m, layers);
        NetParams.W = W;
        NetParams.b = b;
        NetParams.use_bn = false;

        GDparams.n_s = n_s;
        GDparams.n_batches = batch_size;
        GDparams.cycles = cycles;
        [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, NetParams, lambda);
        acc = ComputeAccuracy(testX, testY, Wstar, bstar);
        fprintf("Accuracy: %.2f%%\n", acc);
    end
    
    if layer6test
        layers = 3;
        m = [50, 30, 20, 20, 10];
        cycles = 2;
        batch_size=100;
        lambda = .005;
        n_s = 5*45000/batch_size;
        
        d = size(X_train, 1);
        K = size(Y_train, 1);
        
        [W, b] = InitializeParameters(K, d, m, layers);
        NetParams.W = W;
        NetParams.b = b;
        NetParams.use_bn = false;

        GDparams.n_s = n_s;
        GDparams.n_batches = batch_size;
        GDparams.cycles = cycles;
        [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, NetParams, lambda);
        acc = ComputeAccuracy(testX, testY, Wstar, bstar);
        fprintf("Accuracy: %.2f%%\n", acc);
    end
    
    
end

function BatchNormTest(layers, m, trainX, trainY, trainy, testX, testy)
    if false
        lambda = 0;
        K = size(trainY(:, 1:5),1);
        d = size(trainX(:, 1:5),1);

        [W, b, gammas, betas] = InitializeParametersBN(K, m, d);
        NetParams.W = W;
        NetParams.b = b;
        NetParams.gammas = gammas;
        NetParams.betas = betas;
        NetParams.use_bn = true;

        fprintf("Evaluating Classifier... \n")
        meta = EvaluateClassifierBN(trainX(:, 1:5), NetParams);
        fprintf("Computing gradients... \n");
        [dl_dW, dl_db, dl_dgammas, dl_dbetas] = ComputeGradientsBN(trainX(:, 1:5),trainY(:, 1:5), meta, NetParams, lambda);
        fprintf("Computing gradientsNumSlow... \n");
        [grad_b, grad_W, grad_gam, grad_beta] = ComputeGradsNumSlow(trainX(:, 1:5), trainY(:, 1:5), NetParams, lambda, 1e-5);
        [grad_W_err, grad_b_err] = RelativeError(dl_dW, grad_W, dl_db, grad_b)
        [grad_g_err, grad_beta_err] = RelativeError(dl_dgammas, grad_gam, dl_dbetas, grad_beta)
        %         acc = ComputeAccuracy(X, y, W, b);
%         fprintf("Accuracy: %d\n",)

    %     [grad_W_err, grad_b_err] = RelativeError(dl_dW, grad_W, dl_db, grad_b)
    %     [grad_gam_err, grad_beta_err] = RelativeError(dl_dgammas, grad_gam, dl_dbetas, grad_beta)
%         for i = 1 : numel(dl_dW)
%                 if i == numel(dl_dW)
%                     fprintf("Max abs divergence is: \n W(%d) %e \nb(%d) %e \n\n",i,  ...
%                     max(max(abs(dl_db{i}-grad_W{i}))),i, ...
%                     max(abs(dl_db{i}-grad_b{i})));
%                 else
%                     fprintf("Max abs divergence is: \n W(%d) %e \nb(%d) %e \ngamma(%d) %e \nbeta(%d) %e\n\n",i,  ...
%                     max(max(abs(dl_dW{i}-grad_W{i}))),i, ...
%                     max(abs(dl_db{i}-grad_b{i})),...
%                     i,max(abs(dl_dgammas{i}-grad_gam{i})),...
%                     i,max(abs(dl_dbetas{i}-grad_beta{i}))...
%                     );
%                 end
%         end
    end
    
    if false
        % train 3 layer network
        m = [50, 50];
        lambda = 0.005;
        [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = LoadAll(5000);
        K = size(Y,1);
        d = size(X,1);
        GDParams.cycles = 2;
        N = size(X, 2);
        GDParams.n_batches = 100;
        GDParams.n_s = (5*45000)/GDParams.n_batches; 
        [W , b, gammas, betas] = InitializeParametersBN(K,m,d);
        NetParams.W = W;
        NetParams.b = b;
        NetParams.gammas = gammas;
        NetParams.betas = betas;
        NetParams_star = MiniBatchGDBN(X, Y, GDParams, NetParams, lambda, X_val, Y_val);

        %P = EvaluateClassifierBN(X_test, NetParams_star, NetParams_star.mean, NetParams_star.var);
        acc = ComputeAccuracyBN(X_test, Y_test, NetParams_star);
        
        fprintf("Accuracy on test data is : %f\n",acc);
        % 52.8%
    end
    
    if true
        % train 6 layer network
        m = [50, 30, 20, 20, 10];
        lambda = 0.005;
        [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = LoadAll(5000);
        K = size(Y,1);
        d = size(X,1);
        GDParams.cycles = 2;
        N = size(X, 2);
        GDParams.n_batches = 100;
        GDParams.n_s = (5*45000)/GDParams.n_batches; 
        [W , b, gammas, betas] = InitializeParametersBN(K,m,d);
        NetParams.W = W;
        NetParams.b = b;
        NetParams.gammas = gammas;
        NetParams.betas = betas;
        NetParams_star = MiniBatchGDBN(X, Y, GDParams, NetParams, lambda, X_val, Y_val);

        %P = EvaluateClassifierBN(X_test, NetParams_star, NetParams_star.mean, NetParams_star.var);
        acc = ComputeAccuracyBN(X_test, Y_test, NetParams_star);
        
        fprintf("Accuracy on test data is : %f\n",acc);
        % 52.8%
    end
    
    
    % do the coarse search, 10 iterations
    if false
        m = [50, 50];
        GDParams.cycles = 2;
        GDParams.n_batches = 100;
        l_min = -5;
        l_max = -2;
        [X_train, Y_train, y_train, Xval, Yval, yVal, testX, testY, testy] = LoadAll(5000);
        n_s = 2*floor(size(X_train, 2)/GDParams.n_batches);
        GDParams.n_s = n_s;
        K = size(Y_train,1);
        d = size(X_train,1);
        %n_s = 2*floor(size(X_train, 2)/GDParams.n_batches); 
        %fprintf("n_s: %d\n", n_s);
        [W , b, gammas, betas] = InitializeParametersBN(K,m,d);
        NetParams.W = W;
        NetParams.b = b;
        NetParams.gammas = gammas;
        NetParams.betas = betas;
        CoarseSearchBN(X_train, Y_train, Xval, Yval, testX, testY, l_min, l_max, GDParams, NetParams);
    end
end

function SigTests(sig)
    val_size_limit = 5000;
    [X_train, Y_train, y_train, XVal, YVal, yVal, testX, testY, testy] = LoadAll(val_size_limit);
    
    % no batch normalization
    m = [50, 50];
    cycles = 2;
    batch_size=100;
    lambda = .005;
    n_s = 5*45000/batch_size;

    d = size(X_train, 1);
    K = size(Y_train, 1);

    [W, b] = InitializeParametersSig(K, d, m, sig);
    NetParams.W = W;
    NetParams.b = b;

    GDparams.n_s = n_s;
    GDparams.n_batches = batch_size;
    GDparams.cycles = cycles;
    [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, NetParams, lambda);
    acc = ComputeAccuracy(testX, testY, Wstar, bstar);
    fprintf("Accuracy with no batch normalization: %.2f%%\n", acc);
    
    % batch normalization
    GDParams.cycles = 2;
    GDParams.n_batches = 100;
    GDParams.n_s = (5*45000)/GDParams.n_batches; 
    [W , b, gammas, betas] = InitializeParametersBNSig(K,m,d, sig);
    NetParams.W = W;
    NetParams.b = b;
    NetParams.gammas = gammas;
    NetParams.betas = betas;
    NetParams_star = MiniBatchGDBN(X_train, Y_train, GDParams, NetParams, lambda, XVal, YVal);
    acc = ComputeAccuracyBN(testX, testY, NetParams_star);

    fprintf("Accuracy with batch normalization : %f%%\n",acc);
end
