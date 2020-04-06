% input sizes:
% x: dx1
% p: Kx1
% W_1: mxd
% W_2: Kxm
% b_1: mx1
% b_2: Kx1

% learning rates: too small --> training takes too long, too large -->
% training diverging -- want an adaptive learning rate which changes to
% match the local shape of the cost surface at the current estimate of the
% network's parameters
addpath Datasets/cifar-10-matlab/;
rng(400);
d = 3072;
m = 50;
% Exercise 1: read in the data
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
K = size(trainY, 1);
%disp(trainy);
[ValX, ValY, Valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
[W, b] = InitializeParameters(K, d, m);
lambda = 1;
h = 1e-5;



n_s = 1250;
batch_size = 100;
cycles = 3;
GDparams.n_s = n_s;
GDparams.n_batches = batch_size;
GDparams.cycles = cycles;

% W{1} = W{1}(:, 1:20);
% W{2} = W{2}(:, :);
% [P, H_batch] = EvaluateClassifier(trainX(1:20, 1:2), W, b);
% [dl_dW, dl_db] = ComputeGradients(trainX(1:20, 1:2),trainY(:, 1:2), P, W, lambda, H_batch);
% [grad_b, grad_W] = ComputeGradsNumSlow(trainX(1:20, 1:2), trainY(:, 1:2), W, b, lambda, 1e-5);
% [grad_W_err, grad_b_err] = RelativeError(dl_dW, grad_W, dl_db, grad_b)
lambda = 0.0025585;
% [X_train, Y_train, y_train, XVal, YVal, yVal, testX, testY, testy] = LoadAll();
% [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, W, b, lambda);
FinalTest(GDparams, W, b, lambda);
% l_min =-3;
% l_max = -1;
% CoarseSearch(l_min, l_max, cycles, batch_size, GDparams, W, b);
%.0025585



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
function [W, b] = InitializeParameters(K, d, m)
    % Gaussian distribution w mean 0, std 1/sqrt(d)
    W1 = randn(m, d) * 1.0/sqrt(d);
    % Gaussian distribution w mean 0, std 1/sqrt(m)
    W2 = randn(K, m) * 1.0/sqrt(m);
    % store in cell array
    W = {W1, W2};
    b1 = zeros(m, 1);
    b2 = zeros(K, 1);
    b = {b1, b2};
end
% Exercise 2: compute the gradients 
% function that returns the final p values and the intermediary activation
% values
function [P, H_batch] = EvaluateClassifier(X, W, b)
    % finding s, s = Wx+b
    W1 = W{1};
    W2 = W{2};
    b1 = b{1};
    b2 = b{2};
    s1 = W1*X + b1*ones(1, size(X, 2));
    
    % Activation
    H_batch = max(s1, 0);
    
    s = W2*H_batch + b2*ones(1, size(X, 2));
    
    % do softmax(s) = exp(s)/((1^T)(exp(s)))
    exponential = exp(s);
    % 1 means the 1st dimension (AKA rows)
    D = ones(size(W, 1), 1)*sum(exponential, 1);
    P = exponential./D;
end

function [J, loss] = ComputeCost(X, Y, W, b, lambda)  
    p = EvaluateClassifier(X, W, b);
    loss_func = -log(sum(Y .* p,1));
    
    W1 = W{1};
    W2 = W{2};
    reg_sum = sum(sum(W1 .* W1,'double'),'double');
    reg_sum = reg_sum + sum(sum(W2 .* W2,'double'),'double');
   
    % regularization term, summation of W^2 multiplied with lambda
    reg_term = lambda * reg_sum;
    loss = mean(loss_func);
    J = loss + reg_term;
end

% TODO: compute accuracy
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

% function that computes the gradients of the cost function for a
% mini-batch of data given the values computed from the forward pass
function [dl_dW, dl_db] = ComputeGradients(X, Y, P, W, lambda, H_batch)
    N = size(X, 2);
    g_batch = -(Y - P);
    
    W1 = W{1};
    W2 = W{2};
   
    % gradient of loss
    dl_dW2 = (1/N)*g_batch*transpose(H_batch);
    dl_dW2 = dl_dW2 + 2*lambda*W2;
    dl_db2 = (1/N)*g_batch*ones(size(X, 2), 1);
    
    % back propogate the gradient back through the 2nd layer
    g_batch = transpose(W2)*g_batch;
    g_batch = g_batch .* (H_batch > 0);
    dl_dW1 = (1/N)*g_batch*transpose(X);
    dl_dW1 = dl_dW1 + 2*lambda*W1;
    dl_db1 = (1/N)*g_batch*ones(size(X, 2), 1);
    
    % store in cell arrays
    dl_dW = {dl_dW1, dl_dW2};
    dl_db = {dl_db1, dl_db2};
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));

        for i=1:length(b{j})

            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            [c1, ~] = ComputeCost(X, Y, W, b_try, lambda);

            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            [c2, ~] = ComputeCost(X, Y, W, b_try, lambda);

            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));

        for i=1:numel(W{j})

            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            [c1, ~] = ComputeCost(X, Y, W_try, b, lambda);

            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);

            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end

% Exercise 3: train network with cyclical learning rates
function [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, W, b, lambda)
    n_batches = GDparams.n_batches;
    n_s = GDparams.n_s;
    cycles = GDparams.cycles;
    
    plot_idx = 1;
    
    eta_min = 1e-3;
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
        [gradW, gradb] = ComputeGradients(Xbatch, Ybatch, Pbatch, W, lambda, H_batch);
             
        % "update the parameters using the gradient"
        W{1} = W{1} - eta*gradW{1};
        W{2} = W{2} - eta*gradW{2};
        b{1} = b{1} - eta*gradb{1};
        b{2} = b{2} - eta*gradb{2};
        
        
        % save the cost and accuracy after each epoch
        if mod(t,100)==0
            [cost_train, loss_train] = ComputeCost(X_train, Y_train, W, b, lambda);
            [cost_val, loss_val] = ComputeCost(XVal, YVal, W, b, lambda);
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

% Exercise 4: train network for real
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


function FinalTest(GDparams, W, b, lambda)
    val_size_limit = 1000;
    [X_train, Y_train, y_train, XVal, YVal, yVal, testX, testY, testy] = LoadAll(val_size_limit);
    [Wstar, bstar] = CyclicalMiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, W, b, lambda);
    acc = ComputeAccuracy(testX, testY, Wstar, bstar)
    
end
% Relative error function
function [grad_W_err, grad_b_err] = RelativeError(grad_W, ngrad_W, grad_b, ngrad_b)  
    numerator_W = abs(ngrad_W{1} - grad_W{1});
    denominator_W = max(0.0001, abs(ngrad_W{1}) + abs(grad_W{1}));
    grad_W1_err = max(max(numerator_W ./ denominator_W));
    
    numerator_W = abs(ngrad_W{2} - grad_W{2});
    denominator_W = max(0.0001, abs(ngrad_W{2}) + abs(grad_W{2}));
    grad_W2_err = max(max(numerator_W ./ denominator_W));
    
    numerator_b = abs(ngrad_b{1} - grad_b{1});
    denominator_b = max(0.0001, abs(ngrad_b{1}) + abs(grad_b{1}));
    grad_b1_err = max(numerator_b ./ denominator_b);
    
    numerator_b = abs(ngrad_b{2} - grad_b{2});
    denominator_b = max(0.0001, abs(ngrad_b{2}) + abs(grad_b{2}));
    grad_b2_err = max(numerator_b ./ denominator_b);
    
    grad_W_err = {grad_W1_err, grad_W2_err};
    grad_b_err = {grad_b1_err, grad_b2_err};
    
end