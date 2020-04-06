addpath Datasets/cifar-10-matlab/;
data_batch_1 = load('data_batch_1.mat');
%I = reshape(A.data', 32, 32, 3, 10000);
%I = permute(I, [2, 1, 3, 4]);
%montage(I(:, :, :, 1:500), 'Size', [5, 5]);

data_batch_2 = load('data_batch_2.mat');
test_batch = load('test_batch.mat');

rng(400);
[X, Y, y] = LoadBatch('data_batch_1.mat');
[X_train, Y_train, y_train] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
K = size(Y, 1);
d = size(X, 1);
W = randn(K, d) * 0.01;
b = randn(K, 1) * 0.01;
%X = X(1:20, 1);
%Y = Y(:, 1);
K = size(Y, 1);
d = size(X, 1);
%W = W(:, 1:20);
P = EvaluateClassifier(X, W, b);
%[grad_w, grad_b] = ComputeGradients(X, Y, P, W, lambda);

%[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, 1e-6);
%[grad_W_err, grad_b_err] = RelativeError(grad_w, ngrad_W, grad_b,ngrad_b);

GDparams.n_batches = 100;
GDparams.n_eta = 0.1;
GDparams.epochs = 40;
lambda = 0.0;
[Wstar, bstar] = MiniBatchGD(X, Y, X_train, Y_train, GDparams, W, b, lambda);
acc = ComputeAccuracy(X_test, Y_test, Wstar, bstar);

% visualization of the weight matrix W as an image, assuming W is a 10xd
% matrix
for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:)))/(max(im(:)) - min (im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

%montage(s_im(:, :, :, :), 'Size', [5, 5]);

function [X, Y, y] = LoadBatch(filename)
    batch = load(filename);
    X = double(batch.data);
    X = transpose(X)/255.0;
    y = double(batch.labels) + 1;
    % create one hot encoding
    N = size(X, 2);
    for i = 1 : N
        Y(y(i), i) = 1;
    end
end

function P = EvaluateClassifier(X, W, b)
    % finding s, s = Wx+b
    s = W*X + b*ones(1, size(X, 2));
    
    % do softmax(s) = exp(s)/((1^T)(exp(s)))
    exponential = exp(s);
    % 1 means the 1st dimension (AKA rows)
    D = ones(size(W, 1), 1)*sum(exponential, 1);
    P = exponential./D; 
end

function J = ComputeCost(X, Y, W, b, lambda)  
    p = EvaluateClassifier(X, W, b);
    sum_cross_entr = -log(sum(Y .* p,1));
    
    reg_sum = sum(sum(W .* W,'double'), 'double');
   
    % regularization term, summation of W^2 multiplied with lambda
    reg_term = lambda * reg_sum;
    J = mean(sum_cross_entr) + reg_term;
end

function acc = ComputeAccuracy(X, y, W, b)
 
    p = EvaluateClassifier(X, W, b);
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

function [grad_w, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    N = size(X, 2);
    g_batch = -(Y - P);
    % divide by magnitude of N
    grad_w = (1/N)*g_batch*transpose(X);
    grad_b = (1/N)*g_batch*ones(size(X, 2), 1);
    
    % add regularization term
    grad_w = grad_w + 2*lambda*W;
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end

    for i=1:numel(W)

        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c1) / (2*h);
    end
end

function [Wstar, bstar] = MiniBatchGD(X_train, Y_train, XVal, YVal, GDparams, W, b, lambda)
    eta = GDparams.n_eta;
    epochs = GDparams.epochs;
    n_batch = GDparams.n_batches;
    N = size(X_train, 1);
    % cost and accuracy matrices
    costs = zeros(epochs,2);
    accs = zeros(epochs,2);
    
    % do all epocs, runs through all imgs 
    for e=1:epochs
        % TODO: shuffle training samples before each epoch
        % generate batches to iterate through
        for j=1:N/n_batch
            % "Sample a batch of training data"
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X_train(:, j_start:j_end);
            Ybatch =  Y_train(:, j_start:j_end);
                        
            % complete forward pass - "calculate loss/cost"
            Pbatch = EvaluateClassifier(Xbatch, W, b);
            
            % complete backwards pass - "to calculate gradients"
            [gradW, gradb] = ComputeGradients(Xbatch, Ybatch, Pbatch, W, lambda);
            
            % "update the parameters using the gradient"
            W = W - eta*gradW;
            b = b - eta*gradb;
        end
        
        % save the cost and accuracy after each epoch
        costs(e,1) = ComputeCost(X_train, Y_train, W, b, lambda);
        costs(e,2) = ComputeCost(XVal, YVal, W, b, lambda);
        
        accs(e,1) = ComputeAccuracy(X_train, Y_train, W, b);
        accs(e,2) = ComputeAccuracy(XVal, YVal, W, b);
        
    end
    % plotting costs and accuracies
    x = 1 : epochs;
    % the train costs are blue, the validation costs are red
    plot(x, costs(:,1), x, costs(:,2));
    figure();
    % the train accuracies are blue, the validation accuracies are red
    plot(x, accs(:,1),x, accs(:,2));
        
    Wstar = W;
    bstar = b;
end

function [grad_W_err, grad_b_err] = RelativeError(grad_W, ngrad_W, grad_b, ngrad_b)
    numerator_W = abs(ngrad_W - grad_W);
    denominator_W = max(0.0001, abs(ngrad_W) + abs(grad_W));
    grad_W_err = max(max(numerator_W ./ denominator_W));
    
    numerator_b = abs(ngrad_b - grad_b);
    denominator_b = max(0.0001, abs(ngrad_b) + abs(grad_b));
    grad_b_err = max(numerator_b ./ denominator_b);
    
end








