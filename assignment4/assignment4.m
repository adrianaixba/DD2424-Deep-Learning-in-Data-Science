book = 'Datasets/goblet_book.txt';
fid = fopen(book, 'r');
% all the characters are now in a vector
book_data = fscanf(fid, '%c');
fclose(fid);

% list of the characters in book_data
book_chars = unique(book_data);

% dimensionality 
K = length(book_chars);

% dimensionality of hidden state, defined by assignment page
m = 100;

% learning rate, default defined by assignment page
eta = .1;

% length of input sequence, default defined by assignment page
seq_length = 25;

% default defined by assignment page
sig = .01;

%???
d = K;

% maps created
[char_to_ind, ind_to_char] = CreateMaps(book_chars, K);
RNN = set_HyperParameters(m, K, sig, seq_length);

h0 = zeros(m, 1);
x0 = zeros(d, 1);
n = 10;

SynthesizeText(RNN, h0, x0, 10);


X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);
X = one_hot_enc(X_chars, char_to_ind);
Y = one_hot_enc(Y_chars, char_to_ind);
% CheckGrads(X, Y, h0, m, K, sig);

train_Ada(book_data, RNN, char_to_ind, ind_to_char);

function [char_to_ind, ind_to_char] = CreateMaps(char_vect, k)
    char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');
    for i = 1: k
        char_to_ind(char_vect(i)) = i;
        ind_to_char(i) = char_vect(i);
    end
end

function RNN = set_HyperParameters(m, k, sig, length)
    % initialize bias vectors
    RNN.b = zeros(m, 1);
    RNN.c = zeros(k, 1);
    RNN.m = m;
    RNN.k = k;
    RNN.seq_len = length;
    
    % initialize weight vectors
    RNN.u = randn(m, k)*sig;
    RNN.w = randn(m, m)*sig;
    RNN.v = randn(k, m)*sig;
end

% returns one-hot encoding of a given sequence
% var is of size Kxseq_len
function one_hot = one_hot_enc(chars_seq, char_to_ind)
    one_hot = zeros(length(char_to_ind.keys()), length(chars_seq));
    
    for i=1:length(chars_seq)
        ind = char_to_ind(chars_seq(i));
        one_hot(ind, i) = 1;
    end
end

function txt = indxs2txt(idxs, idx_to_char)
    txt = "";
    
    for i = 1 :length(idxs)
        txt = txt + idx_to_char(idxs(i));
    end
    
end

% returns a synthesized text 
function synthesized_text = SynthesizeText(RNN, h0, x0, n)
    
    synthesized_text = zeros(1, n);
    x = x0;
    % makes a identity matrix KxK
    X = eye(size(RNN.c, 1));
    a = zeros(RNN.m, n);
    h = zeros(RNN.m, n);
    o = zeros(size(RNN.c, 1), n);
    p = zeros(size(RNN.c, 1), n);
    % iterate for the length of sequence you want to generate 
    for t=1:n
        % hidden state at t, before non-linearality (mx1)
        if t == 1
            a = RNN.w*h0 + RNN.u*x + RNN.b;
        else
            a = RNN.w*h + RNN.u*x + RNN.b;
        end
        % hidden state (mx1)
        h = tanh(a);
        o = RNN.v*h + RNN.c;
        % do softmax(o{t}) = exp(o{t})/((1^T)(exp(o{t})))
        exponential = exp(o);
        p = exponential/sum(exponential);
        % now select a random character based on the output probability
        % scores p
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a > 0);
        % ii is representative of the char in the vocabulary
        ii = ixs(1);
        % save the index as the newly generated char
        synthesized_text(t) = ii;
        % get the (t+1)th char in the seq, which is the input for the next
        % iteration
        x = X(:, ii);
    end
    % return one-hot encoding of each sampled character, synthesized text
end

% computes the forward function, returns the loss and the final and
% intermediary output vectors at each time step (for back-pass algo)
% eqns 1-4
function Params = ForwardPass(RNN, X, Y, h0)
    % seq length
    T = size(X, 2);
    loss = zeros(1, T);
    
    % initialize params - based off assignment page & lecture 9
    a = zeros(RNN.m, T);
    h = zeros(RNN.m, T);
    o = zeros(RNN.k, T);
    p = zeros(RNN.k, T);
    
    
    for t=1:T
        if t == 1
            a(:, t) = RNN.w*h0 + RNN.u*X(:, t) + RNN.b;
        else
            % hidden state before non-linearity (1)
            a(:, t) = RNN.w*h(:, t-1) + RNN.u*X(:, t) + RNN.b;
        end
        % hidden state (2)
        h(:, t) = tanh(a(:, t));
        % output vector of unormalized probabilities for each class (3)
        o(:, t) = RNN.v*h(:, t) + RNN.c;
        % output prob vector (softmax of o(:, t)) (4)
        exponential = exp(o(:, t));
        p(:, t) = exponential/sum(exponential);
        
        % compute the loss (5)
        loss(t) = transpose(Y(:, t)) * p(:, t);
    end
    
    Params.a = a;
    Params.h = h;
    Params.o = o;
    Params.p = p;
    Params.h0 = h0;
    
    % sum of cross entropy loss 
    Params.loss = -sum(log(loss));  
end

% computes the backwards function, from equations in lecture 9
function RNN = BackwardPass(RNN, Params, X, Y, grads)
    eps = 1e-14;

    for f = {'b','c','u','w','v'}
        
        % clipping gradients for exploding problem
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);

        M.(f) = M.(f) + grads.(f).^2;
        RNN.(f) = RNN.(f{1}) - RNN.eta*(grads.(f)./(M.(f) + eps).^(0.5));
    end

end

function grads = ComputeGrads(RNN, Params, X, Y)
    N = RNN.seq_len;
    dh_do = zeros(N, RNN.m);
    da_do = zeros(N, RNN.m);
    
    p = Params.p;
    h = Params.h;
    a = Params.a;
    
    dl_do = -transpose((Y - Params.p));
    
    grads.v = transpose(dl_do)*transpose(h);
    grads.c = sum(transpose(dl_do), 2);
    
    % backwards gradients
    dl_dh(N, :) = dl_do(N, :) * RNN.v;  
    dl_da(N, :) = dl_dh(N, :) * diag(1 - (tanh(a(:, N))).^2);
    
    for t=N-1: -1:1
        dl_dh(t, :) = dl_do(t, :) * RNN.v + dl_da(t+1, :) * RNN.w;
        dl_da(t, :) = dl_dh(t, :) * diag(1 - (tanh(a(:, t))).^2);  
    end
    
    grads.b = sum(transpose(dl_da), 2);
    
    h_new = [Params.h0, h(:,1:end-1)];
    grads.w = transpose(dl_da) * transpose(h_new);
    grads.u = transpose(dl_da) * transpose(X);
end

function loss = ComputeLoss(X, Y, RNN, h)
    w = RNN.w;
    u = RNN.u;
    v = RNN.v;
    b = RNN.b;
    c = RNN.c;
    n = size(X, 2);
    loss = 0;

    for t = 1 : n
        at = w*h + u*X(:, t) + b;
        h = tanh(at);
        o = v*h + c;
        pt = exp(o);
        p = pt/sum(pt);

        loss = loss - log(transpose(Y(:, t))*p);
    end
end

function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = {'v','c','b','w','u'}
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.w, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end

function gradErr = RelativeError(gradsNum, gradsAn)
    eps = 1e-9;
%     disp(gradsNum)
    for f = {'v','w','u'}
        disp(['computing relative error for: ' (f) ]);
        n = size(gradsNum.(f{1}), 2);
        
        disp(n);
        for i=1:n
            num = double(abs(gradsNum.(f{1})(i) - gradsAn.(f{1})(i)));
            den = max(eps, abs(gradsNum.(f{1})(i)) + abs(gradsAn.(f{1})(i)));
            gradErr.(f{1})(i) = double(max(num ./ den));
        end
        %disp(size(gradErr.(f{1})));
    end

end

function [RNN, h, SmoothLoss] = AdaGrad(book_data, RNN, char_to_ind, ind_to_char)
    e = 1;
    step = 0;
    Loss = [];
    SmoothLoss = [];
    M.u = zeros(size(RNN.u));
    M.v = zeros(size(RNN.v));
    M.w = zeros(size(RNN.w));
    M.c = zeros(size(RNN.c));
    M.b = zeros(size(RNN.b));
    h = zeros(RNN.m,1);
    smooth_loss = 0;
    
    % at each iteration
    X_seq = book_data(e:e+RNN.seq_len-1);
    Y_seq = book_data(e+1:e+RNN.seq_len);
    
    
    % one-hot encodings
    X = one_hot_enc(X_seq, char_to_ind);
    
    txt_idx = SynthesizeText(RNN,h,X(:,end), 200);
    disp(indxs2txt(txt_idx, ind_to_char));
    
    while step < 100000
        h = zeros(RNN.m,1);
        e = 1;
        while e < length(book_data) - RNN.seq_len - 1
            X_seq = book_data(e:e+RNN.seq_len-1);
            Y_seq = book_data(e+1:e+RNN.seq_len);
            X = one_hot_enc(X_seq, char_to_ind);
            Y = one_hot_enc(Y_seq, char_to_ind);
            
            Params = ForwardPass(RNN, X, Y, h);
            
            if smooth_loss == 0
               smooth_loss = Params.loss;
            else 
               smooth_loss = 0.999*smooth_loss + 0.001*Params.loss;
            end
           
            h = Params.h(:,end);
            grads = ComputeGrads(RNN, Params, X, Y);
            for f = {"u","v","w","c","b"}
              M.(f{1}) = M.(f{1})+grads.(f{1}) .^ 2;
              
              %avoid exploding gradient
              grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
              
              RNN.(f{1}) = RNN.(f{1})-RNN.eta * grads.(f{1}) ./ sqrt(M.(f{1})+ 1e-15);
            end
            step = step+1;
            e = e+RNN.seq_len;
            if mod(e-1,25*100) == 0
              Loss = [Loss;Params.loss];
              if length(SmoothLoss) == 0 %#ok<ISMT>
                SmoothLoss = [Params.loss];
              else
                SmoothLoss = [SmoothLoss; smooth_loss];
              end
           end
%            if mod(step,10000)==0
%             fprintf("step = %d loss = %f %\n\n", step, smooth_loss ,e *100 / length(book_data));
%             txt_idx = SynthesizeText(RNN,h,X(:,end), 200);
%             disp(indxs2txt(txt_idx, ind_to_char));
%             fprintf("\n");
%            end
        end
    end
%     txt_idx = SynthesizeText(RNN,h,X(:,end), 1000);
%     
%     disp(indxs2txt(txt_idx, ind_to_char));
    
    
    figure();
    plot(Loss);
    figure();
    plot(SmoothLoss(2:end));
            
end

function CheckGrads(X, Y, h0, m, k, sig)
    rng(400);
    seq_length = 25;
    RNN = set_HyperParameters(m, k, sig, seq_length);
    Params = ForwardPass(RNN, X, Y, h0);
    grads = ComputeGrads(RNN, Params, X, Y);
    % from assignment page
    h = 1e-4;
    num_grads = ComputeGradsNum(X, Y, RNN, h);
    
    gradErr = RelativeError(grads, num_grads)
%     max(gradErr.v)
%     max(gradErr.w)
%     max(gradErr.u)
    
    
end

function train_Ada(book_data, RNN, char_to_ind, ind_to_char)
    RNN.n_epoch = 5;
    RNN.eta = 0.1;
    [RNN_trained, h, smoothL] = AdaGrad(book_data,RNN,char_to_ind,ind_to_char);
    x0 = zeros(83,1);
    x0(1) = 1;
    txt_idx = SynthesizeText(RNN_trained,h, x0, 1000);
    indxs2txt(txt_idx, ind_to_char)
end
