# Neural_net_adam_from_scratch
clear; clc;
filepath = 'C:\Users\User\Downloads\1.xlsx';
data = xlsread(filepath);
X = data(:, 1:8);     % Input features
y = data(:, 9);       % Target variable

X_norm = zscore(X);
y_norm = zscore(y);


degree = 2;
X_poly = ones(size(X_norm, 1), 1);
for i = 1:size(X_norm, 2)
    for d = 1:degree
        X_poly = [X_poly, X_norm(:, i).^d];
    end
end

for i = 1:size(X_norm, 2)
    for j = i+1:size(X_norm, 2)
        X_poly = [X_poly, X_norm(:,i).*X_norm(:,j)];
    end
end


input_size = size(X_poly, 2);
hidden_sizes = [128, 64, 64, 64, 64, 64, 64, 64, 64, 32];  
output_size = 1;


num_layers = length(hidden_sizes) + 1;
W = cell(1, num_layers);
b = cell(1, num_layers);
m_W = cell(1, num_layers);
v_W = cell(1, num_layers);
m_b = cell(1, num_layers);
v_b = cell(1, num_layers);

for i = 1:num_layers
    if i == 1
        W{i} = randn(input_size, hidden_sizes(i)) * sqrt(2 / input_size);
    elseif i == num_layers
        W{i} = randn(hidden_sizes(end), output_size) * sqrt(2 / hidden_sizes(end));
    else
        W{i} = randn(hidden_sizes(i-1), hidden_sizes(i)) * sqrt(2 / hidden_sizes(i-1));
    end
    
    if i == num_layers
        b{i} = zeros(1, output_size);
    else
        b{i} = zeros(1, hidden_sizes(i));
    end
    
    m_W{i} = zeros(size(W{i}));
    v_W{i} = zeros(size(W{i}));
    m_b{i} = zeros(size(b{i}));
    v_b{i} = zeros(size(b{i}));
end


learning_rate = 0.001;
iterations = 5000;
batch_size = 32;
lambda = 0.0001;  


beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;


for iteration = 1:iterations

    batch_indices = randperm(size(X_poly, 1), batch_size);
    X_batch = X_poly(batch_indices, :);
    y_batch = y_norm(batch_indices);
    
 
    a = cell(1, num_layers+1);
    z = cell(1, num_layers);
    a{1} = X_batch;
    
    for i = 1:num_layers
        z{i} = a{i} * W{i} + repmat(b{i}, size(a{i}, 1), 1);
        if i == num_layers
            a{i+1} = z{i}; 
        else
            a{i+1} = max(z{i}, 0);  
        end
    end
    
    y_pred = a{end};
    

    loss = mean((y_pred - y_batch).^2);
    for i = 1:num_layers
        loss = loss + lambda * sum(W{i}(:).^2); 
    end
    

    delta = cell(1, num_layers);
    delta{num_layers} = 2 * (y_pred - y_batch) / batch_size;
    
    for i = num_layers-1:-1:1
        delta{i} = (delta{i+1} * W{i+1}') .* (z{i} > 0); 
    end
    
  
    dW = cell(1, num_layers);
    db = cell(1, num_layers);
    
    for i = 1:num_layers
        dW{i} = a{i}' * delta{i} + 2 * lambda * W{i};
        db{i} = sum(delta{i}, 1);
    end
    

    t = iteration;
    
    for i = 1:num_layers
        [W{i}, m_W{i}, v_W{i}] = adam_update_inline(W{i}, dW{i}, m_W{i}, v_W{i}, learning_rate, beta1, beta2, epsilon, t);
        [b{i}, m_b{i}, v_b{i}] = adam_update_inline(b{i}, db{i}, m_b{i}, v_b{i}, learning_rate, beta1, beta2, epsilon, t);
    end
    

    if mod(iteration, 1000) == 0
        fprintf('Iteration %d: Loss = %.6f\n', iteration, loss);
    end
end


a = X_poly;
for i = 1:num_layers
    z = a * W{i} + repmat(b{i}, size(a, 1), 1);
    if i == num_layers
        a = z;  
    else
        a = max(z, 0);  
    end
end
y_pred_final = a;

% Denormalizeeee
y_pred_final = y_pred_final * std(y) + mean(y);


mserr = mean((y_pred_final - y).^2);
fprintf('Mean Squared Error (MSE): %.6f\n', mserr);

ss_res = sum((y - y_pred_final).^2);
ss_tot = sum((y - mean(y)).^2);
r_squared = 1 - (ss_res / ss_tot);
fprintf('R-squared (R^2): %.6f\n', r_squared);


fprintf('Sample predictions (first 5):\n');
disp([y(1:5), y_pred_final(1:5)]);


figure;
scatter(y, y_pred_final);
hold on;
plot([min(y), max(y)], [min(y), max(y)], 'r--');
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs Predicted Values');
hold off;

figure;
plot(1:length(y), y, 'b-', 'DisplayName', 'Actual Values');
hold on;
plot(1:length(y), y_pred_final, 'r--', 'DisplayName', 'Predicted Values');
xlabel('Data Index');
ylabel('Values');
title('Actual vs Predicted Values');
legend show;
grid on;
hold off;


