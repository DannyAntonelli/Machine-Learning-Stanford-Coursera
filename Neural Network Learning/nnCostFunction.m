function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                    num_labels, (hidden_layer_size + 1));

m = size(X, 1);

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];

z2 = X * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];

z3 = a2 * Theta2';
h = sigmoid(z3);

Y_mat = (1:num_labels) == y;

J = 1 / m * sum(sum(-Y_mat .* log(h) - (1 - Y_mat) .* log(1 - h)));

for i = 1:m
    a1 = X(i, :)';

    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    
    z3 = Theta2 * a2;
    h = sigmoid(z3);
    
    delta3 = h - Y_mat(i, :)';
    delta2 = ((Theta2' * delta3) .* [1; sigmoidGradient(z2)])(2:end);
    
    Theta1_grad += (delta2 * a1');
    Theta2_grad += (delta3 * a2');
endfor

Theta1_grad /= m;
Theta2_grad /= m;

J += lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

Theta1_grad += lambda / m * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad += lambda / m * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

grad = [Theta1_grad(:); Theta2_grad(:)];

end
