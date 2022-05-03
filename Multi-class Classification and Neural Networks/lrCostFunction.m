function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
h = sigmoid(X * theta);
theta(1) = 0;

J = 1 / m * ((-y' * log(h)) - (1 - y)' * log(1 - h));
J += lambda / (2 * m) * theta' * theta;

grad = 1 / m * (h - y)' * X;
grad += lambda / m * theta';

grad = grad(:);

end
