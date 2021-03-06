function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);

[J, grad] = costFunction(theta, X, y);

theta(1) = 0;
J += lambda / (2 * m) * theta' * theta;
grad += lambda / m * theta';

end
