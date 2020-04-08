function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z = X*theta;

h_x = sigmoid(z);

log_h_x_1 = log(h_x);
log_h_x_2 = log(1-h_x);
log_h_x_1_y = (-y)'*log_h_x_1;
log_h_x_2_y = (1 - y)'*log_h_x_2;
sum_cost = log_h_x_1_y - log_h_x_2_y;

J = sum_cost/m + (lambda * sum(theta(2:end).^2))/(2*m);

grad= sum((h_x - y).*X)/m;

grad_reg = grad'(2:end) + (lambda * theta(2:end))/m;
grad(2:end) = grad_reg;


% =============================================================

end
