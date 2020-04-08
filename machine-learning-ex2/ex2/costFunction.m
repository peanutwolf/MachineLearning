function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

z = X*theta;

h_x = sigmoid(z);

log_h_x_1 = log(h_x);
log_h_x_2 = log(1-h_x);
log_h_x_1_y = (-y)'*log_h_x_1;
log_h_x_2_y = (1 - y)'*log_h_x_2;
sum_cost = log_h_x_1_y - log_h_x_2_y;

J = sum_cost/m;

grad = sum((h_x - y).*X)/m;



% =============================================================

end
