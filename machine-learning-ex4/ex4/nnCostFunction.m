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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Y = zeros(m, num_labels);
for i=1:m
  index = y(i);
  Y(i, index) = 1;
endfor

X = [ones(m, 1) X];
first_layer = sigmoid(X*Theta1');
first_layer = [ones(m, 1), first_layer];
second_layer = sigmoid(first_layer*Theta2');

log_h_x_1 = log(second_layer);
log_h_x_2 = log(1-second_layer);
log_h_x_1_y = (-Y).*log_h_x_1;
log_h_x_2_y = (1 - Y).*log_h_x_2;
sum_cost = log_h_x_1_y - log_h_x_2_y;

regularisation = (lambda * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))))/(2*m);

J= sum(sum(sum_cost))/m + regularisation;

% -------------------------------------------------------------

DELTA_1 = zeros(size(Theta1));
DELTA_2 = zeros(size(Theta2));
for t = 1:m
  a_1 = X(t, 1:end);
  z_2 = a_1*Theta1';
  a_2 = sigmoid(z_2);
  a_2_with_bias = [1, a_2];
  z_3 = a_2_with_bias*Theta2';
  a_3 = sigmoid(z_3);
  delta_3 = a_3 - Y(t, 1:end);
  Theta2WithoutBias = Theta2(:, 2:end);
  delta_2 = delta_3*Theta2WithoutBias.*sigmoidGradient(z_2);
  DELTA_2 = DELTA_2 + a_2_with_bias.*delta_3';
  DELTA_1 = DELTA_1 + a_1.*delta_2';
endfor

regularization2 = Theta2;
regularization2(:, 2:end)  = (lambda*regularization2(:, 2:end))/m;
regularization2(1:end, 1) = regularization2(1:end, 1) * 0;
regularization1 = Theta1;
regularization1(:, 2:end)  = (lambda*regularization1(:, 2:end))/m;
regularization1(1:end, 1) = regularization1(1:end, 1) * 0;

Theta2_grad = DELTA_2/m + regularization2;
Theta1_grad = DELTA_1/m + regularization1;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
