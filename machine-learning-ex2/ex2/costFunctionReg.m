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

%Step 1 : Create a hypothesis vector based on X and theta given to the program
h = zeros(size(y));
for i = 1:m
  X_row = X(i,:)'; %Make it a Nx1 vector, theta is also a NX1 vector
  h(i) = sigmoid(theta' * X_row);
end

%Step 2 : Calculate cost based on hyothesis vs Y (for logistic regression/classification)
J = (sum(-y.*log(h) - (1-y).*log(1-h)))/m; %some vector math to calculate it in one go

%Add regularization - don't regularize theta0
regularization_sum = 0;
for i = 2:length(theta)
  regularization_sum += theta(i)^2;
end
regularization_sum *= lambda;
regularization_sum /= (2*m);

J = J + regularization_sum; %increase the cost by regularized amount

%Step 3 : Calculate the partial derivatives per feature
for i = 1:size(X,2) %N times
  grad(i) = (sum((h - y) .* X(:,i)))/m;
end

%Add regularization - don't regularize theta0
grad_temp = lambda/m * theta; 
grad_temp(1) = 0; %As theta0 must not be regularized
grad = grad + grad_temp;

% =============================================================

end
