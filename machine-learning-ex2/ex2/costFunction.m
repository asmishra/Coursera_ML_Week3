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

%Step 1 : Create a hypothesis vector based on X and theta given to the program
h = zeros(size(y));
for i = 1:m
  X_row = X(i,:)'; %Make it a Nx1 vector, theta is also a NX1 vector
  h(i) = sigmoid(theta' * X_row);
end

%Step 2 : Calculate cost based on hyothesis vs Y (for logistic regression/classification)
J = (sum(-y.*log(h) - (1-y).*log(1-h)))/m; %some vector math to calculate it in one go

%Step 3 : Calculate the partial derivatives per feature
for i = 1:size(X,2) %N times
  grad(i) = (sum((h - y) .* X(:,i)))/m;
end


% =============================================================

end
