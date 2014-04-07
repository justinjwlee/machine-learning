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

vectorisedVar = X * theta;
J = - 1/m * ((log(sigmoid(vectorisedVar)))' * y + (log(ones(size(vectorisedVar), 1)-sigmoid(vectorisedVar)))' * ...
										 (ones(size(y),1) - y)) + lambda/(2*m) * (theta(2:end)' * theta(2:end));
for j=1:size(X,2)
	if (j==1)
	grad(j)=(1/m)*sum((sigmoid(X*theta)-y).*X(:,j));
	else
	grad(j)=(1/m)*sum((sigmoid(X*theta)-y).*X(:,j))+ (1/m)*(theta(j)*lambda);
	endif
endfor
% =============================================================

end
