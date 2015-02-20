function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 
% 1/2m * sum(ho(x^(i))-y^(i))^2
j = power(theta(1,1)+theta(2,1)*X(1,2)-y(1,1), 2);

J = 0;
for i=1:m
    J = J + power(theta(1,1)+theta(2,1)*X(i,2)-y(i,1), 2);
end
% j = syssum(power(theta(1,1)*X(i,2)-y(i,1), 2), i, [1;m]);
J = J/(2*m);

% =======================================================================
end
