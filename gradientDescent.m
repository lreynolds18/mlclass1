function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    A = 0;
    for i=1:m
        A = A + theta(1,1)+theta(2,1)*X(i,2)-y(i,1);
    end
    B = 0;
    for i=1:m
        B = B + (theta(1,1)+theta(2,1)*X(i,2)-y(i,1))*X(i,2);
    end
    temp0 = theta(1,1)-alpha*A/m
    temp1 = theta(2,1)-alpha*B/m
    theta(1,1) = temp0;
    theta(2,1) = temp1;
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
