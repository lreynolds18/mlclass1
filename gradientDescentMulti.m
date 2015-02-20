function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters
    temp = zeros(size(theta,1),1);

    for i=1:size(theta,1)
        A = 0;
        for j=1:size(X,1)
            A = A + (((X(j,:)*theta)-y(j,1))*X(j,i));
        end
        temp(i,1)=theta(i,1)-alpha*A/m;
    end
    theta = temp;
    J_history(iter) = computeCostMulti(X, y, theta);
    
end

end
