function [J grad] = anntrain(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

                               
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

%Feed-Forward Propogation
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2*Theta2';
A3 = sigmoid(Z3);
H = A3;

%Cost Computation
penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
J = (1/m)*sum(sum((-Y).*log(H) - (1-Y).*log(1-H), 2));
J = J + penalty;

%Back Propogation
z = [ones(size(Z2, 1), 1) Z2];
g = sigmoid(z).*(1-sigmoid(z));  %Computing Sigmoid Gradient

Sigma3 = A3 - Y;
Sigma2 = (Sigma3*Theta2 .* g);
Sigma2 = Sigma2(:,2:end);

Delta_1 = Sigma2'*A1;
Delta_2 = Sigma3'*A2;

%Theta Gradients
Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];


grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
             
             