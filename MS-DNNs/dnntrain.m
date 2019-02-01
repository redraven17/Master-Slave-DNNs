function [J grad] = dnntrain(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size1,hidden_layer_size2, hidden_layer_size3,hidden_layer_size4,...
                                   num_labels, ...
                                   X, y, lambda)


Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + hidden_layer_size1 * (input_layer_size + 1)):(hidden_layer_size1 * (input_layer_size + 1) + hidden_layer_size2*(1+hidden_layer_size1))), ...
                 hidden_layer_size2, (hidden_layer_size1 + 1));

Theta3 = reshape(nn_params(1+ hidden_layer_size1 * (input_layer_size + 1) + hidden_layer_size2*(1+hidden_layer_size1):(hidden_layer_size1 * (input_layer_size + 1) + hidden_layer_size2*(1+hidden_layer_size1) + hidden_layer_size3*(1+hidden_layer_size2))), ...
                 hidden_layer_size3, (hidden_layer_size2 + 1));

Theta4 = reshape(nn_params((1 + (hidden_layer_size1 * (input_layer_size + 1) + hidden_layer_size2*(1+hidden_layer_size1)) + hidden_layer_size3*(1+hidden_layer_size2):(hidden_layer_size1 * (input_layer_size + 1) + hidden_layer_size2*(1+hidden_layer_size1)) + hidden_layer_size3*(1+hidden_layer_size2) + hidden_layer_size4*(1+hidden_layer_size3))), ...
                 hidden_layer_size4, (hidden_layer_size3 + 1));
             
Theta5 = reshape(nn_params((1+ (hidden_layer_size1 * (input_layer_size + 1) + hidden_layer_size2*(1+hidden_layer_size1) + hidden_layer_size3*(1+hidden_layer_size2)) + hidden_layer_size4*(1+hidden_layer_size3):end)), ...
                 num_labels, (hidden_layer_size4 + 1));
           
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
A3 = [ones(size(Z3, 1), 1) sigmoid(Z3)];
Z4 = A3*Theta3';
A4 = sigmoid(Z4);
A4 = [ones(size(Z4, 1), 1) sigmoid(Z4)];
Z5 = A4*Theta4';
A5 = [ones(size(Z5, 1), 1) sigmoid(Z5)];
Z6 = A5*Theta5';
A6 = sigmoid(Z6);

H = A6;

%Cost Computation
penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)) + sum(sum(Theta3(:, 2:end).^2, 2)) + sum(sum(Theta4(:, 2:end).^2, 2)) + sum(sum(Theta5(:, 2:end).^2, 2)));
J = (1/m)*sum(sum((-Y).*log(H) - (1-Y).*log(1-H), 2));
J = J + penalty;

%Back Propogation
z1 = [ones(size(Z2, 1), 1) Z2];
g1 = sigmoid(z1).*(1-sigmoid(z1));  %Computing Sigmoid Gradient
z2 = [ones(size(Z3, 1), 1) Z3];
g2 = sigmoid(z2).*(1-sigmoid(z2));  %Computing Sigmoid Gradient
z3 = [ones(size(Z4, 1), 1) Z4];
g3 = sigmoid(z3).*(1-sigmoid(z3));  %Computing Sigmoid Gradient
z4 = [ones(size(Z5, 1), 1) Z5];
g4 = sigmoid(z4).*(1-sigmoid(z4));  %Computing Sigmoid Gradient


Sigma6 = A6 - Y;
Sigma5 = (Sigma6*Theta5 .* g4);
Sigma5 = Sigma5(:,2:end);
Sigma4 = (Sigma5*Theta4 .* g3);
Sigma4 = Sigma4(:,2:end);
Sigma3 = (Sigma4*Theta3 .* g2);
Sigma3 = Sigma3(:,2:end);
Sigma2 = (Sigma3*Theta2 .* g1);
Sigma2 = Sigma2(:,2:end);

Delta_1 = Sigma2'*A1;
Delta_2 = Sigma3'*A2;
Delta_3 = Sigma4'*A3;
Delta_4 = Sigma5'*A4;
Delta_5 = Sigma6'*A5;

%Theta Gradients
Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
Theta3_grad = Delta_3./m + (lambda/m)*[zeros(size(Theta3,1), 1) Theta3(:, 2:end)];
Theta4_grad = Delta_4./m + (lambda/m)*[zeros(size(Theta4,1), 1) Theta4(:, 2:end)];
Theta5_grad = Delta_5./m + (lambda/m)*[zeros(size(Theta5,1), 1) Theta5(:, 2:end)];


grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:) ; Theta4_grad(:) ; Theta5_grad(:)];
end
             
             