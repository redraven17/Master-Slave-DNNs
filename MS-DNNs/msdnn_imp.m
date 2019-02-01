        %% //////////////// Classification of Hand Gestures \\\\\\\\\\\\\\\\%

        real_data = rand(500,200); %% input data
        data_dnn = []; %% data being used for classification
        split= 0.5; %% splitting ratio for the data
        act=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]; %% classes
        repN = 10; %% no. of samples per class
        for k=1:3 %% k=1=>Master,k=2=>Slave,k=3=>Single
        %Training and Testing Data
        ft = 20;
        if (k==1)
            y2 = [ones(repN*length(act)/2,1);2*ones(repN*length(act)/2,1)];
            num_labels = 2;
            for j=0:length(act)-1
                data_dnn = [data_dnn;real_data((j*repN)+1:(j*repN)+(split*repN),:)];
                y2 = [y2;y2((j*repN)+1:(j*repN)+(split*repN),:)];
            end
            disp(['Master Classifier Active....'])
        elseif (k==2)
            data_dnn = data_dnn( (repN*length(act)/2)+1 : repN*length(act),:); %% for static classification
            y2 = [ones(repN,1);2*ones(repN,1);3*ones(repN,1);4*ones(repN,1);5*ones(repN,1)];
            num_labels = 5;
            for j=0:length(act)-1
                data_dnn = [data_dnn;real_data((j*repN/2)+1:(j*repN/2)+(split*repN),:)];
                y2 = [y2;y2((j*repN/2)+1:(j*repN/2)+(split*repN),:)];
            end
            disp(['Slave Classifier Active....'])
        else
            data_dnn = data_rep;
            y2 = [ones(repN,1);2*ones(repN,1);3*ones(repN,1);4*ones(repN,1);5*ones(repN,1);6*ones(repN,1);7*ones(repN,1);8*ones(repN,1);9*ones(repN,1);10*ones(repN,1)];
            num_labels = 10;
            for j=0:length(act)-1
                data_dnn = [data_dnn;real_data((j*repN)+1:(j*repN)+(split*repN),:)];
                y2 = [y2;y2((j*repN)+1:(j*repN)+(split*repN),:)];
            end
            disp(['Single Classifier Active....'])
        end
        data_comb = [data_comb;data_dnn];y_comb = [y_comb;y2];
        nfold=3; %% no. of folds for cv-partitioning
        cv = cvpartition(y2,'kfold',nfold);
        clear TestCsel;
        for bb=1:nfold
            TestCsel(:,bb) = cv.test(bb);
        end
        acc1_n = [];acc2_n = [];acc3_n = [];
%         iter = [10:15:150];
cost_synth = [];
        for i=1:length(iter)
        for cc=1:nfold
            clear w q y_test2 test_data train_data   
            test_data= find(TestCsel(:,cc));
            train_data= find(TestCsel(:,cc)==0);
            y_test2= y2(test_data);
            y_train2 = y2(train_data);
            data_train2 = data_dnn(train_data,:);
            data_test2 = data_dnn(test_data,:);

            [q,w] = pca_feature_reduction(data_train2,ft,data_test2); 
            factor = 1.5;
            %Initializing DNN Parameters%
            input_layer_size  = size(q',1); 
            hidden_layer_size1 = floor(factor*size(q',1));
            hidden_layer_size2 = floor(factor*size(q',1));
            hidden_layer_size3 = floor(factor*size(q',1));
            hidden_layer_size4 = floor(factor*size(q',1));
            m = size(q, 1);
            lambda = 0;

            %Randomly initializing weights of the layers%
            epsilon = 0.1;
            Theta1 = rand(hidden_layer_size1, 1 + input_layer_size) * 2 * epsilon - epsilon;
            Theta2 = rand(hidden_layer_size2, 1 + hidden_layer_size1) * 2 * epsilon - epsilon;
            Theta3 = rand(hidden_layer_size3, 1 + hidden_layer_size2) * 2 * epsilon - epsilon;
            Theta4 = rand(hidden_layer_size4, 1 + hidden_layer_size3) * 2 * epsilon - epsilon;
            Theta5 = rand(num_labels, 1 + hidden_layer_size4) * 2 * epsilon - epsilon;
            nn_params = [Theta1(:) ; Theta2(:) ; Theta3(:) ; Theta4(:) ; Theta5(:)];
            % weights_dnn = [weights_dnn,nn_params];
            %nn_params = weights_dnn(:,subj_no*nfold);

            J = dnntrain(nn_params, input_layer_size, hidden_layer_size1,hidden_layer_size2, hidden_layer_size3,hidden_layer_size4,...
                       num_labels, q, y_train2, lambda);

            %Training Neural Network%
            options = optimset('MaxIter', 150);

            train1 = @(p) dnntrain(p, ...
                                input_layer_size, ...
                                hidden_layer_size1,hidden_layer_size2, hidden_layer_size3,hidden_layer_size4,...
                                num_labels, q, y_train2, lambda);

            [nn_params, cost] = fmincg(train1, nn_params, options);
                for k = length(cost):150
                cost(k) = cost(length(cost));
                end

            
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

            %Predicting the outputs%
            a1 = [ones(size(w,1),1), w];
            z2 = Theta1*a1';
            a2 = sigmoid(z2);
            a2 = [ones(1,size(w,1));a2];
            z3 = Theta2*a2;
            a3 = sigmoid(z3');
            a3 = [ones(size(w,1),1),a3];
            z4 = Theta3*a3';
            a4 = sigmoid(z4);
            a4 = [ones(1,size(w,1));a4];
            z5 = Theta4*a4;
            a5 = sigmoid(z5');
            a5 = [ones(size(w,1),1),a5];
            z6 = Theta5*a5';
            b = sigmoid(z6);
            [M,I] = max(b);
            p2 = I';
            z2 = confusionmat(p2,y_test2);
            %figure;plotconfmat(z2);
            acc1 = 100*sum(diag(z2))/sum(sum(z2));
            acc1_n(cc) = acc1;
            
            
            %SVM Classification%
            [q,w] = pca_feature_reduction(data_train2,ft,data_test2);
            Mdl = fitcecoc(q,y_train2,'Coding', 'onevsall'); %%RG%%
            result = predict(Mdl,w);
            %result = multisvm(q,y_train2,w);
            z1 = confusionmat(result,y_test2);
            %figure;plotconfmat(z1);
            acc2 = 100*sum(diag(z1))/sum(sum(z1));
            acc2_n(cc) = acc2;

            %KNN Classification
            [q,w] = pca_feature_reduction(data_train2,ft,data_test2);
            Mdl = fitcknn(q,y_train2,'NumNeighbors',3,'Standardize',1); %%RG%%
            class = predict(Mdl,w);
            % class = knnclassify(w,q,y_train2,3);
            z3 = confusionmat(class,y_test2);
            %figure;plotconfmat(z3);
            acc3 = 100*sum(diag(z3))/sum(sum(z3));
            acc3_n(cc) = acc3;
        end
        acc1 = mean(acc1_n);
        acc2 = mean(acc2_n);
        acc3 = mean(acc3_n);
        end
    cost_synth = [cost_synth,cost];
        end
    % %------------------------------------------------------------------------%
    % % save('weights_dnn.mat','weights_dnn');


