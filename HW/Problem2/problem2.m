clear all;
load ('problem2.mat');

% define cross validation train and test set errors
% define cross validation model

cvTrainE = [];
cvTestE = [];
cvModels = {};

% Import Kfold from bioinfomatics toolbox for dataset spliting

indice = crossvalind('Kfold',400,2);


% Slice the dataset into training and testing datasets with indice

Xtrain = x(indice == 1,:);
Ytrain = y(indice == 1);
Xtest = x(indice == 2,:);
Ytest = y(indice == 2);


d = 1;

% define lambda 
lbd = 0:0.5:2000;

% Loop and find d with regression function
for lambda=lbd
    [err, model, errT] = ridge_regression(Xtrain, Ytrain, lambda, Xtest, Ytest);
    cvTrainE(d) = err ;
    cvTestE(d) = errT ;
    cvModels{d} = model ;
    d = d + 1;
end

% Plot Training and Testing Dataset Errors

plot(lbd, cvTrainE, 'b' );
hold on
plot(lbd , cvTestE, 'r' );


%Find the min of Test Error
%Define counter and min
minimun = inf;
counting = 0;
for i=1:length(cvTestE)
    if (minimun > cvTestE(i))
        minimun = cvTestE(i);
        counting = i;
    end
end

% Plot Figure
plot(lbd(counting) , cvTestE(counting), 'r*') ;
xlabel ('Lambda Value');
ylabel ('Error');
legend('Training Set Error' , 'Testing Set Error');
