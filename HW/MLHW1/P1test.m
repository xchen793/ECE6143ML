load('problem1.mat')
% finding a good value for $d$
full_dataset_errors = [];
full_dataset_models = {};
for d=1:9
    [err , model] = polyreg (x , y , d );
    title(sprintf ('d = %d ', d ) ) ;
    full_dataset_errors(d) = err;
    full_dataset_models{d} = model;

end
close all ;
% cross validation
cross_validation_train_errors = [];
cross_validation_test_errors = [];
cross_validation_models = {};
ind = crossvalind ( 'Kfold ' , 500, 2);
xtrain = x(ind == 1 );
ytrain = y(ind == 1 );
xtest = x(ind == 2 );
ytest = y( ind == 2 );
for d=1:40
    [ err , model , errT ] = polyreg ( xtrain , ytrain , d, xtest , ytest );
    title ( sprintf ( 'd = %d ' , d ) ) ;
    cross_validation_train_errors(d) = err ;
    cross_validation_test_errors(d) = errT ;
    cross_validation_models{d} = model ;
end
clf ;
hold on ;
plot(cross_validation_test_errors , 'b');
plot(cross_validation_train_errors , 'r ');
[m , i] = min(cross_validation_test_errors);
plot (i, cross_validation_test_errors(i) , 'bx ' ) ;
xlabel ( 'degree o f polynomial + 1' );
ylabel ( 'error ' );
legend( ' test ' , ' train ' );
title ( 'Cross−Validation test error ' );
%print('p1c_vs_plot.png','−dpng' );