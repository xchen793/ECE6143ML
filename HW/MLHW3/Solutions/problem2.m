function problem2
clc;
%load the dataset
load C:\Users\zhran\Desktop\xka\ML\MLHW3\dataset;
n = size(X,1);
x_train = randsample(n,n/2);
x_test = setdiff(1:n, x_train);
xtr = X(x_train,:);
xte = X(x_test,:);
ytr = Y(x_train,:);
yte = Y(x_test,:);

d_max = 50;
d_error = zeros(1, d_max);
%classify using polynomials/*****1
for d = 1:d_max
    %train
    [nsv, alpha, bias] = svc(xtr, ytr, 'poly',inf);
    %predict
    Y_pred = svcoutput(xtr, ytr, xte, 'poly', alpha, bias);
    %complete test error
    d_error(d) = svcerror(xtr, ytr. xte, yte, 'poly', alpha, bias);
end
 
%plot errors(different polynomial degree)
f = figure(1);
clf(f);
plot(1:d_max, d_error);
print ( f, '-depsc', 'poly.eps') ;


%classify using rbfs
sigmas = .1:.1 :2 ;
sigma_error = zeros(1, numel(sigmas)) ;
for i=1:numel(sigmas)
    %train
    [nsv, alpha, bias] = svc(xtr, ytr, 'rbf', inf) ;
    %predict
    Y1_pred = svcoutput(xtr, ytr, xte, 'rbf', alpha, bias) ;
    %compute test error
    sigma_error(i) = svcerror(xtr, ytr, xte, yte, 'rbf', alpha, bias);
end
%plot errors
f = figure(1);
clf(f);
plot (sigmas, sigma_error) ;
print ( f, '-depsc', 'rbf.eps') ;





