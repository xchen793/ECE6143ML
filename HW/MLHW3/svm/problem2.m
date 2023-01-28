function problem2
clear;
%load and split the dataset  
load C:\Users\zhran\Desktop\xka\ML\MLHW3\dataset;
dataset = importdata('C:\Users\zhran\Desktop\xka\ML\MLHW3\dataset.mat');
X = dataset.X; Y = dataset.Y;
n = size(X,1);
indices = randperm(n);
split = 0.5 * n;
xtr = X(indices(1:split),:);
xte = X(indices(split + 1:length(X)),:);
Y(Y==0) = -1; % replace Y=0 with Y=-1
ytr = Y(indices(1:split),:);
yte = Y(indices(split + 1:length(X)),:);

ker = 'poly';
global p1;
global p2;

%classify using linear
if strcmp(ker, 'linear') == 1
    C = 1:10:100;
    p1 = 1; 
    p2 = 0;
    errorsC = zeros(1, numel(C));
    for d = 1:numel(C)
        %train
       [~, alpha, bias] = svc(xtr, ytr, ker, C(d));
        %predict
        Y_pred = svcoutput(xtr, ytr, xte, ker, alpha, bias);
        %complete test error
        %%errorsC(d) = svcerror(xtr, ytr, xte, yte, ker, alpha, bias);
        misClassificationCount = sum(((ytr .* Y_pred) == -1));
        errorsC(d) = misClassificationCount * 100 / length(ytr);
    end
    %plot errors(different polynomial degree)
    f = figure(1);
    clf(f);
    plot(1:numel(C), errorsC, '--x');
    %svcplot(xtr, ytr ,'linear',alpha,bias);
    title('SVM Classification with linear Kernel');
    xlabel('C');
    ylabel('Classification Error (%)');
    print ( f, '-depsc', 'linear.eps') ;
end 

%classify using polynomials
if strcmp(ker, 'poly') == 1
    C = [0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000];
    order = 1:10;
    errorsC = zeros(numel(order), numel(C));
    for i = 1:length(order)
        p1 = i;
        for d = 1:numel(C)
            %train
            [nsv, alpha, bias] = svc(xtr, ytr, ker, C(d));
            %predict
            Y_pred = svcoutput(xtr, ytr, xte, ker, alpha, bias);
            %complete test error
            errorsC(i,d) = svcerror(xtr, ytr, xte, yte, ker, alpha, bias);
        end   
    end
    % C == inf
    err_inf = zeros (numel(order),1) ;
    for i = 1:length(order)
        p1 = i;
        %train
        [nsv, alpha, bias] = svc(xtr, ytr, ker, C(d));
        %predict
        Y_pred = svcoutput(xtr, ytr, xte, ker, alpha, bias);
        %complete test error
        err_inf(i,1) = svcerror(xtr, ytr, xte, yte, ker, alpha, bias);  
    end 
    errorsC = [err_inf errorsC];
    C = [inf, C];
    %plot errors(different polynomial degree) 
    f = figure(2);
    clf(f);
    x = [inf -5:1:6];
    bar3(errorsC);
    set(gca, 'xticklabel', x);
    title('SVM Classification with Poly Kernel');
    xlabel('C (log 10 base)');
    ylabel('Degree of the polynomial');
    zlabel('Error');
    print (f, '-depsc', 'poly.eps') ;
end


%classify using rbfs
if strcmp(ker, 'rbf') == 1
    sigmas =[0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000];
    C = [0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000];
    sigma_error = zeros(length(sigmas), length(C)) ;
    nsv = zeros(length(sigmas), length(C));
    for i=1:length(sigmas)
        p1 = i;
        for j = 1:length(C)
            %train
            [nsv, alpha, bias] = svc(xtr, ytr, ker, C(j)) ;
            %predict
            Y_pred = svcoutput(xtr, ytr, xte, ker, alpha, bias) ;
            %compute test error
            sigma_error(i, j) = svcerror(xtr, ytr, xte, yte, ker, alpha, bias);       
        end
    end
    
     % C == inf
    err_inf = zeros (numel(sigmas),1) ;
    for i = 1:length(sigmas)
        p1 = i;
        %train
        [nsv, alpha, bias] = svc(xtr, ytr, ker, inf);
        %predict
        Y_pred = svcoutput(xtr, ytr, xte, ker, alpha, bias);
        %complete test error
        err_inf(i,1) = svcerror(xtr, ytr, xte, yte, ker, alpha, bias);  
    end 
    sigma_error = [err_inf sigma_error];
    C = [inf, C];
    %plot errors
    f = figure(3);
    clf(f);
    %plot (sigmas, sigma_error) ;
    bar3( sigma_error);
    x = [inf -5:1:6];
    y = -5:1:6;
    set(gca, 'xticklabel', x,  'yticklabel', y);
    zlabel('Error');
    xlabel('C (log 10 base)');
    ylabel('Sigma (log 10 base)');
    title('SVM classification with RBF kernel');
    print (f, '-depsc', 'rbf.eps') ;
end




