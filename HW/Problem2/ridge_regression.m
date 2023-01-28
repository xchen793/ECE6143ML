function [Error, Model, ErrorTest] = ridge_regression(Xtrain, Ytrain, lbd, Xtest, Ytest)

x2 = Xtrain' * Xtrain ;

Model = (Xtrain' * Xtrain + lbd * eye ( size (x2))) \ Xtrain' * Ytrain;

Error = (1/(2 * size (Xtrain, 1))) * sum((Ytrain - Xtrain * Model).^2);

if (nargin == 5)
    ErrorTest = (1/(2 * size (Xtest, 1))) * sum((Ytest - Xtest * Model).^2);
end
    
end