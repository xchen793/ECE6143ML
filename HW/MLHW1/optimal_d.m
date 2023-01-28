
function min_loss = optimal_d(x,y)
%
% Finds an order d = D-1 for polynomial fit to the data 
% with minimized training loss
%
%    function [D, errT, err] = cross_val(x,y,model,xT,yT)
%
% x = vector of input scalars for training
% y = vector of output scalars for training
% D = the order plus one of the polynomial being fit with 
% minimum training loss
% err = average squared loss on training
% model = vector of polynomial parameter coefficients
% errT = average squared loss on testing
%

% randomly splitting the data-set
trainX = x((1:0.6*length(x)),:);
trainY = y((1:0.6*length(x)),:);
testX = x((0.6*length(x)+1:end),:);
testY = y((0.6*length(x)+1:end),:);


    function err = poly_err(x,y,D)
    %
    % compute average squared loss of polynomial fitting with D-1 order
    % on dataset (x,y)
    %

    xx = zeros(length(x),D);
    for i=1:D
      xx(:,i) = x.^(D-i);
    end
    model = pinv(xx)*y;
    err   = (1/(2*length(x)))*sum((y-xx*model).^2);

    end



% plot the training and testing risk across various choices of d
q = (2:1:10)';
clf
plot(q,poly_err(trainX,trainY,q),'r');
hold on
plot(q,poly_err(testX,testY,q),'g');
hold off
% for loop to get minimum testing loss and optimal order d 
d = 0;
min_loss = inf;
for i=1:length(q)
    if (min_loss > poly_err(testX,testY,q))
        min_loss = poly_err(testX,testY,q); 
        d = q(i);
    end
end
% plot f(x;θ) for the best choice of d

end


