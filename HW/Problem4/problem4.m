function problem4()
    %eliminate all other codes and data
    close all;
    
    %load data from assignment given 
    load('dataset4.mat');
    
    % assign stepsize, tolerance and related theta 
    % 5 sets of stepsize and tolerances are implemented for observation
    % set1 (Epsilon:0.001,Eta:0.05)
    % set2 (Epsilon:0.002,Eta:0.1)
    % set3 (Epsilon:0.002,Eta:0.5)
    % set4 (Epsilon:0.001,Eta:0.5)
    % set5 (Epsilon:0.001,Eta:5)
    
    c = 0;
    maxrange = 10000000;
    step = 5;
    tolerance = 0.001;
    theta = rand(size(X, 2),1);
    
    % assign vector
    risks = [];
    errors = [];
    pt = 2*tolerance + theta;
    
    % loop and obtain c
    while norm(theta - pt) >= tolerance
        if c > maxrange
            break;
        end
        
        %Loop for errors and risks with function risk
        rthetaXY = risk(X, Y, theta);
        f = 1./(1+exp(-X*theta));
        f(f>= 0.5) = 1;
        f(f<0.5) = 0;
        erroor = sum(f~=Y)/length(Y) ;
        
        %Print Iteration Error and Risk in Command line
        fprintf('Iteration:%d,Error:%0.4f,Risk:%0.4f\n',c,erroor ,rthetaXY);
        risks = cat(1,risks ,rthetaXY);
        errors = cat(1,errors ,erroor);
        
        % Find G with function gradient
        pt = theta ; 
        Grafound = gradient (X, Y, theta ) ;
        theta = theta - step*Grafound;
        c = c + 1;
    end
    
    %Plot gigures
    figure
    plot(1:c, errors ,'r',1:c,risks,'b');
    
    title('Error(red) & Risk(blue) vs Iterations');
    disp('theta');
    disp(theta)
    
    x=0:0.01:1;
    y = (-theta(3) - theta( 1 ).*x)/theta (2);
    
    figure
    plot(x,y,'b'); 
    hold on;
    plot(X(:,1),X(:,2),'.');
    title('Resulted Linear Decision Boundary for 2D X data');
end

% Find risk
function R = risk(x,y,theta)
    f = 1./(1+exp(-x*theta));
    r = (y-1).*log(1-f )-y.*log(f); r(isnan(r)) = 0;
    R = mean(r);
end


% Find gradient
function gra = gradient(x,y ,theta)
    yreform = repmat(y,1,size(x,2)) ;
    f = 1./(1+exp(-x* theta)) ;
    freform = repmat(f,1,size (x ,2)) ;
    d = x.*repmat (exp(-x*theta ),1,size (x,2));
    
    gra = (1-yreform).*( x - d.* freform) - yreform.*d.*freform;
    gra = sum(gra);
    gra = gra/length(y);
    gra = gra';
end
