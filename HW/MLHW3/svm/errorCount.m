function [percentMisClassified, nsv] = errorCount(xTrain, yTrain, xTest, yTest, ker, C)
    [nsv, alpha, bias] = svc(xTrain, yTrain, ker, C);
    % building predictions
    L = length(xTest);
    prediction = zeros(L, 1);
    for i = 1:L
        xi = xTest(i, :);
        totals = zeros(L, 1);
        for j = 1:L
            totals(j) = svkernel(ker, xTest(j, :), xi);
        end
        prediction(i) = sign(sum(alpha .* yTest .* totals) + bias);
    end

    misClassificationCount = sum(((yTest .* prediction) == -1));
    percentMisClassified = misClassificationCount * 100 / length(yTest);
end
