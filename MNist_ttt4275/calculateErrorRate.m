function errorRate = calculateErrorRate(confMat)
    n_preds = sum(confMat,'all');
    correct = 0;
    for i = 1:length(confMat)
        correct = correct + confMat(i,i);
    end
    errorRate = round(100*(1-correct/n_preds),2);
end