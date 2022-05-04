function confusionMatrix = calculateConfusionMatrix(predictions,targets)
    confusionMatrix = zeros(10,10);
    for i = 1:length(predictions)
        confusionMatrix(targets(i)+1,predictions(i)+1) = confusionMatrix(targets(i)+1,predictions(i)+1) + 1;
    end
end