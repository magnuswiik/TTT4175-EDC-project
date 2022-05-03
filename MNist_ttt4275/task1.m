load("data_all.mat")

data_all = load("data_all.mat");
startTime = clock;
dataChuncks = chunckData(data_all,10);
predsKort = KNN(dataChuncks(1),7);
confMatKort = calculateConfusionMatrix(predsKort,dataChuncks(1).testlab);
errorRateKort = calculateErrorRate(confMatKort);
train_set = dataChuncks(1).trainv;
train_label = dataChuncks(1).trainlab;
test_set = dataChuncks(1).testv;
preds = KNN(train_set, train_label, test_set, 7);

%preds = KNN(data_all,7);
%confMat = calculateConfusionMatrix(preds,data_all.testlab);
%errorRate = calculateErrorRate(confMat);

endTime = clock;
timeTaken = endTime-startTime;

function errorRate = calculateErrorRate(confMat)
    n_preds = sum(confMat,'all');
    correct = 0;
    for i = 1:length(confMat)
        correct = correct + confMat(i,i);
    end
    errorRate = round(100*(1-correct/n_preds),2);
end

function confusionMatrix = calculateConfusionMatrix(predictions,targets)
    confusionMatrix = zeros(10,10);
    for i = 1:length(predictions)
        confusionMatrix(predictions(i)+1,targets(i)+1) = confusionMatrix(predictions(i)+1,targets(i)+1) + 1;
    end
end

function dataChuncks = chunckData(data_all,n_chuncks)
    for i = 1:n_chuncks
        dataChuncks(i)=struct('trainv',0,'trainlab',0,'testv',0,'testlab',0);

        startTrain = (i-1)*length(data_all.trainv)/n_chuncks+1;
        stopTrain = i*length(data_all.trainv)/n_chuncks;
        startTest = (i-1)*length(data_all.testv)/n_chuncks+1;
        stopTest = i*length(data_all.testv)/n_chuncks;

        dataChuncks(i).trainv=data_all.trainv(startTrain:stopTrain,:);
        dataChuncks(i).trainlab=data_all.trainlab(startTrain:stopTrain);
        dataChuncks(i).testv=data_all.testv(startTest:stopTest,:);
        dataChuncks(i).testlab=data_all.testlab(startTest:stopTest);
    end
end