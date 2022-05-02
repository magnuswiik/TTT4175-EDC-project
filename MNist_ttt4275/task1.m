load("data_all.mat")

data_all = load("data_all.mat");
dataChuncks = chunckData(data_all,10);
train_set = dataChuncks(1).trainv;
train_label = dataChuncks(1).trainlab;
test_set = dataChuncks(1).testv;
preds = KNN(train_set, train_label, test_set, 7);

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