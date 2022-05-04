function dataChunks = chunkData(data_all,n_chuncks)
    for i = 1:n_chuncks
        dataChunks(i)=struct('trainv',0,'trainlab',0,'testv',0,'testlab',0);

        startTrain = (i-1)*length(data_all.trainv)/n_chuncks+1;
        stopTrain = i*length(data_all.trainv)/n_chuncks;
        startTest = (i-1)*length(data_all.testv)/n_chuncks+1;
        stopTest = i*length(data_all.testv)/n_chuncks;

        dataChunks(i).trainv=data_all.trainv(startTrain:stopTrain,:);
        dataChunks(i).trainlab=data_all.trainlab(startTrain:stopTrain);
        dataChunks(i).testv=data_all.testv(startTest:stopTest,:);
        dataChunks(i).testlab=data_all.testlab(startTest:stopTest);
    end
end