load("data_all.mat")

data_all = load("data_all.mat");
dataChuncks = chunckData(data_all,10);
preds = KNN(dataChuncks(1),7);

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
function pred = NN(neighbors,targets,testImg,K)
    distances = zeros(2,length(neighbors));
    for n = 1:length(neighbors)
        distances(1,n)=targets(n,1);
        neighbor = neighbors(n,:);
        distances(2,n)=norm(neighbor-testImg);
    end
    distances = sortrows(distances.',2).';
    pred = mode(distances(1,1:K));
end
function predictions = KNN(data,K)
    n_testImg = length(data.testv);
    neighbors = data.trainv;
    predictions = zeros(1,n_testImg);
    for test = 1:n_testImg
        predictions(1,test) = NN(neighbors,data.trainlab,data.testv(test,:),K);
    end
end