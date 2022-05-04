clear
close all

%% Fetch data
load("data_all.mat")
data_all = load("data_all.mat");
test_set = data_all.testv;
test_lab = data_all.testlab;

%% Clustering
startTime = clock;

% Variables
digits = 10;
M = 64;
cluster_labels = repelem([0 1 2 3 4 5 6 7 8 9]', M);

clusters = clustering(data_all.trainv, data_all.trainlab, digits, M);

%% Predicting
preds = KNN(clusters, cluster_labels, data_all.testv, 1);

% Timer stop
endTime = clock;
timeTaken = endTime-startTime;

%% Calculate confusion matrix and error rate
confMat = calculateConfusionMatrix(preds,data_all.testlab);
errorRate = calculateErrorRate(confMat);
confusionchart(confMat);

for i = 1:length(preds)
    if preds(i) ~= test_lab(i)
        img = test_set(i,:);
        plotImg(img, preds(i), test_lab(i), "false");
        w = waitforbuttonpress;
    else
        img = test_set(i,:);
        plotImg(img, preds(i), test_lab(i), "correct");
        w = waitforbuttonpress;
    end
end

