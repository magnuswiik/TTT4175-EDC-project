load("data_all.mat")

data_all = load("data_all.mat");

digits = 10;
M = 64;
cluster_tags = repelem([0 1 2 3 4 5 6 7 8 9]', M);
clusters = clustering(data_all.trainv, data_all.trainlab, digits, M);

preds = KNN(clusters, cluster_tags, data_all.testv, 7);

