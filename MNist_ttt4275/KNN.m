function predictions = KNN(train_set, train_label, test_set,K)
    n_testImg = length(test_set);
    neighbors = train_set;
    predictions = zeros(1,n_testImg);
    for test = 1:n_testImg
        predictions(1,test) = NN(neighbors,train_label,test_set(test,:),K);
    end
end