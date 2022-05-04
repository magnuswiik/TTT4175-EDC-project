function pred = NN(train_set,train_label,testImg,K)
    [num_trainx, num_trainy] = size(train_set);
    distances = zeros(2,num_trainx);
    for n = 1:num_trainx
        distances(1,n)=train_label(n,1);
        neighbor = train_set(n,:);
        distances(2,n)=norm(neighbor-testImg);
    end
    distances = sortrows(distances.',2).';
    pred = mode(distances(1,1:K));
end