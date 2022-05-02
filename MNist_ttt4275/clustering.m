
function clusters = clustering(trainv, trainlab, digits, M)
    clusters = zeros(M*digits, 28*28);
    [sorted_trainlab, index] = sort(trainlab);
    sorted_trainv = zeros(size(trainv));
    cluster_tags = repelem([0 1 2 3 4 5 6 7 8 9]', M);

    for i = 1:length(trainv)
        sorted_trainv(i,:) = trainv(index(i),:);
    end
    stop = 1;
    for i = 0:9
        start = stop;
        while stop < length(trainv) && sorted_trainlab(stop+1) == i
            stop = stop + 1;
        end

        partition = sorted_trainv(start:stop,:);

        [~, C_i] = kmeans(partition,M);
        clusters(i*M+1:(i+1)*M,:) = C_i;
    end
end