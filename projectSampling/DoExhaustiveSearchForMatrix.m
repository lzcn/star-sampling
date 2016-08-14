% exhaustive search
paths = {'./data/hetrec2011-lastfm-2k/matrix',...
        './data/hetrec2011-movielens-2k-v2/matrix',...
        './data/MovieLens/ml-10m/matrix',...
        './data/MovieLens/ml-20m/matrix'};
for n = 1:length(paths)
    path = paths{n};
    load(fullfile(path,'UserItem.mat'));
    load(fullfile(path,'Posts.mat'));
    % CP decomposition
    sp_tensor = sptensor([UserItem(:,1),UserItem(:,2)],Posts);
    % use cp_als to do decomposition
    rank = 200;
    CP = cp_als(sp_tensor,rank);
    lambda = CP.lambda;
    User = zeros(size(CP.u{1}));
    Item = zeros(size(CP.u{2}));
    for i =1:rank
        User(:,i) = CP.u{1}(:,i)*lambda(i);
        Item(:,i) = CP.u{2}(:,i)*lambda(i);
    end
    save(fullfile(path,'User.mat'),'User');
    save(fullfile(path,'Item.mat'),'Item');
    [topEuclidean, TimeEuclidean] = ExhaustiveSearchMatrix(User',Item',1000,'Euclidean');
    [topCosine, TimeCosine] = ExhaustiveSearchMatrix(User',Item',1000,'Cosine');
    save(fullfile(path,'topEuclidean.mat'),'topEuclidean');
    save(fullfile(path,'TimeEuclidean.mat'),'TimeEuclidean');
    save(fullfile(path,'topCosine.mat'),'topCosine');
    save(fullfile(path,'TimeCosine.mat'),'TimeCosine');
end