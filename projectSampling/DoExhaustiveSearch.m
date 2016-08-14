
% data paths
paths = { './data/hetrec2011-delicious-2k', ...
          './data/hetrec2011-lastfm-2k', ...
          './data/hetrec2011-movielens-2k-v2', ...
          './data/MovieLens/ml-10m', ...
          './data/MovieLens/ml-20m', ...
         };
% rank , top_t, knn
rank = 200;
knn = 1000;
top_t = 1000;
for n = 1: length(paths)
    path = paths{n};
    load(fullfile(path,'UserItemTag.mat'));
    load(fullfile(path,'Posts.mat'));
    % CP Decomposition
    sp_tensor = sptensor([UserItemTag(:,1),UserItemTag(:,2),UserItemTag(:,3)],Posts);
    CP = cp_als(sp_tensor,rank);
    User = zeros(size(CP.u{1}));
    Item = zeros(size(CP.u{2}));
    Tag = zeros(size(CP.u{3}));
    lambda = CP.lambda;
    for i = 1 : rank
        User(:,i) = CP.u{1}(:,i) * lambda(i); 
        Item(:,i) = CP.u{2}(:,i) * lambda(i); 
        Tag(:,i)  = CP.u{3}(:,i) * lambda(i);
    end
    save(fullfile(path,'User.mat'),'User');
    save(fullfile(path,'Item.mat'),'Item');
    save(fullfile(path,'Tag.mat'),'Tag');
    % exhaustive search 
    [topValue, fullTime, topIndexes] = exactSearchThreeOrderTrensor(User',Item',Tag',top_t);
    % save the result
    save(fullfile(path,'topValue.mat'),'topValue');
    save(fullfile(path,'fullTime.mat'),'fullTime');
    save(fullfile(path,'topIndexes.mat'),'topIndexes');
    % Query Search
    [valueQuery, timeQuery] = queryFullSearch(User',Item',Tag',knn);
    save(fullfile(path,'valueQuery.mat'),'valueQuery');
    save(fullfile(path,'timeQuery.mat'),'timeQuery');
end
