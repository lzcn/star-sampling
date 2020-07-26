
% data paths
paths = { './data/hetrec2011-movielens-2k-v2', ...
          './data/MovieLens/ml-10m', ...
          './data/MovieLens/ml-20m', ...
          './data/hetrec2011-delicious-2k', ...
          './data/hetrec2011-lastfm-2k', ...
         };
knn = 1000;
top_t = 1000;
for n = 1: length(paths)
    path = paths{n};
    load(fullfile(path,'User.mat'));
    load(fullfile(path,'Item.mat'));
    load(fullfile(path,'Tag.mat'));
    % exhaustive search
    [topValue, fullTime, topIndexes] = exactSearchThreeOrderTrensor(User,Item,Tag,top_t);
    % save the result
    save(fullfile(path,'topValue.mat'),'topValue');
    save(fullfile(path,'fullTime.mat'),'fullTime');
    save(fullfile(path,'topIndexes.mat'),'topIndexes');
    % Query Search
    [valueQuery, timeQuery] = queryFullSearch(User,Item,Tag,knn);
    save(fullfile(path,'valueQuery.mat'),'valueQuery');
    save(fullfile(path,'timeQuery.mat'),'timeQuery');
end
