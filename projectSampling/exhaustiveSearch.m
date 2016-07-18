% exhaustive search
clc; clear all;
%% delicious
path = './data/hetrec2011-delicious-2k';
load(fullfile(path,'UserUrlTag.mat'));
load(fullfile(path,'tagWeight.mat'));
% CP decomposition
sp_tensor = sptensor([UserUrlTag(:,1),UserUrlTag(:,2),UserUrlTag(:,3)],tagWeight);
% use cp_als to do decomposition
rank = 200;
CP = cp_als(sp_tensor,rank);
lambda = CP.lambda;
User = zeros(size(CP.u{1}));
Url = zeros(size(CP.u{2}));
Tag = zeros(size(CP.u{3}));

for i = 1 : rank
    User(:,i) = CP.u{1}(:,i)*lambda(i); 
    Url(:,i) = CP.u{2}(:,i)*lambda(i); 
    Tag(:,i) = CP.u{3}(:,i)*lambda(i);
end
% save result 
save(fullfile(path,'User.mat'),'User');
save(fullfile(path,'Url.mat'),'Url');
save(fullfile(path,'Tag.mat'),'Tag');
% exhaustive search for top 1000 largest value
[topValue, fullTime, topIndexes] = exactSearchThreeOrderTrensor(User',Url',Tag',1000);
% save the result
save(fullfile(path,'topValue.mat'),'topValue');
save(fullfile(path,'fullTime.mat'),'fullTime');
save(fullfile(path,'topIndexes.mat'),'topIndexes');
% finde the knn value of each query
KNN = 1000;
[valueQuery, timeQuery] = queryFullSearch(User',Url',Tag',KNN);
% save result
save(fullfile(path,'valueQuery.mat'),'valueQuery');
save(fullfile(path,'timeQuery.mat'),'timeQuery');

%% lastfm
clear all;
path = './data/hetrec2011-lastfm-2k';
load(fullfile(path,'UserArtistTag.mat'));
load(fullfile(path,'weight.mat'));
% CP decomposition
sp_tensor = sptensor([UserArtistTag(:,1),UserArtistTag(:,2),UserArtistTag(:,3)],weight);
% use cp_als to do decomposition
rank = 200;

CP = cp_als(sp_tensor,rank);
lambda = CP.lambda;
User = zeros(size(CP.u{1}));
Artist = zeros(size(CP.u{2}));
Tag = zeros(size(CP.u{3}));

for i =1:rank
    User(:,i) = CP.u{1}(:,i)*lambda(i); 
    Artist(:,i) = CP.u{2}(:,i)*lambda(i); 
    Tag(:,i) = CP.u{3}(:,i)*lambda(i);
end

save(fullfile(path,'User.mat'),'User');
save(fullfile(path,'Artist.mat'),'Artist');
save(fullfile(path,'Tag.mat'),'Tag');
[topValue, fullTime, topIndexes] = exactSearchThreeOrderTrensor(User',Artist',Tag',1000);
save(fullfile(path,'topValue.mat'),'topValue');
save(fullfile(path,'fullTime.mat'),'fullTime');
save(fullfile(path,'topIndexes.mat'),'topIndexes');
KNN = 1000;
[valueQuery, timeQuery] = queryFullSearch(User',Artist',Tag',KNN);
% save result
save(fullfile(path,'valueQuery.mat'),'valueQuery');
save(fullfile(path,'timeQuery.mat'),'timeQuery');

%% movielens 2k
clear all;
path = './data/hetrec2011-movielens-2k-v2';
load(fullfile(path,'UserMovieTag.mat'));
load(fullfile(path,'rating.mat'));
% CP decomposition
sp_tensor = sptensor([UserMovieTag(:,1),UserMovieTag(:,2),UserMovieTag(:,3)],rating);
% use cp_als to do decomposition
rank = 200;
CP = cp_als(sp_tensor,rank);
lambda = CP.lambda;
User = zeros(size(CP.u{1}));
Movie = zeros(size(CP.u{2}));
Tag = zeros(size(CP.u{3}));

for i = 1:rank
    User(:,i) = CP.u{1}(:,i)*lambda(i); 
    Movie(:,i) = CP.u{2}(:,i)*lambda(i); 
    Tag(:,i) = CP.u{3}(:,i)*lambda(i);
end

save(fullfile(path,'User.mat'),'User');
save(fullfile(path,'Movie.mat'),'Movie');
save(fullfile(path,'Tag.mat'),'Tag');
[topValue, fullTime, topIndexes] = exactSearchThreeOrderTrensor(User',Movie',Tag',1000);
save(fullfile(path,'topValue.mat'),'topValue');
save(fullfile(path,'fullTime.mat'),'fullTime');
save(fullfile(path,'topIndexes.mat'),'topIndexes');
KNN = 1000;
[valueQuery, timeQuery] = queryFullSearch(User',Movie',Tag',KNN);
% save result
save(fullfile(path,'valueQuery.mat'),'valueQuery');
save(fullfile(path,'timeQuery.mat'),'timeQuery');
%% ml - 10m
clear all;
path = './data/MovieLens/ml-10m';
load(fullfile(path,'UserMovieTag.mat'));
load(fullfile(path,'rating.mat'));
% CP decomposition
sp_tensor = sptensor([UserMovieTag(:,1),UserMovieTag(:,2),UserMovieTag(:,3)],rating);
% use cp_als to do decomposition
rank = 200;
CP = cp_als(sp_tensor,rank);
lambda = CP.lambda;
User = zeros(size(CP.u{1}));
Movie = zeros(size(CP.u{2}));
Tag = zeros(size(CP.u{3}));

for i = 1:rank
    User(:,i) = CP.u{1}(:,i)*lambda(i); 
    Movie(:,i) = CP.u{2}(:,i)*lambda(i); 
    Tag(:,i) = CP.u{3}(:,i)*lambda(i);
end

save(fullfile(path,'User.mat'),'User');
save(fullfile(path,'Movie.mat'),'Movie');
save(fullfile(path,'Tag.mat'),'Tag');
[topValue, fullTime, topIndexes] = exactSearchThreeOrderTrensor(User',Movie',Tag',1000);
save(fullfile(path,'topValue.mat'),'topValue');
save(fullfile(path,'fullTime.mat'),'fullTime');
save(fullfile(path,'topIndexes.mat'),'topIndexes');
KNN = 1000;
[valueQuery, timeQuery] = queryFullSearch(User',Movie',Tag',KNN);
% save result
save(fullfile(path,'valueQuery.mat'),'valueQuery');
save(fullfile(path,'timeQuery.mat'),'timeQuery');

%% ml - 20m
clear all;
path = './data/MovieLens/ml-20m';
load(fullfile(path,'UserMovieTag.mat'));
load(fullfile(path,'rating.mat'));
% CP decomposition
sp_tensor = sptensor([UserMovieTag(:,1),UserMovieTag(:,2),UserMovieTag(:,3)],rating);
% use cp_als to do decomposition
rank = 200;
CP = cp_als(sp_tensor,rank);
lambda = CP.lambda;
User = zeros(size(CP.u{1}));
Movie = zeros(size(CP.u{2}));
Tag = zeros(size(CP.u{3}));

for i = 1:rank
    User(:,i) = CP.u{1}(:,i)*lambda(i); 
    Movie(:,i) = CP.u{2}(:,i)*lambda(i); 
    Tag(:,i) = CP.u{3}(:,i)*lambda(i);
end

save(fullfile(path,'User.mat'),'User');
save(fullfile(path,'Movie.mat'),'Movie');
save(fullfile(path,'Tag.mat'),'Tag');
[topValue, fullTime, topIndexes] = exactSearchThreeOrderTrensor(User',Movie',Tag',1000);
save(fullfile(path,'topValue.mat'),'topValue');
save(fullfile(path,'fullTime.mat'),'fullTime');
save(fullfile(path,'topIndexes.mat'),'topIndexes');
KNN = 1000;
[valueQuery, timeQuery] = queryFullSearch(User',Movie',Tag',KNN);
% save result
save(fullfile(path,'valueQuery.mat'),'valueQuery');
save(fullfile(path,'timeQuery.mat'),'timeQuery');