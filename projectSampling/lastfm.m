addpath('bin'); clc;
% read data
filename = 'data\lastfm.dat';
delimiter = '\t';
startRow = 2;
formatSpec = '%f%f%f%f%[^\n\r]';

fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, ...
    'Delimiter', delimiter, ...
    'EmptyValue' ,NaN,...
    'HeaderLines' ,startRow-1, ...
    'ReturnOnError', false);
fclose(fileID);
userID = dataArray{:, 1};
itemID = dataArray{:, 2};
tagID = dataArray{:, 3};
posts = dataArray{:, 4};

% construct sparse tensor
sp_tensor = sptensor([userID,itemID,tagID],posts);
clearvars filename delimiter startRow formatSpec ;
clearvars fileID dataArray ans ;
clearvars tagID userID posts itemID;

% use cp_als to do decomposition
Rank = 200;
CP = cp_als(sp_tensor,Rank); 
lambda = CP.lambda;
A = zeros(size(CP.u{1}));
B = zeros(size(CP.u{2}));
C = zeros(size(CP.u{3}));
for i =1:Rank
    A(:,i) = CP.u{1}(:,i)*lambda(i); 
    B(:,i) = CP.u{2}(:,i)*lambda(i); 
    C(:,i) = CP.u{3}(:,i)*lambda(i);
end
clearvars sp_tensor CP lambda Rank;
% save the result
save('data\lastfm\A.mat','A');
save('data\lastfm\B.mat','B');
save('data\lastfm\C.mat','C');


KNN = 1000;
% do exhaustive search
A = A'; B = B'; C= C';
tic;
valueTrue = exact_search_three_order_tensor(A,B,C);
exactTime = toc;
% save the true value and time
save('data\lastfm\valueTrue.mat','valueTrue');
save('data\lastfm\exactTime.mat','exactTime');
% query exhaustive search
[valueQuery,timeQuery] = queryFullSearch(A,B,C,KNN);
% save result
save('data/lastfm/valueQuery.mat','valueQuery');
save('data/lastfm/timeQuery.mat','timeQuery');

