% run demo
% moivelens
% read data userID , itemID ,tagID ,posts
filename = 'movielens.dat';
delimiter = '\t';
startRow = 2;
formatSpec = '%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, ...
                     formatSpec, ...
                    'Delimiter', delimiter, ...
                    'EmptyValue' ,NaN,...
                    'HeaderLines' ,startRow-1, ...
                    'ReturnOnError', false);

fclose(fileID);
userID = dataArray{:, 1};
itemID = dataArray{:, 2};
tagID = dataArray{:, 3};
posts = dataArray{:, 4};

% CP decomposition
Rank = 200;
subs = [userID,itemID,tagID];
sp_tensor = sptensor(subs,posts);
CP = cp_als(sp_tensor,Rank);

lambda = CP.lambda;
fp = fopen('movielens','w');
for i = 1 : size(CP.u,1)
    fprintf(fp,'%d\t',size(CP.u{i},1));
end
fprintf(fp,'\n%d\t\n',size(CP.u{i},2));
for i = 1 : size(CP.u,1)
    matrix = CP.u{i};
    for j = 1 : size(matrix,1)
        for k = 1: size(matrix,2)
            fprintf(fp,'%f\t',lambda(k)*matrix(j,k));
        end
        fprintf(fp,'\n');
    end
end
fclose(fp);


filename = 'lastfm.dat';
delimiter = '\t';
startRow = 2;
formatSpec = '%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, ...
                     formatSpec, ...
                    'Delimiter', delimiter, ...
                    'EmptyValue' ,NaN,...
                    'HeaderLines' ,startRow-1, ...
                    'ReturnOnError', false);

fclose(fileID);
userID = dataArray{:, 1};
itemID = dataArray{:, 2};
tagID = dataArray{:, 3};
posts = dataArray{:, 4};

% CP decomposition
Rank = 200;
subs = [userID,itemID,tagID];
sp_tensor = sptensor(subs,posts);
CP = cp_als(sp_tensor,Rank);
disp('Doing exact search!');


lambda = CP.lambda;
fp = fopen('lastfm','w');
for i = 1 : size(CP.u,1)
    fprintf(fp,'%d\t',size(CP.u{i},1));
end
fprintf(fp,'\n%d\t\n',size(CP.u{i},2));
for i = 1 : size(CP.u,1)
    matrix = CP.u{i};
    for j = 1 : size(matrix,1)
        for k = 1: size(matrix,2)
            fprintf(fp,'%f\t',lambda(k)*matrix(j,k));
        end
        fprintf(fp,'\n');
    end
end
fclose(fp);