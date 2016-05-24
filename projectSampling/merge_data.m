function setup_data(filename1,filename2,outfile)
% function setup_data(filename1,filename2,outfile)
% inparam:
%   filename1 : the file contain userID , itemID , posts
%   filename2 : the file contain userID , itemID , tagID
% output:
%   outfile : the file contain userID , itemID , tagID
delimiter = '\t';
startRow = 2;
formatSpec = '%f%f%f%*s%[^\n\r]';
fileID = fopen(filename1,'r');
dataArray = textscan(fileID, ...
                     formatSpec, ...
                    'Delimiter', delimiter, ...
                    'EmptyValue' ,NaN,...
                    'HeaderLines' ,startRow-1, ...
                    'ReturnOnError', false);

fclose(fileID);
userID1 = dataArray{:, 1};
itemID1 = dataArray{:, 2};
posts = dataArray{:, 3};
fileID = fopen(filename2,'r');
dataArray = textscan(fileID, ...
                     formatSpec, ...
                    'Delimiter', delimiter, ...
                    'EmptyValue' ,NaN,...
                    'HeaderLines' ,startRow-1, ...
                    'ReturnOnError', false);
fclose(fileID);
userID2 = dataArray{:, 1};
itemID2 = dataArray{:, 2};
tagID = dataArray{:, 3};
merge(userID1,itemID1,posts,userID2,itemID2,tagID);
system(['rename' ' out_tmp.dat ' outfile]);
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
clear userID1 movieID1 rating userID2 movieID2 tagID;