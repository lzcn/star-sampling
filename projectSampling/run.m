% run demo
%% moivelens
% read data userID , itemID ,tagID ,posts
filename = 'data\movielens.dat';
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
fprintf(fp,'a');
disp('Doing exact search!');
% exact search
tic;
% true_value = exact_search(CP.u{1}',CP.u{2}',CP.u{3}');
t_exact = toc;
% diamond sampling
disp('Doing diamonf sampling!');
t = zeros(4,1);
recall1 = zeros(4,1);
recall10 = zeros(4,1);
recall1h = zeros(4,1);
recall1k = zeros(4,1);
tic;
[~, values] = ds1k(CP.u{1}',CP.u{2},CP.u{3});
t(1)  = toc;
recall1(1) = (values(1) >= true_value(1))/1.0;
recall10(1) = sum(values(1:10) >= true_value(10))/10.0;
recall1h(1) = sum(values(1:100) >= true_value(100))/100.0;
recall1k(1) = sum(values(1:100) >= true_value(1000))/1000.0;
tic;
[~, values] = ds10k(CP.u{1}',CP.u{2},CP.u{3});
t(2) = toc;
recall1(2) = (values(1) >= true_value(1))/1.0;
recall10(2) = sum(values(1:10) >= true_value(10))/10.0;
recall1h(2) = sum(values(1:100) >= true_value(100))/100.0;
recall1k(2) = sum(values(1:100) >= true_value(1000))/1000.0;
tic;
[~, values] = ds100k(CP.u{1}',CP.u{2},CP.u{3});
t(3) = toc;

recall1(3) = (values(1) >= true_value(1))/1.0;
recall10(3) = sum(values(1:10) >= true_value(10))/10;
recall1h(3) = sum(values(1:100) >= true_value(100))/100;
recall1k(3) = sum(values(1:100) >= true_value(1000))/1000;

tic;
[~, values] = ds1m(CP.u{1}',CP.u{2},CP.u{3});
t(4) = toc;

recall1(4) = (values(1) >= true_value(1))/1.0;
recall10(4) = sum(values(1:10) >= true_value(10))/10;
recall1h(4) = sum(values(1:100) >= true_value(100))/100;
recall1k(4) = sum(values(1:100) >= true_value(1000))/1000;

h = figure;
hold on;
plot(1:4,t(:),'r');
plot([1,4],[t_exact,t_exact]);
saveas(h,'sample-time-1.jpg');
h = figure;
hold on;
plot(1:4,recall1(:),'r');
plot(1:4,recall10(:),'b');
plot(1:4,recall1h(:),'k');
plot(1:4,recall1k(:),'r');
legend('t=1','t=10','t=100','t=1000');
saveas(h,'sample-recall-1.jpg');

%%
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
% exact search
tic;
true_value = exact_search(CP.u{1}',CP.u{2}',CP.u{3}');
t_exact = toc;
% diamond sampling
disp('Doing diamonf sampling!');
t = zeros(4,1);
recall1 = zeros(4,1);
recall10 = zeros(4,1);
recall1h = zeros(4,1);
recall1k = zeros(4,1);
tic;
[~, values] = ds1k(CP.u{1}',CP.u{2},CP.u{3});
t(1)  = toc;
recall1(1) = (values(1) >= true_value(1))/1.0;
recall10(1) = sum(values(1:10) >= true_value(10))/10.0;
recall1h(1) = sum(values(1:100) >= true_value(100))/100.0;
recall1k(1) = sum(values(1:100) >= true_value(1000))/1000.0;
tic;
[~, values] = ds10k(CP.u{1}',CP.u{2},CP.u{3});
t(2) = toc;
recall1(2) = (values(1) >= true_value(1))/1.0;
recall10(2) = sum(values(1:10) >= true_value(10))/10.0;
recall1h(2) = sum(values(1:100) >= true_value(100))/100.0;
recall1k(2) = sum(values(1:100) >= true_value(1000))/1000.0;
tic;
[~, values] = ds100k(CP.u{1}',CP.u{2},CP.u{3});
t(3) = toc;

recall1(3) = (values(1) >= true_value(1))/1.0;
recall10(3) = sum(values(1:10) >= true_value(10))/10;
recall1h(3) = sum(values(1:100) >= true_value(100))/100;
recall1k(3) = sum(values(1:100) >= true_value(1000))/1000;

tic;
[~, values] = ds1m(CP.u{1}',CP.u{2},CP.u{3});
t(4) = toc;

recall1(4) = (values(1) >= true_value(1))/1.0;
recall10(4) = sum(values(1:10) >= true_value(10))/10;
recall1h(4) = sum(values(1:100) >= true_value(100))/100;
recall1k(4) = sum(values(1:100) >= true_value(1000))/1000;

h = figure;
hold on;
plot(1:4,t(:),'r');
plot([1,4],[t_exact,t_exact]);
saveas(h,'sample-time-2.jpg');
h = figure;
hold on;
plot(1:4,recall1(:),'r');
plot(1:4,recall10(:),'b');
plot(1:4,recall1h(:),'k');
plot(1:4,recall1k(:),'r');
legend('t=1','t=10','t=100','t=1000');
saveas(h,'sample-recall-2.jpg');