clc;
clear all;
CP.u{1} = ones(10,50);
CP.u{2} = ones(13,50);
CP.u{3} = ones(13,50);

% exact search
tic;
true_value = exact_search(CP.u{1}',CP.u{2}');
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
recall1k(1) = sum(values(1:1000) >= true_value(1000))/1000.0;
tic;
[~, values] = ds10k(CP.u{1}',CP.u{2},CP.u{3});
t(2) = toc;
recall1(2) = (values(1) >= true_value(1))/1.0;
recall10(2) = sum(values(1:10) >= true_value(10))/10.0;
recall1h(2) = sum(values(1:100) >= true_value(100))/100.0;
recall1k(2) = sum(values(1:1000) >= true_value(1000))/1000.0;
tic;
[~, values] = ds100k(CP.u{1}',CP.u{2},CP.u{3});
t(3) = toc;

recall1(3) = (values(1) >= true_value(1))/1.0;
recall10(3) = sum(values(1:10) >= true_value(10))/10;
recall1h(3) = sum(values(1:100) >= true_value(100))/100;
recall1k(3) = sum(values(1:1000) >= true_value(1000))/1000;

tic;
[index, values] = ds1m(CP.u{1}',CP.u{2},CP.u{3});
t(4) = toc;

recall1(4) = (values(1) >= true_value(1))/1.0;
recall10(4) = sum(values(1:10) >= true_value(10))/10;
recall1h(4) = sum(values(1:100) >= true_value(100))/100;
recall1k(4) = sum(values(1:1000) >= true_value(1000))/1000;

h = figure;
hold on;
plot(1:4,t(:),'r');
plot([1,4],[t_exact,t_exact]);
saveas(h,'sample-time-1.png');
h = figure;
hold on;
plot(1:4,recall1(:),'r');
plot(1:4,recall10(:),'b');
plot(1:4,recall1h(:),'k');
plot(1:4,recall1k(:),'r');
legend('t=1','t=10','t=100','t=1000');
saveas(h,'sample-recall-1.png');