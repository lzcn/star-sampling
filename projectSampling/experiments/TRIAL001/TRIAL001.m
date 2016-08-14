% data_path: directory data set in.
% out_dir: out put directory
% samples: eg: power(10,:0,6)
% top_t: eg: power(10,:0,3)
% budget: the meaning of budget(i,j): 
%    the budget for top_t(j) under samplie numbers samples[i]
% turn: run how many turns
function TRIAL001(data_path, out_dir, ...
                budget, samples, top_t, ...
                turn, draw)

    % set variables
    % if draw picture for accuracy and time
    Variables.draw = draw;
    % the full time  exhaustive search cots
    Variables.fullTime = 0;
    % dataName for data set
    Variables.dataName = '';
    Variables.topValue = [];
    Variables.top_t = top_t;
    Variables.varSize = [size(samples,2),size(Variables.top_t,2)];
    Variables.samples = samples;
    Variables.budget = budget;
    Variables.turn = turn;
    Variables.out_dir = out_dir;
    %% ml-10m
    Variables.dataName = 'ml-10m';
    ml_10m_path = fullfile(data_path,'MovieLens','ml-10m');
    load(fullfile(ml_10m_path,'User.Mat'));
    load(fullfile(ml_10m_path,'Movie.Mat'));
    load(fullfile(ml_10m_path,'Tag.Mat'));
    load(fullfile(ml_10m_path,'topValue.Mat'));
    load(fullfile(ml_10m_path,'fullTime.Mat'));
    
    Variables.A = User;
    Variables.B = Movie;
    Variables.C = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    
    oneDataSet(Variables);
    %% ml-20m
    Variables.dataName = 'ml-20m';
    ml_20m_path = fullfile(data_path,'MovieLens','ml-20m');
    load(fullfile(ml_20m_path,'User.Mat'));
    load(fullfile(ml_20m_path,'Movie.Mat'));
    load(fullfile(ml_20m_path,'Tag.Mat'));
    load(fullfile(ml_20m_path,'topValue.Mat'));
    load(fullfile(ml_20m_path,'fullTime.Mat'));
 
    Variables.A = User;
    Variables.B = Movie;
    Variables.C = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    
    oneDataSet(Variables);

    %% ml-2k
    Variables.dataName = 'ml-2k';
    ml_2k_path = fullfile(data_path, 'hetrec2011-movielens-2k-v2');
    load(fullfile(ml_2k_path,'User.Mat'));
    load(fullfile(ml_2k_path,'Movie.Mat'));
    load(fullfile(ml_2k_path,'Tag.Mat'));
    load(fullfile(ml_2k_path,'topValue.Mat'));
    load(fullfile(ml_2k_path,'fullTime.Mat'));

    Variables.A = User;
    Variables.B = Movie;
    Variables.C = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;   

    oneDataSet(Variables);
    %% lastfm 
    Variables.dataName = 'lastfm';
    lastfm_path = fullfile(data_path, 'hetrec2011-lastfm-2k');
    load(fullfile(lastfm_path,'User.Mat'));
    load(fullfile(lastfm_path,'Artist.Mat'));
    load(fullfile(lastfm_path,'Tag.Mat'));
    load(fullfile(lastfm_path,'topValue.Mat'));
    load(fullfile(lastfm_path,'fullTime.Mat'));
 
    Variables.A = User;
    Variables.B = Artist;
    Variables.C = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
     
    oneDataSet(Variables);
     
    %% delicious
    Variables.dataName = 'delicious';
    delicious_path = fullfile(data_path, 'hetrec2011-delicious-2k');
    load(fullfile(delicious_path,'User.Mat'));
    load(fullfile(delicious_path,'Url.Mat'));
    load(fullfile(delicious_path,'Tag.Mat'));
    load(fullfile(delicious_path,'topValue.Mat'));
    load(fullfile(delicious_path,'fullTime.Mat'));
 
    Variables.A = User;
    Variables.B = Url;
    Variables.C = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    
    oneDataSet(Variables);

end

% V.recall(i,j): the recall for finding top_t[j] under samples[i]
% V.recall(:,j): recall-samples for finding top_t[j]
% V.times(i,j):  the time cots for finding top_t[j] under samples[i]
% V.times(:,j):  time-samples for finding top_t[j]

function [ diamond, central, extension ] = initVar(Variables)
    varSize = [length(Variables.samples),length(Variables.top_t)];
    diamond.recall   = zeros(varSize);
    diamond.times    = zeros(varSize);
    central.recall   = zeros(varSize);
    central.times    = zeros(varSize);
    extension.recall = zeros(varSize);
    extension.times  = zeros(varSize);
end

function [ diamond, central, extension ] = oneSampling(Variables)
    % initialization for veriables
    [ diamond, central, extension ] = initVar(Variables);
    % the matrices
    A = Variables.A;
    B = Variables.B;
    C = Variables.C;
    for TurnNum = 1:Variables.turn
        [ diamondTemp, centralTemp, extensionTemp ] = initVar(Variables);
        for i = 1 : length(Variables.samples)
            s = Variables.samples(i);
            for j = 1 : length(Variables.top_t)
                % budget
                tp = Variables.budget(i,j);
                % top-t
                t = Variables.top_t(j);
                % top-t value
                value = Variables.topValue(t);
                [dValue, dTime, ~] = diamondSampling(A',B,C,tp,s,t);
                [cValue, cTime, ~] = centralSampling(A,B,C,tp,s,t);
                [eValue, eTime, ~] = extensionSampling(A,B,C,tp,s,t);
                diamondTemp.recall(i,j) = sum(dValue(1:t) >= value)/t;
                centralTemp.recall(i,j) = sum(cValue(1:t) >= value)/t;
                extensionTemp.recall(i,j) = sum(eValue(1:t) >= value)/t;
                extensionTemp.times(i,j) = eTime;
                diamondTemp.times(i,j) = dTime;
                centralTemp.times(i,j) = cTime;
            end
        end
        diamond.recall = diamond.recall + diamondTemp.recall;
        diamond.times = diamond.times + diamondTemp.times;
        central.recall = central.recall + centralTemp.recall;
        central.times = central.times + centralTemp.times;
        extension.recall = extension.recall + extensionTemp.recall;
        extension.times = extension.times + extensionTemp.times;
    end
    % save the result
    out_dir = fullfile(Variables.out_dir,Variables.dataName);
    if ~isdir(out_dir)
        mkdir(out_dir);
    end
    diamond.recall = diamond.recall/Variables.turn;
    diamond.times = diamond.times/Variables.turn;
    diamond_recall = diamond.recall;
    diamond_times = diamond.times;
    save(fullfile(out_dir,'diamond_recall.mat'),'diamond_recall');
    save(fullfile(out_dir,'diamond_times.mat'),'diamond_times');
    
    central.recall = central.recall/Variables.turn;
    central.times = central.times/Variables.turn;
    central_recall = central.recall;
    central_times = central.times;
    save(fullfile(out_dir,'central_recall.mat'),'central_recall');
    save(fullfile(out_dir,'central_times.mat'),'central_times');
    
    extension.recall = extension.recall/Variables.turn;
    extension.times = extension.times/Variables.turn;
    extension_recall = extension.recall;
    extension_times = extension.times;
    save(fullfile(out_dir,'extension_recall.mat'),'extension_recall');
    save(fullfile(out_dir,'extension_times.mat'),'extension_times');
end

function drawTimeFig(Variables, diamond, central, extension)
    samples = Variables.samples;
    fullTime = Variables.fullTime;
    titlename = Variables.titlename;
    out_dir = Variables.out_dir;
    h = figure; hold on; title(titlename); 
    xlabel('log_{10}Samples'); 
    ylabel('log_{10}T(sec)');
    plot(log10(samples),log10(fullTime*ones(size(samples))),'b','LineWidth',2);
    plot(log10(samples),log10(diamond.times(:,end)),'r','LineWidth',2); 
    plot(log10(samples),log10(central.times(:,end)),'--g','LineWidth',2);
    plot(log10(samples),log10(extension.times(:,end)),'--c','LineWidth',2);
    legend('exhaustive','diamond','central','extension',4);
    saveas(h,fullfile(out_dir,[titlename,'.pdf']));
    close(h);
end

function drawRecallFig(Variables, diamond, central, extension)
    samples = Variables.samples;
    titlename = Variables.titlename;
    out_dir = Variables.out_dir;
    % draw recall-samples diamond vs central
    h = figure; hold on; title([titlename,'-diamond-central']);
    xlabel('log_{10}Samples'); 
    ylabel('Accuracy');
    axis([log10(samples(1)) log10(samples(end)) 0 1.1]);
    c = ['r','b','k','g', 'c'];
    desc = cell(size(Variables.top_t,2)*2, 1);
    for i = 1:size(Variables.top_t,2)
        top_t = Variables.top_t(i);
        plot(log10(samples),diamond.recall(:,i),c(i),'LineWidth',2);
        desc{2*i-1} = ['diamond:t=',num2str(top_t)];
        plot(log10(samples),central.recall(:,i),['--',c(i)],'LineWidth',2); 
        desc{ 2*i } = ['central:t=',num2str(top_t)];
    end
    legend(desc,4);  
    saveas(h,fullfile(out_dir,['diamond-central-',titlename,'.pdf']));
    close(h);
    % draw recall-samples diamond vs extension
    h = figure; hold on; title([titlename,'-diamond-extension']);
    xlabel('log_{10}Samples'); 
    ylabel('recall');
    axis([log10(samples(1)) log10(samples(end)) 0 1.1]);
    c = ['r','b','k','g', 'c'];
    desc = cell(size(Variables.top_t,2)*2, 1);
    for i = 1:size(Variables.top_t,2)
        top_t = Variables.top_t(i);
        plot(log10(samples),diamond.recall(:,i),c(i),'LineWidth',2);
        desc{2*i-1} = ['diamond:t=',num2str(top_t)];
        plot(log10(samples),extension.recall(:,i),['--',c(i)],'LineWidth',2); 
        desc{ 2*i } = ['extension:t=',num2str(top_t)];
    end
    legend(desc,4);  
    saveas(h,fullfile(out_dir,['diamond-extension-',titlename,'.pdf']));
    close(h);
    % draw recall-samples extension vs central
    h = figure; hold on; title([titlename,'-extension-central']);
    xlabel('log_{10}Samples'); 
    ylabel('recall');
    axis([log10(samples(1)) log10(samples(end)) 0 1.1]);
    c = ['r','b','k','g', 'c'];
    desc = cell(size(Variables.top_t,2)*2, 1);
    for i = 1:size(Variables.top_t,2)
        top_t = Variables.top_t(i);
        plot(log10(samples),extension.recall(:,i),c(i),'LineWidth',2); 
        desc{2*i-1} = ['extension:t=',num2str(top_t)];
        plot(log10(samples),central.recall(:,i),['--',c(i)],'LineWidth',2); 
        desc{ 2*i } = ['central:t=',num2str(top_t)];
    end
    legend(desc,4);  
    saveas(h,fullfile(out_dir,['extension-central-', titlename,'.pdf']));
    close(h);
end

function oneDataSet(Variables)
    [ diamond, central, extension ] = oneSampling(Variables);
    if Variables.draw
        dataName = Variables.dataName;
        Variables.titlename = ['time-samples-',dataName];
        drawTimeFig(Variables, diamond, central, extension);
        Variables.titlename = ['recall-samples-',dataName];
        drawRecallFig(Variables, diamond, central, extension);
    end
end
