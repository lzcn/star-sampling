function TRIAL001(data_path, out_dir, samples, budget, turn)

    % set data path
    ml_10m_path = fullfile(data_path,'MovieLens','ml-10m');
    ml_20m_path = fullfile(data_path,'MovieLens','ml-20m');
    ml_2k_path = fullfile(data_path, 'hetrec2011-movielens-2k-v2');
    lastfm_path = fullfile(data_path, 'hetrec2011-lastfm-2k');
    delicious_path = fullfile(data_path, 'hetrec2011-delicious-2k');
    randData_path = fullfile(data_path, 'random');
    
    % set variables
    Variables.fullTime = 0;
    Variables.topValue = [];
    Variables.topIndexes = [];
    Variables.top_t = power(10,0:3); 
    Variables.varSize = [size(samples,2),size(Variables.top_t,2)];
    Variables.samples = samples;
    Variables.budget = budget;
    Variables.turn = turn;
    Variables.out_dir = out_dir;
    %% ml-10m
    Variables.dataName = 'ml-10m';

    load(fullfile(ml_10m_path,'User.Mat'));
    load(fullfile(ml_10m_path,'Movie.Mat'));
    load(fullfile(ml_10m_path,'Tag.Mat'));
    load(fullfile(ml_10m_path,'topValue.Mat'));
    load(fullfile(ml_10m_path,'fullTime.Mat'));
    load(fullfile(ml_10m_path,'topIndexes.Mat'));
    
    Variables.MatA = User;
    Variables.MatB = Movie;
    Variables.MatC = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    Variables.topIndexes = topIndexes;
    
    oneDataSet(Variables);
    %% ml-20m
    Variables.dataName = 'ml-20m';
 
    load(fullfile(ml_20m_path,'User.Mat'));
    load(fullfile(ml_20m_path,'Movie.Mat'));
    load(fullfile(ml_20m_path,'Tag.Mat'));
    load(fullfile(ml_20m_path,'topValue.Mat'));
    load(fullfile(ml_20m_path,'fullTime.Mat'));
    load(fullfile(ml_20m_path,'topIndexes.Mat'));
 
    Variables.MatA = User;
    Variables.MatB = Movie;
    Variables.MatC = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    Variables.topIndexes = topIndexes;
    
    oneDataSet(Variables);

    %% ml-2k
    Variables.dataName = 'ml-2k';

    load(fullfile(ml_2k_path,'User.Mat'));
    load(fullfile(ml_2k_path,'Movie.Mat'));
    load(fullfile(ml_2k_path,'Tag.Mat'));
    load(fullfile(ml_2k_path,'topValue.Mat'));
    load(fullfile(ml_2k_path,'fullTime.Mat'));
    load(fullfile(ml_2k_path,'topIndexes.Mat'));

    Variables.MatA = User;
    Variables.MatB = Movie;
    Variables.MatC = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    Variables.topIndexes = topIndexes;   

    oneDataSet(Variables);
    %% lastfm 
    Variables.dataName = 'lastfm';

    load(fullfile(lastfm_path,'User.Mat'));
    load(fullfile(lastfm_path,'Artist.Mat'));
    load(fullfile(lastfm_path,'Tag.Mat'));
    load(fullfile(lastfm_path,'topValue.Mat'));
    load(fullfile(lastfm_path,'fullTime.Mat'));
    load(fullfile(lastfm_path,'topIndexes.Mat'));
 
    Variables.MatA = User;
    Variables.MatB = Artist;
    Variables.MatC = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    Variables.topIndexes = topIndexes;
     
    oneDataSet(Variables);
     
    %% delicious
    Variables.dataName = 'delicious';
 
    load(fullfile(delicious_path,'User.Mat'));
    load(fullfile(delicious_path,'Url.Mat'));
    load(fullfile(delicious_path,'Tag.Mat'));
    load(fullfile(delicious_path,'topValue.Mat'));
    load(fullfile(delicious_path,'fullTime.Mat'));
    load(fullfile(delicious_path,'topIndexes.Mat'));
 
    Variables.MatA = User;
    Variables.MatB = Url;
    Variables.MatC = Tag;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    Variables.topIndexes = topIndexes;
    
    oneDataSet(Variables);

    %% random data
    Variables.dataName = 'random';
    
    load(fullfile(randData_path, 'MatA.Mat'));
    load(fullfile(randData_path, 'MatB.Mat'));
    load(fullfile(randData_path, 'MAtC.Mat'));
    load(fullfile(randData_path, 'topValue.Mat'));
    load(fullfile(randData_path, 'fullTime.Mat'));
    load(fullfile(randData_path, 'topIndexes.Mat'));
    Variables.MatA = MatA;
    Variables.MatB = MatB;
    Variables.MatC = MatC;
    Variables.topValue = topValue;
    Variables.fullTime = fullTime;
    Variables.topIndexes = topIndexes;
    
    oneDataSet(Variables);

end
function [ diamond, equality, extension ] = initVar(Variables)
    diamond.recall = zeros(Variables.varSize);
    diamond.times = zeros(size(Variables.samples));
    equality.recall = zeros(Variables.varSize);
    equality.times = zeros(size(Variables.samples));
    extension.recall = zeros(Variables.varSize);
    extension.times = zeros(size(Variables.samples));
end

function [ diamond, equality, extension ] = oneSampling(Variables)
    [ diamond, equality, extension ] = initVar(Variables);
    MatA = Variables.MatA;
    MatB = Variables.MatB;
    MatC = Variables.MatC;
    for s = 1:Variables.turn
        [ diamondTemp, equalityTemp, extensionTemp ] = initVar(Variables);
        for i = 1:length(Variables.samples)
            for t = 1:length(Variables.top_t)
                  top_t = Variables.top_t(t);
                  budget = Variables.budget(i);
                  sample = Variables.samples(i);
                  
                  [value, time, ~] = diamondTensor(MatA', MatB, MatC, budget, sample, top_t);
                  diamondTemp.recall(i,t) = sum(value(1:top_t) >= Variables.topValue(top_t))/top_t;
                  diamondTemp.times(i) = time;
                  
                  [value, time, ~] = equalitySampling(MatA, MatB, MatC, budget, sample, top_t);
                  equalityTemp.recall(i,t) = sum(value(1:top_t) >= Variables.topValue(top_t))/top_t;
                  equalityTemp.times(i) = time;
                  
                  [value, time, ~] = extensionSampling(MatA, MatB, MatC, budget, sample, top_t);
                  extensionTemp.recall(i,t) = sum(value(1:top_t) >= Variables.topValue(top_t))/top_t;
                  extensionTemp.times(i) = time;
            end
        end
        diamond.recall = diamond.recall + diamondTemp.recall;
        diamond.times = diamond.times + diamondTemp.times;
        
        equality.recall = equality.recall + equalityTemp.recall;
        equality.times = equality.times + equalityTemp.times;
        
        extension.recall = extension.recall + extensionTemp.recall;
        extension.times = extension.times + extensionTemp.times;
    end
    diamond.recall = diamond.recall/Variables.turn;
    diamond.times = diamond.times/Variables.turn;
    
    equality.recall = equality.recall/Variables.turn;
    equality.times = equality.times/Variables.turn;
    
    extension.recall = extension.recall/Variables.turn;
    extension.times = extension.times/Variables.turn;

end

function drawTimeFig(Variables, diamond, equality, extension)
    samples = Variables.samples;
    fullTime = Variables.fullTime;
    titlename = Variables.titlename;
    out_dir = Variables.out_dir;
    h = figure; hold on;title(titlename); 
    xlabel('log_{10}Samples'); 
    ylabel('log_{10}T(sec)');
    
    plot(log10(samples),log10(fullTime*ones(size(samples))),'b','LineWidth',2);
    plot(log10(samples),log10(diamond.times),'r','LineWidth',2); 
    plot(log10(samples),log10(equality.times),'--g','LineWidth',2);
    plot(log10(samples),log10(extension.times),'--c','LineWidth',2);
    
    legend('exhaustive','diamond','equality','extension',4);
    saveas(h,fullfile(out_dir,[titlename,'.pdf']));
    close(h);
end

function drawRecallFig(Variables, diamond, equality, extension)
    samples = Variables.samples;
    titlename = Variables.titlename;
    out_dir = Variables.out_dir;
    % draw recall-samples diamond vs equality
    h = figure; hold on; title([titlename,'-diamond-equality']);
    xlabel('log_{10}Samples'); 
    ylabel('recall');
    axis([log10(samples(1)) log10(samples(end)) 0 1.1]);
    c = ['r','b','k','g', 'c'];
    desc = cell(size(Variables.top_t,2)*2, 1);
    for i = 1:size(Variables.top_t,2)
        top_t = Variables.top_t(i);
        plot(log10(samples),diamond.recall(:,i),c(i),'LineWidth',2);
        desc{2*i-1} = ['diamond:t=',num2str(top_t)];
        plot(log10(samples),equality.recall(:,i),['--',c(i)],'LineWidth',2); 
        desc{ 2*i } = ['equality:t=',num2str(top_t)];
    end
    legend(desc,4);  
    saveas(h,fullfile(out_dir,['diamond-equality-',titlename,'.pdf']));
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
    % draw recall-samples extension vs equality
    h = figure; hold on; title([titlename,'-extension-equality']);
    xlabel('log_{10}Samples'); 
    ylabel('recall');
    axis([log10(samples(1)) log10(samples(end)) 0 1.1]);
    c = ['r','b','k','g', 'c'];
    desc = cell(size(Variables.top_t,2)*2, 1);
    for i = 1:size(Variables.top_t,2)
        top_t = Variables.top_t(i);
        plot(log10(samples),extension.recall(:,i),c(i),'LineWidth',2); 
        desc{2*i-1} = ['extension:t=',num2str(top_t)];
        plot(log10(samples),equality.recall(:,i),['--',c(i)],'LineWidth',2); 
        desc{ 2*i } = ['equality:t=',num2str(top_t)];
    end
    legend(desc,4);  
    saveas(h,fullfile(out_dir,['extension-equality-', titlename,'.pdf']));
    close(h);
end

function oneDataSet(Variables)
    [ diamond, equality, extension ] = oneSampling(Variables);
    
    dataName = Variables.dataName;
    Variables.titlename = ['time-samples-',dataName];
    drawTimeFig(Variables, diamond, equality, extension);
    
    Variables.titlename = ['recall-samples-',dataName];
    drawRecallFig(Variables, diamond, equality, extension);

end
