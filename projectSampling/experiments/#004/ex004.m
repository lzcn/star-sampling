function ex004(data_path,out_dir,samples,budget,top_t,turn)
    ml_10m_path = fullfile(data_path,'MovieLens','ml-10m');
    ml_20m_path = fullfile(data_path,'MovieLens','ml-20m');
    ml_2k_path = fullfile(data_path, 'hetrec2011-movielens-2k-v2');
    lastfm_path = fullfile(data_path, 'hetrec2011-lastfm-2k');
    delicious_path = fullfile(data_path, 'hetrec2011-delicious-2k');


    %% ml-10m
    dataName = 'ml-10m';

    load(fullfile(ml_10m_path,'User.Mat'));
    load(fullfile(ml_10m_path,'Movie.Mat'));
    load(fullfile(ml_10m_path,'Tag.Mat'));
    load(fullfile(ml_10m_path,'topValue.Mat'));
    load(fullfile(ml_10m_path,'fullTime.Mat'));
    load(fullfile(ml_10m_path,'topIndexes.Mat'));
    oneDataSet(dataName, out_dir, ...
                    samples, budget, ...
                    User, Movie, Tag,...
                    topValue, top_t, turn);    
    %% ml-20m
    dataName = 'ml-20m';

    load(fullfile(ml_20m_path,'User.Mat'));
    load(fullfile(ml_20m_path,'Movie.Mat'));
    load(fullfile(ml_20m_path,'Tag.Mat'));
    load(fullfile(ml_20m_path,'topValue.Mat'));
    load(fullfile(ml_20m_path,'fullTime.Mat'));
    load(fullfile(ml_20m_path,'topIndexes.Mat'));

    oneDataSet(dataName, out_dir, ...
                    samples, budget, ...
                    User, Movie, Tag,...
                    topValue, top_t, turn);
    %%
    dataName = 'ml-2k';

    load(fullfile(ml_2k_path,'User.Mat'));
    load(fullfile(ml_2k_path,'Movie.Mat'));
    load(fullfile(ml_2k_path,'Tag.Mat'));
    load(fullfile(ml_2k_path,'topValue.Mat'));
    load(fullfile(ml_2k_path,'fullTime.Mat'));
    load(fullfile(ml_2k_path,'topIndexes.Mat'));

    oneDataSet(dataName, out_dir, ...
                    samples, budget, ...
                    User, Movie, Tag,...
                    topValue, top_t, turn);    
    %%

    dataName = 'lastfm';

    load(fullfile(lastfm_path,'User.Mat'));
    load(fullfile(lastfm_path,'Artist.Mat'));
    load(fullfile(lastfm_path,'Tag.Mat'));
    load(fullfile(lastfm_path,'topValue.Mat'));
    load(fullfile(lastfm_path,'fullTime.Mat'));
    load(fullfile(lastfm_path,'topIndexes.Mat'));

    oneDataSet(dataName, out_dir, ...
                    samples, budget, ...
                    User, Artist, Tag,...
                    topValue, top_t, turn);      
    %%
    dataName = 'delicious';

    load(fullfile(delicious_path,'User.Mat'));
    load(fullfile(delicious_path,'Url.Mat'));
    load(fullfile(delicious_path,'Tag.Mat'));
    load(fullfile(delicious_path,'topValue.Mat'));
    load(fullfile(delicious_path,'fullTime.Mat'));
    load(fullfile(delicious_path,'topIndexes.Mat'));
    oneDataSet(dataName, out_dir, ...
                    samples, budget, ...
                    User, Url, Tag,...
                    topValue, top_t, turn); 
    %% random data
    dataName = 'random';
    load('A.Mat');
    load('B.Mat');
    load('C.Mat');
    topValue = 1600*ones(1000,1);
    oneDataSet(dataName, out_dir, ...
                    samples, budget, ...
                    A, B, C,...
                    topValue, top_t, turn);  

end

function [ Recall_v, Recall] = initVar(budget)

    Recall_v = zeros(size(budget));
    Recall = zeros(size(budget));

end


function [ Recall_v, Recall ] = oneSampling(samples, budget, A, B, C, topValue, turn)

    top_t = 100;

    [ Recall_v, Recall ] = initVar(budget);

    for s = 1:turn
        [recall_v_temp,recall_temp] = initVar(budget);
        for i = 1 : length(budget)
            [value, ~, ~] = equalitySampling(A, B, C, budget(i),samples,top_t);
            recall_temp(i) = sum(value(1:top_t) >= topValue(top_t))/top_t;
            [value_v, ~, ~] = diamondTensor(A', B, C, budget(i),samples,top_t);
            recall_v_temp(i) = sum(value_v(1:top_t) >= topValue(top_t))/top_t;
        end
        
        Recall = Recall + recall_temp;
        Recall_v = Recall_v + recall_v_temp;
    end
    Recall_v = Recall_v/turn;
    Recall = Recall/turn;
end

function drawRecallFig(titlename, out_dir, budget, Recall1,Recall2,Recall3,Recall4)
    h = figure; hold on; title(titlename); 
    xlabel('log_{10}Budget'); 
    ylabel('recall');
    axis([log10(budget(1)) log10(budget(end)) 0 1.1]);
    plot(log10(budget),Recall1,'r','LineWidth',2);
    plot(log10(budget),Recall2,'k','LineWidth',2);
    plot(log10(budget),Recall3,'g','LineWidth',2);
    plot(log10(budget),Recall4,'b','LineWidth',2); 
    legend('v1','v2','v3','v4',4);  
    saveas(h,fullfile(out_dir,[titlename,'.png'])); 

end

function oneDataSet(dataName, out_dir, samples, budget, A, B, C, topValue,top_t,turn)
    [Recall4] = diamond_exp(A',B,C,budget,samples,top_t,topValue(top_t));
    %[Recall_v] = equality_v2_exp(A,B,C,budget,samples,top_t,topValue(top_t));
    %[Recall] = equality_exp(A,B,C,budget,samples,top_t,topValue(top_t));
    %[ Recall_v, Recall ] = oneSampling(samples, budget, A, B, C, topValue, top_t, turn);
    [Recall1,Recall2,Recall3,~] = equality_v2_exp(A,B,C,budget,samples,top_t,topValue(top_t));
    titlename = ['recall-budget-',dataName];
    drawRecallFig(titlename, out_dir, budget, Recall1,Recall2,Recall3,Recall4);
	Recall1
end