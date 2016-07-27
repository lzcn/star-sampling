function TRIAL005()
  
end
function [p1, p2] = getProbability(Variables)
    MatA = Variables.MatA;
    MatB = Variables.MatB;
    MatC = Variables.MatC;
    weightD = zeros(size(MatA));
    weightR = zeros(size(MatA,2),1);
    for r = 1 : size(MatA,2)
        weightR(r) = sum(abs(MatA(:,r)))*sum(MatB(:,r))*sum(MatC(:,r));
        for i = 1 : size(MatA,1)
            weightD(i,r) = abs(MatA(i,r))*sum(abs(MatA(i,:)))*sum(MatB(:,r))*sum(MatC(:,r));
        end
    end
    sum1 = sum(sum(weightD));
    sum2 = sum(sum(weightR));
    sum3 = sum(sum(weightA));
    p1 = zeros(size(topValue));
    p2 = zeros(size(topValue));
    for t = 1: length(topValue)
        idx = topIndexes(t,1);
        value = sum( MatA(topIndexes(t,1),:).* MatB(topIndexes(t,2),:).* MatC(topIndexes(t,3),:) );
        p1(t) = value*sum(abs(MatA(idx,:)))/sum1;
        p2(t) = value/sum2;
    end
    titlename = ['probability-instances-', Variables.dataName];
    h = figure; 
    hold on; 
    title(titlename);
    plot(p1,'b');
    plot(p2,'r');
    legend('diamond','equality');
    saveas(h,fullfile(Variables.out_dir,[titlename,'.png']));
end
