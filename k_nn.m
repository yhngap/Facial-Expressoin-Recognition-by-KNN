%function returns labels of all the test images
function [label] = k_nn(trainx,trainy,testx,testy)
label = zeros(size(testy,1),7);
count = 0;
    for i=1:size(testy,1)
     idx = knnsearch(trainx,testx(i,:),'k',10,...
         'distance','seuclidean');
     t1 = trainy(idx,:);
     t1 = sum(t1,1);
     [m,id] = max(t1);
     label(i,id) = 1;
    end
end