clear all
clc;
%%
%single view knn
load('features\ftrain15.mat');
load('features\ftest15.mat');

load('features\trainY.mat');
load('features\testY.mat');


trainx = ftrain15; trainy = trainY;
testx = ftest15;testy = testY;

%%
%knn model
label = zeros(size(testy,1),7);
count = 0;
for i=1:size(testy,1)
 idx = knnsearch(trainx,testx(i,:),'k',10,...
     'distance','seuclidean');
 t1 = trainy(idx,:);
 t1 = sum(t1,1);
 [m,id] = max(t1);
 label(i,id) = 1;
 if(testy(i,id) == 1)
   count = count + 1;
 end
end
Accuracy = (count/size(testy,1))*100


%%
%confusion matrix
confmat = cell(8,9);
confmat{1,1} = 'Emotions';
confmat{2,1} = 'Anger';confmat{3,1} = 'Disgust';confmat{4,1} = 'Fear'; confmat{5,1} = 'Happy'; 
confmat{6,1} = 'Neutral' ; confmat{7,1} = 'Sad'; confmat{8,1} = 'Surprise';
confmat{1,2} = 'Anger';confmat{1,3} = 'Disgust';confmat{1,4} = 'Fear'; confmat{1,5} = 'Happy'; 
confmat{1,6} = 'Neutral' ; confmat{1,7} = 'Sad'; confmat{1,8} = 'Surprise'; confmat{1,9} = 'Total';
for i=2:8
    t1 = find(testy(:,i-1)==1);
    t2 = sum(label(t1,:),1);
    for j=2:8
    confmat{i,j}  = t2(j-1);
    end
    confmat{i,j+1} = sum(t2);
end