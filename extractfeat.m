
%finds hog features of an image.

%lip 140:195 50:140 
%eye1 55:105 15:80 
%eye2 55:105 95:150 
%nose 60:150 65:105

%vlfeat package is used
run('vlfeat-0.9.20\toolbox\vl_setup.m');

%feature train data
Dir = dir('Data\Pose_0_-15\training\*');
ft = [];lbl = [];
for i=3:size(Dir,1)
    im = imread(strcat('Data\Pose_0_-15\training\',Dir(i).name));
    temp = strsplit(Dir(i).name,'_');
    lbl = [lbl;label(temp{1})];
    I = single(rgb2gray(im));
    
    %resizing image to same height and width
    I = imresize(I,[210 178]);
    
    %extracting hog features
    t1 = vl_hog(I(140:195,50:140),15);t1 = t1(:)'; 
    t2 = vl_hog(I(55:105,15:80),15);t2 = t2(:)';
    t3 = vl_hog(I(55:105,95:150),15);t3 = t3(:)';
    t4 = vl_hog(I(60:150,65:105),15);t4 = t4(:)';
    
    t = [t1,t2,t3,t4]; %concatenating all features of single image
    ft = [ft;t];
end
ftrainn15 = ft;

%feature test data
Dir1 = dir('Data\Pose_0_-15\testing\*');
tft = [];
atlbl = [];
for i=3:size(Dir1,1)
    im = imread(strcat('Data\Pose_0_-15\testing\',Dir1(i).name));
    temp = strsplit(Dir1(i).name,'_');
    atlbl = [atlbl;label(temp{1})];
    I = single(rgb2gray(im));
    
    %resizing image to same height and width
    I = imresize(I,[210 178]); 
    
    %extracting hog features
    t1 = vl_hog(I(140:195,50:140),15);t1 = t1(:)';
    t2 = vl_hog(I(55:105,15:80),15);t2 = t2(:)';
    t3 = vl_hog(I(55:105,95:150),15);t3 = t3(:)';
    t4 = vl_hog(I(60:150,65:105),15);t4 = t4(:)';
    
    t = [t1,t2,t3,t4]; %concatenating all features of single image
    tft = [tft;t];
end
ftestn15 = tft;
