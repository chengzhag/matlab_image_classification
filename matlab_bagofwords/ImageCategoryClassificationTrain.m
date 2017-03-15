%% 数据集
outputFolder='../../datasets/caltech101';
rootFolder = fullfile(outputFolder, '101_ObjectCategories');

% 加载数据集
categoriesFolders=dir(rootFolder);
categoriesFolders(1:3)=[];
categories= {categoriesFolders(:).name}';
% categories=categories(randperm(length(categories),20));
% categories=categories(1:10);

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

%% 
% 显示类别和数量
tbl = countEachLabel(imds);
tbl(1:5,:)

%%
% 为了使各类样本数量平衡，选取数量最少的为基准抽取样本
minSetCount = min(tbl{:,2}); 
imds = splitEachLabel(imds, minSetCount, 'randomize');
tbl = countEachLabel(imds)
tbl(1:5,:)

%% 分割样本
% 将样本随机分为训练集和测试集
[trainingSet, validationSet] = splitEachLabel(imds, 0.3, 'randomize');

%% 提取 bag of words 词典
bag = bagOfFeatures(trainingSet);

%%
% 根据词典训练分类器
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

%% 测试分类器
% 测试训练集
confMatrix = evaluate(categoryClassifier, trainingSet);
mean(diag(confMatrix))

% 测试测试集
confMatrix = evaluate(categoryClassifier, validationSet);
mean(diag(confMatrix))

%% 保存分类器
save('classifier','categoryClassifier');
