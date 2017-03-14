function [categories,scores]=predictCategory(im,categoryClassifier)
if nargin==1
    categoryClassifier=load('classifier.mat');
    categoryClassifier=categoryClassifier.categoryClassifier;
end
[labelIdx, scores] = predict(categoryClassifier, im);
categories=categoryClassifier.Labels(labelIdx);
