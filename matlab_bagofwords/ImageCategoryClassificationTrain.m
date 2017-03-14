%% Image Category Classification Using Bag of Features
% This example shows how to use a bag of features approach for image
% category classification. This technique is also often referred to as bag
% of words. Visual image categorization is a process of assigning a
% category label to an image under test. Categories may contain images
% representing just about anything, for example, dogs, cats, trains, boats.
%
% Copyright 2014 The MathWorks, Inc. 

%% Download Caltech101 Image Set
% To learn about bag of features image category classification, you will
% first download a suitable image data set. One of the most widely cited
% and used data sets is  <http://www.vision.caltech.edu/Image_Datasets/Caltech101 Caltech 101>, collected by Fei-Fei Li, Marco Andreetto,
% and Marc 'Aurelio Ranzato.

% Location of the compressed data set
% url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% % Store the output in a temporary folder
% outputFolder = fullfile(tempdir, 'caltech101'); % define output folder

%%
% Note that downloading the set from the web can take a very long time depending on
% your Internet connection. It can be 30 minutes or more since the set contains 
% 126MB of data. The commands below will block MATLAB for that period of
% time. Alternatively, you can use your web browser to first download the
% set to your local disk. If you choose that route, re-point the 'url' variable
% above to the file that you downloaded.

% if ~exist(outputFolder, 'dir') % download only once
%     disp('Downloading 126MB Caltech101 data set...');
%     untar(url, outputFolder);
% end

%% Load Image Sets
% Instead of operating on the entire Caltech 101 set, which can be time
% consuming, use three categories: airplanes, ferry, and laptop.
% Note that for the bag of features approach to be effective, majority of
% each image's area must be occupied by the subject of the category, for
% example, an object or a type of scene.

outputFolder='../../datasets/caltech101';
rootFolder = fullfile(outputFolder, '101_ObjectCategories');

%%
% Construct an ImageDatastore based on the following categories from
% Caltech 101: 'airplanes', 'ferry', 'laptop'. Use |imageDatastore| to help
% you manage the data. Since |imageDatastore| operates on image file locations,
% and therefore does not load all the images into memory, it is safe to use
% on large image collections. 
categoriesFolders=dir(rootFolder);
categoriesFolders(1:3)=[];
categories= {categoriesFolders(:).name}';
% categories=categories(randperm(length(categories),20));
% categories=categories(1:10);
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

%%
% You can easily inspect the number of images per category as well as
% category labels as shown below:
tbl = countEachLabel(imds)

%%
% Note that the labels were derived from directory names used to construct
% the ImageDatastore, but can be customized by manually setting the Labels
% property of the ImageDatastore object.

%% Prepare Training and Validation Image Sets
% Since |imds| above contains an unequal number of images per category,
% let's first adjust it, so that the number of images in the training set is balanced.

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%%
% Separate the sets into training and validation data. Pick 30% of images
% from each set for the training data and the remainder, 70%, for the 
% validation data. Randomize the split to avoid biasing the results.

[trainingSet, validationSet] = splitEachLabel(imds, 0.3, 'randomize');

%%
% % The above call returns two imageDatastore objects ready for training
% % and validation tasks. Below, you can see example images from the three
% % categories included in the training data.
% 
% % Find the first instance of an image for each category
% airplanes = find(trainingSet.Labels == 'airplanes', 1);
% ferry = find(trainingSet.Labels == 'ferry', 1);
% laptop = find(trainingSet.Labels == 'laptop', 1);
% 
% % figure
% 
% subplot(1,3,1);
% imshow(readimage(trainingSet,airplanes))
% subplot(1,3,2);
% imshow(readimage(trainingSet,ferry))
% subplot(1,3,3);
% imshow(readimage(trainingSet,laptop))

%% Create a Visual Vocabulary and Train an Image Category Classifier
% Bag of words is a technique adapted to computer vision from the
% world of natural language processing. Since images do not actually
% contain discrete words, we first construct a "vocabulary" of 
% <matlab:doc('extractFeatures'); SURF> features representative of each image category.

%%
% This is accomplished with a single call to |bagOfFeatures| function,
% which:
%
% # extracts SURF features from all images in all image categories
% # constructs the visual vocabulary by reducing the number of features
%   through quantization of feature space using K-means clustering
bag = bagOfFeatures(trainingSet);

%%
% Additionally, the bagOfFeatures object provides an |encode| method for
% counting the visual word occurrences in an image. It produced a histogram
% that becomes a new and reduced representation of an image.

% img = readimage(imds, 1);
% featureVector = encode(bag, img);
% 
% % Plot the histogram of visual word occurrences
% figure
% bar(featureVector)
% title('Visual word occurrences')
% xlabel('Visual word index')
% ylabel('Frequency of occurrence')

%%
% This histogram forms a basis for training a classifier and for the actual
% image classification. In essence, it encodes an image into a feature vector. 
%
% Encoded training images from each category are fed into a classifier
% training process invoked by the |trainImageCategoryClassifier| function.
% Note that this function relies on the multiclass linear SVM classifier
% from the Statistics and Machine Learning Toolbox(TM).

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

%%
% The above function utilizes the |encode| method of the input |bag| object
% to formulate feature vectors representing each image category from the 
% |trainingSet|.

%% Evaluate Classifier Performance
% Now that we have a trained classifier, |categoryClassifier|, let's
% evaluate it. As a sanity check, let's first test it with the training
% set, which should produce near perfect confusion matrix, i.e. ones on 
% the diagonal.

confMatrix = evaluate(categoryClassifier, trainingSet);
mean(diag(confMatrix))

%%
% Next, let's evaluate the classifier on the validationSet, which was not
% used during the training. By default, the |evaluate| function returns the
% confusion matrix, which is a good initial indicator of how well the
% classifier is performing.

confMatrix = evaluate(categoryClassifier, validationSet);

% Compute average accuracy
mean(diag(confMatrix))

%%
% Additional statistics can be derived using the rest of arguments returned
% by the evaluate function. See help for |imageCategoryClassifier/evaluate|.
% You can tweak the various parameters and continue evaluating the trained
% classifier until you are satisfied with the results.

%% Try the Newly Trained Classifier on Test Images
% You can now apply the newly trained classifier to categorize new images.

% img = imread(fullfile(rootFolder, 'airplanes', 'image_0690.jpg'));
% [labelIdx, scores] = predict(categoryClassifier, img);
% 
% % Display the string label
% categoryClassifier.Labels(labelIdx)
% 
% displayEndOfDemoMessage(mfilename)

%% ±£¥Ê∑÷¿‡∆˜
save('classifier','categoryClassifier');
