# project1_image_classification

基于matlab和bag of words的图像分类，
目录中不包含数据集！

## 设计目标

- 输入一幅图像，输出标签
- 可以固定图片大小

## 实现

- 数据集：caltech101

### matlab + bag of words

>改自ImageCategoryClassificationTrainSample

- bag of words：利用matlab中bagOfFeatures函数提取SURF特征并K-means聚类构造“词典”
- svm：利用trainImageCategoryClassifier函数训练线性SVM分类器

测试结果：
- 训练集正确率：97.91%
- 测试集正确率：30.11%


### matlab + cnn + svm

> 改自DeepLearningImageClassificationSample

- cnn：利用预先训练好的AlexNet CNN网络获取特征向量，由于AlexNet已经针对ImageNet上的众多样本进行了训练，从其中抽取的特征向量对于一般图像具有较强的区分能力
- svm：fitcecoc函数可以方便地训练基于SVM的多分类分类器

测试结果：
- 训练集正确率：99.67%
- 测试集正确率：77.95%

### 运行说明

- 两个实验文件夹都包含xml、m文件
- xml为matlab2016b的新功能，旧版本可使用m文件
- 两个文件夹都有predictCategory.m文件，该函数输入参数为一幅任意图形，输出参数为类别字符串的元胞数组

