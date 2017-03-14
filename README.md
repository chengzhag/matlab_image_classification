# project1_image_classification

基于matlab和bag of words的图像分类，
目录中不包含数据集！

## 设计目标

- 输入一幅图像，输出标签
- 可以固定图片大小

## 实现思路

- 数据集：caltech101、caltech256

### matlab + bag of words


- bag of words：利用matlab中bagOfFeatures函数提取SURF特征并K-means聚类构造“词典”
- 分类器：利用trainImageCategoryClassifier函数训练线性SVM

### matlab + cnn

