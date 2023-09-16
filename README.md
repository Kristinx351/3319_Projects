# Project 1: Dimensionality Reduction

- `Data/AwA2-feature/ResNet101`

  - `AwA2-features.txt`:  Containing 2048-dim features of 37322 samples.

  - `AwA2-filenames.txt` : Containing the filenames of corresponding pictures of samples by `[animal]_[id].jpg` .
  - `AwA2--labels.txt` : List labels (1 ~ 50) of 37322 samples.

## 0. Data processor

- Covert file:

- Split data: Train : Test = 6 : 4 (have to shuffle it firstly).
- Here we use labels as the standard for stratify to make sure testing set contains approximately the same percentage of samples of each class as the training set.

## 1. Classification

- Use linear SVM:

- Use K-fold & Grid search to determine the `C`  of SVM.

  - Cross validation (CV) could make the parameter more generalized, avoiding the problem of over-fitting. The proportion of training set should > 50%, and training/testing set should be sampled uniformly.

  We use `LinearSVC` from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) as the classification model. And use [GridsearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)  to grid search the C by K-fold.

## 2. Dimensionality reduction

### 2.1 Methods

- Feature selection method:

  - Forward Selection
  - Backward Selection
    - **Recursive Feature Elimination：** RFE is a feature selection method that fits a model and removes the weakest feature (or features) until the specified number of features is reached. Here we adapt `REFCV`  from [yellowbrick.model_selection](https://www.scikit-yb.org/en/latest/api/model_selection/rfecv.html) It selects the best subset of features for the supplied estimator by removing 0 to N features (where N is the number of features) using recursive feature elimination, then selecting the best subset based on the cross-validation score of the model.
  - Genetic algorithm

- Feature projection method:

  - PCA (Principle Component Analysis)

    We use PCA from `sklearn.descomposition`  and keep different components (ranging from $2^3$ to $2^{11}$).

  - LDA (Linear Discriminative Analysis)

    We use LDA from `sklearn.discriminant_analysis`  and keep different components (same as above).

  - Auto-encoder

- Feature learning method:

  - [SNE](https://bindog.github.io/blog/2016/06/04/from-sne-to-tsne-to-largevis/) (visualize it): It is based on the idea that: Data points that are similar in high-dimensional space map to low-dimensional space distances that are also similar. The conventional practice is to express this similarity in terms of Euclidean distance, while SNE converts this distance relationship into a conditional probability to express similarity.

    We adapt SNE from [sklearn.](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)

  - LLE: [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html#sklearn.manifold.locally_linear_embedding).

### 2.2 Experiments:

- Compare performance variance w.r.t diff dimensionality.

```
>>> ConvergenceWarning: Liblinear failed to converge, increase the number of ite... 
```

How to solve: [1](https://blog.csdn.net/weixin_42827025/article/details/122401155), [2](https://huaweicloud.csdn.net/63806875dacf622b8df86ecc.html?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~activity-3-113144702-blog-122401155.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~activity-3-113144702-blog-122401155.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=4):

1. Set `max_iter=5000` ;
2. Extract better feature;
3. Ignore the warning by:

```python
import warnings
warnings.filterwarnings("ignore") # This will ignore all the warnings;
```

# Project 2: Distance Metrics

## 0. Data processor

- The same as [Project 1]
- The way u split the dataset should be the same as project 1!
- Check the data distribution of each dimension of the feature to avoid heterogenous problem.
    
    [ mean + std histgram?]
    

## 1. Classification

- Use KNN for classification based on features:
    
    Here we use KNN from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). 
    
- Could use K-flod CV to decide the hyper-paramete K in the KNN.
- Why the training time of KNN << testing time?
    
    The training time of KNN is simply the time it takes to **load the training data into memory**. This can be done quickly, especially with modern hardware and efficient data loading techniques.
    
    On the other hand, the testing time of KNN is **proportional to the size of the training data, as it needs to compute the distance between the test point and every single training data point.** This can be very time-consuming, especially with large training datasets or high-dimensional feature spaces.
    

## 2. Try different distance metrics

### 2.1 Method

1. Euclidean distance：

    $L2 distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)$

2. Manhattan distance：
    
    $L1 = |x1 - x2| + |y1 - y2|$
      
3. Cosine


### Supervised metric learning


1. LMNN
2. NCA
3. LFDA
# Project 3: Zero shot classification

## 0. Data process

- The original dataset consists of 50 animal classes, and the training: test set is divided in a 4:1 ratio. Three rounds of cross-validation (training set: validation set = 27:13)

## 1. Classification methods
There are three main categories of approaches to zero shot problems: semantic relatedness methods,
semantic embedding methods, and synthetic methods.
Here we introduce one method for each categories.
### 1.1 Semantic Relatedness Methods
Semantic embedding methods are used to represent words or phrases in a continuous vector space where semantically similar words are mapped to nearby points. These methods can be used to compute the semantic similarity between words or phrases by measuring the distance between their vector representations.

Attribute Label Embedding (ALE) is a method for zero-shot learning that uses attribute vectors as label embeddings. In ALE, each class is embedded in the space of attribute vectors, and a compatibility function is used to measure the similarity between an image and a label embedding.

There are two generic methods to integrate attributes into multi-class classification: Direct Attribute Prediction (DAP) and Indirect Attribute Prediction (IAP), which are depicted in the Figure：
[DAP and IAP](Figs/DAP_IAP.pdf)
### 1.2 Semantic Embedding Methods
Semantic relatedness methods are used to quantitatively measure the relationship between two words or concepts based on the similarity or closeness of their meaning.

SCoRe proposes a new convolutional neural network (CNN) framework for zero-shot learning. The paper considers the role of semantics in zero-shot learning and analyzes the effectiveness of previous approaches according to the form of supervision provided.

[The feature extraction process based on common CNN architectures](Figs/score.pdf)

### 1.3 Synthetic Methods.
Synthetic methods perform classification by synthesizing fictitious samples. They typically use some form of generative model, such as a variational self-encoder or a generative adversarial network, to learn how to generate samples from category labels. The model then uses these generated samples to train a classifier and uses it to classify real samples.

TF-VAEGAN is a generative model that synthesizes visual features for unseen classes based on their semantic descriptions. The model uses a combination of Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) to generate semantically consistent features for unseen classes. The model consists of several components: an encoder $E$, a generator $G$, a discriminator $D$, and a semantic embedding decoder $S$.
[TF_VAEGAN](Figs/TF_VAEGAN.pdf)
