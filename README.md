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
    - ***\Recursive Feature Elimination：\*** RFE is a feature selection method that fits a model and removes the weakest feature (or features) until the specified number of features is reached. Here we adapt `REFCV`  from [yellowbrick.model_selection](https://www.scikit-yb.org/en/latest/api/model_selection/rfecv.html) It selects the best subset of features for the supplied estimator by removing 0 to N features (where N is the number of features) using recursive feature elimination, then selecting the best subset based on the cross-validation score of the model.
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

