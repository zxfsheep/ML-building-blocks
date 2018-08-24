## Machine learning building blocks
This place is to accumulate the pieces that I will use repeatedly in data science and machine learning projects.

Everything is linked below with high level comments and caveats. More thoughts and details are in the individual notebooks, as well as the required modules.

## Models

1. [Some common baseline (regression) models with `GridSearchCV`](https://github.com/zxfsheep/ML-building-blocks/blob/master/models/GridSearchCV_Baseline.ipynb), including linear regression, ridge regression, Lasso regression, ElasticNet, SVM. Tunable parameters are described in the notebook. I think these are useful for:
   * Getting a quick baseline result for the problem, which serves as a sanity check.
   * Can throw in a small portion into the final result, which is somewhat a final regularization.
   * Help to find a good KFold split for expensive models, as will be explained in my random-idea repository.

2. The best performing model for many common problems: Gradient tree boosting:
   * [XGBoost with `GridSearchCV`](https://github.com/zxfsheep/ML-building-blocks/blob/master/models/GridSearchCV_XGBoost.ipynb)
   * [LightGBM with `GridSearchCV`](https://github.com/zxfsheep/ML-building-blocks/blob/master/models/GridSearchCV_LGBM.ipynb)
   * [XGBoost with customized CV](https://github.com/zxfsheep/ML-building-blocks/blob/master/models/CustomCV_XGBoost.ipynb)
   * [LightGBM with customized CV](https://github.com/zxfsheep/ML-building-blocks/blob/master/models/CustomCV_LGBM.ipynb)
   
   There are numerous articles on these models and their usage. Here I will discuss additional thoughts from practice:
   * Many people as well as the official documents mentioned that these modules might have technical issues with Scikit-learn, especially `GridSearchCV`. In particular, the extra booster parameters are not affected by `set_params()` method, and also the performance is very poor. I found solutions to these, as in the notebooks. In particular, I wrapped the modules with a Scikit-learn custom estimator which avoids the parameter issue. Also setting `return_train_score = False` in GridSearchCV avoids predicting training set.
   * However, I do not use `GridSearchCV` at late stages, and use the customized CV instead. One practical reason is that one CV can cost hours, so independent CV is more flexible. Another benefit is that it is more convenient to apply early stopping, because `GridSearchCV` does not give a direct handle to validation set during fitting.
   * This leads to another problem: how to produce the final prediction. These additive tree models require early stopping to avoid overfitting. One way is to set aside a validation set from the training set, and use it for the final training after all parameters have been tuned. In practice, I use the whole training set during cross validation, and average the predictions made by the estimator from the best iteration of each fold. This has an additional benefit to obtain a complete non-leaked prediction for the entire training set during CV, ie. the prediction of a sample is given by the round where this sample is in the validation set of CV. This complete prediction can be used for **stacking** models.
   * Some comparisons between XGBoost and LightGBM: 
      * LightGBM is leaf based and XGBoost is level based. Roughly speaking, XGBoost is more like a breadth-first-search, and LightGBM is more like a depth-first-search.
      * As a result, LightGBM runs much faster than XGBoost, but risks overfitting. In Kaggle contests, I found that even though LightGBM can have better local CV scores, it can be outperformed by XGBoost on unseen data.
      * To speed up hyperparameter tuning, I use a smaller dataset, a higher learning rate and/or a smaller set of features. Even though LightGBM and XGBoost share most parameters (in fact their entire API look suspiciously similar...), there are some differences. Since LightGBM is leaf based, `num_leaves` is a crucial tunable parameter to avoid overfitting, which does not exist in XGBoost. Furthermore, the `gamma` parameter in XGBoost often does not make any impact for me, while the equivalent parameter `min_split_gain` has more effect. A good tuning guide for XGBoost is [this blog.](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
   * On the GPU usage of XGBoost: my laptop has an Nvidia 980m graphics card, which has 8GB dedicated memory. However, XGBoost seems to have an issue of not automatically clearing previous training data when retraining, which can lead to stopping without warning. So the booster should be manually garbage collected as in the notebook.
   

## Utilities

1. [Memory reduction trick to deal with large datasets in pandas.](https://github.com/zxfsheep/ML-building-blocks/blob/master/utilities/Reduce_Memory.ipynb) The data types that pandas automatically choose might be too much a waste, so we can manually search for the most efficient data type for each feature. This only needs to be run once if the dataset is static or large enough to be representative of unseen data. We can save the optimal data types and use them directly the next time we read in data.
