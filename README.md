### Comparison of XGB:[7948d1c](https://github.com/dmlc/xgboost/tree/7948d1c7998eeb205f13740c5a1bb3f381c37b6a) and LGBM:[bd7274b](https://github.com/Microsoft/LightGBM/tree/bd7274baee41be744c9cf3969340fe8540000fad)

The objective function of LGBM for binary classification is slightly different from XGB's.
* LGBM: -log(1+exp(-2 * sigmoid * label * score))
* XGB : -log(1+exp(-label * score))
  * label in {-1, 1}, score: leaf values
  * The default value of sigmoid is 1.0.

So LGBM's parameter sigmoid is set to 0.5.

* LightGBM.ipynb: Modified version of [marugari's work](https://github.com/marugari/Notebooks/blob/ed6aa7835579ce9143850ed5956912895c984d56/LightGBM.ipynb)
* [Higgs dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS)
  * eval metric          : AUC(exp004), Logloss(exp007), Training only(exp008)
  * n_train              : 10K, 0.1M, 1M, 10M
  * n_valid              : 0.5M(exp004, exp007)
  * n_features           : 28
  * n_rounds             : 500
  * max_depth            : 5, 10, 15
* Artificial datasets(make_classification(n_clusters_per_class=64))
  * preprocessing        : None(exp005), Equal freq binning(exp006)
  * n_train              : 1M, 2M
  * n_valid              : n_train/4
  * n_features           : 28
  * n_clusters_per_class : 64
  * max_depth            : 5, 10,11,..,16

