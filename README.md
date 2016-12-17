### Comparison of XGB:[2b6aa77](https://github.com/dmlc/xgboost/tree/167864da75073d6bb6f04e3e9dafce10db1b34dd) and LGBM:[ebfc852](https://github.com/Microsoft/LightGBM/tree/c7ef83227976adc4143097b203e3862c485e90cf)

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

