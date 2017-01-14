### Comparison of XGB 2017/1/14:[aeb4e76](https://github.com/dmlc/xgboost/tree/aeb4e76118482b94ac6d052eccef87a73d4bdeb0) and LGBM 2017/1/13:[d72d935](https://github.com/Microsoft/LightGBM/tree/d72d9359296212d004641be50ab54f9bb63d20e0)

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
* exp010
  * model                : XGB(CPU, EQBIN_depthwise, EQBIN_lossguie, GPU), LGB
  * dataset              : make_classification
  * n_train              : 0.5M, 1M, 2M
  * n_valid              : n_train/4
  * n_features           : 32
  * n_rounds             : 100
  * n_clusters_per_class : 8
  * max_depth            : 5, 10, 15
* exp011
  * model                : XGB(EQBIN_depthwise, EQBIN_lossguie), LGB
  * dataset              : make_classification
  * n_train              : 0.5M, 1M, 2M
  * n_valid              : n_train/4
  * n_features           : 32
  * n_clusters_per_class : 8
  * n_rounds             : 100
  * max_depth            : 5, 10, 15, 20
  * num_leaves           : 256, 1024, 4096

