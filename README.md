### Comparisons of XGB [a8f670d](https://github.com/dmlc/xgboost/tree/a8f670d24742002ed35f8e4927d9e7b7d3ec1d14)(2017/11/02) and LGB [7a166fb](https://github.com/Microsoft/LightGBM/tree/7a166fb32271791fc164eca4d65f9819e6f7e902)(2017/11/01)

* LightGBM.ipynb: Modified version of [marugari's work](https://github.com/marugari/Notebooks/blob/ed6aa7835579ce9143850ed5956912895c984d56/LightGBM.ipynb)
* exp013
  * model                : XGB(hist_depthwise, hist_lossguie, hist_GPU, GPU), LGB
  * objective            : Binary classification
  * metric               : Logloss
  * dataset              : make_classification
  * n_train              : 0.5M, 1M, 2M, 4M
  * n_valid              : n_train/4
  * n_features           : 32
  * n_clusters_per_class : 8
  * n_rounds             : 100
  * max_depth            : 5, 10, 15
  * num_leaves           : 2 ** max_depth
* exp014
  * model                : XGB(hist_depthwise, hist_lossguie, hist_GPU, GPU), LGB
  * objective            : Binary classification
  * metric               : Logloss
  * dataset              : make_classification
  * n_train              : 1,2,4,8,16,32,64 * 10K
  * n_valid              : n_train/4
  * n_features           : 256
  * n_clusters_per_class : 8
  * n_rounds             : 100
  * max_depth            : 5, 10
  * num_leaves           : 2 ** max_depth
  
The following codes were run on older versions of XGBoost and LightGBM
* exp010
  * model                : XGB(CPU, EQBIN_depthwise, EQBIN_lossguie, GPU), LGB
  * objective            : Binary classification
  * metric               : Logloss
  * dataset              : make_classification
  * n_train              : 0.5M, 1M, 2M
  * n_valid              : n_train/4
  * n_features           : 32
  * n_rounds             : 100
  * n_clusters_per_class : 8
  * max_depth            : 5, 10, 15
* exp011
  * model                : XGB(EQBIN_depthwise, EQBIN_lossguie), LGB
  * objective            : Binary classification
  * metric               : Logloss
  * dataset              : make_classification
  * n_train              : 0.5M, 1M, 2M
  * n_valid              : n_train/4
  * n_features           : 32
  * n_clusters_per_class : 8
  * n_rounds             : 200
  * max_depth            : 5, 10, 15, 20
  * num_leaves           : 32, 256, 1024, 4096, 16384
* exp012
  * model                : XGB(EQBIN_depthwise, EQBIN_lossguie), LGB
  * objective            : Binary classification
  * metric               : Logloss
  * dataset              : make_classification
  * n_train              : 1, 2, 4, 8, 16, 32 * 10000
  * n_valid              : n_train/4
  * n_features           : 256
  * n_clusters_per_class : 8
  * n_rounds             : 100
  * max_depth            : 5, 10, 15, 20
  * num_leaves           : 32, 256, 1024, 4096

