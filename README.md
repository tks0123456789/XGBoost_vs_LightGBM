### Comparisons of XGB[0.81] and LGB[2.2.1]

* LightGBM.ipynb: Modified version of [marugari's work](https://github.com/marugari/Notebooks/blob/ed6aa7835579ce9143850ed5956912895c984d56/LightGBM.ipynb)
* exp013
  * model                : XGB(hist_depthwise, hist_lossguie, hist_GPU, GPU), LGB
  * objective            : Binary classification
  * metric               : Logloss
  * dataset              : make_classification
  * n_train              : 0.5M, 1M, 2M
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
  * n_train              : 1,2,4,8,16,32 * 10K
  * n_valid              : n_train/4
  * n_features           : 256
  * n_clusters_per_class : 8
  * n_rounds             : 100
  * max_depth            : 5, 10
  * num_leaves           : 2 ** max_depth
