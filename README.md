# All the timings of XGB_CPU are incorrect.
# Building xgboost with gpu enabled may slow XGB_CPU

### Comparison of XGB:[2b6aa77](https://github.com/dmlc/xgboost/tree/2b6aa7736febbad5243e4335be0640cd659d3ce5) and LGBM:[ebfc852](https://github.com/Microsoft/LightGBM/tree/ebfc8521e217204f47cb53843bd56cf2c2395ffb)

The objective function of LGBM for binary classification is slightly different from XGB's.
* LGBM: -log(1+exp(-2 * sigmoid * label * score))
* XGB : -log(1+exp(-label * score))
  * label in {-1, 1}, score: leaf values
  * The default value of sigmoid is 1.0.

So LGBM's parameter sigmoid is set to 0.5.

LGBM's max_depth is not the maximum depth of a tree, **maximum depth + 1** !!!

* exp001: [Higgs dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS)
  * n_train              : 10^5, 10^6, 10^7
  * n_features           : 28
  * max_depth            : 5, 10, 15
* exp002: Artificial datasets
  * n_train              : 10^5, 10^6, 10^7
  * n_features           : 28
  * max_depth            : 5, 10, 15
  * n_clusters_per_class : 8, 16
* exp003: Partially obserbable artificial datasets
  * n_train              : 10^6
  * n_features_used      : 28
  * n_features           : 30, 50
  * max_depth            : 5, 6, .., 15

* LightGBM.ipynb: Changed version of [murugari's work](https://github.com/marugari/Notebooks/blob/ed6aa7835579ce9143850ed5956912895c984d56/LightGBM.ipynb)
