### Comparison of XGB 2017/1/30:[179b384](https://github.com/dmlc/xgboost/tree/179b384e396a8897340c080d9af79af215c1caaf) and LGB 2017/1/30:[cba87c3](https://github.com/Microsoft/LightGBM/tree/cba87c377bd8da1ffa59d484d1f224a9cb6a88e7)

~~The objective function of LGB for binary classification is slightly different from XGB's.~~

* XGB : -log(1+exp(-label * score))
* ~~LGB: -log(1+exp(-2 * sigmoid * label * score))~~
* LGB: -log(1+exp(-sigmoid * label * score))
  * Changed by [46d4eec](https://github.com/Microsoft/LightGBM/commit/46d4eecf2e20ed970fa4f1dbfcf6b146c19a7597)
  * label in {-1, 1}, score: leaf values
  * The default value of sigmoid is 1.0.

~~So LGB's parameter sigmoid is set to 0.5.~~

* LightGBM.ipynb: Modified version of [marugari's work](https://github.com/marugari/Notebooks/blob/ed6aa7835579ce9143850ed5956912895c984d56/LightGBM.ipynb)
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

