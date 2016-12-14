# Comparison of XGB and LGBM

XGBoost  : [2b6aa77](https://github.com/dmlc/xgboost/tree/2b6aa7736febbad5243e4335be0640cd659d3ce5)

LightGBM : [ebfc852](https://github.com/Microsoft/LightGBM/tree/ebfc8521e217204f47cb53843bd56cf2c2395ffb)

The objective function of LGBM for binary classification is slightly different from XGB's.
* LGBM: -log(1+exp(-2 * sigmoid * label * score))
* XGB : -log(1+exp(-label * score))
  * label in {-1, 1}, score: leaf values
  * The default value of sigmoid is 1.0.

So LGBM's parameter sigmoid is set to 0.5.

LGBM's max_depth is not the maximum depth of a tree, **maximum depth + 1** !!!
