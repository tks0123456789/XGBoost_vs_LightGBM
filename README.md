# Comparison of XGB and LGBM

The objective function of LGBM for binary classification is slightly different from XGB's.
LGBM: -log(1+exp(-2 * sigmoid * label * score))
XGB : -log(1+exp(-label * score))
  label in {-1, 1}
  score : leaf values
  The default value of sigmoid is 1.0.
So LGBM's parameter sigmoid is set to 0.5.

LGBM's max_depth is not the maximum depth of a tree, max_depth + 1 !!!
