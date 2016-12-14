"""
2016/12/14 2.7h
exp name  : exp003
desciption: Comparison btw XGB:CPU, XGB:GPU, and LightGBM on partially observable arificial datasets
fname     : exp003.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
result    : logloss, Feature importance, Leaf counts, Time
params:
  n_rounds            : 50
  n_train             : 1000000
  # 28 out of n_features are used
  n_features          : 30, 50 
  max_depth           : 5,6,7, .., 15

The limit of updater_gpu:max_depth is 15.
xgboost.core.XGBoostError: [12:45:46] /home/tks/download/xgboost/plugin/updater_gpu/src/gpu_builder.cu:157: Check failed: param.max_depth < 16 Tree depth too large.

Time
                      XGB_CPU  XGB_GPU  LGBM
n_features max_depth                        
30         5            176.7     29.8   3.3
           6            211.6     32.8   4.1
           7            246.6     36.2   5.2
           8            284.5     38.7   6.3
           9            325.1     40.8   8.0
           10           363.1     43.7   9.7
           11           401.9     46.3  12.5
           12           442.0     48.4  16.4
           13           479.5     51.3  23.9
           14           524.7     54.5  39.1
           15           552.8     57.1  72.2
50         5            174.8     29.5   3.4
           6            213.9     32.9   4.1
           7            250.4     35.4   5.2
           8            288.0     38.2   6.5
           9            325.9     40.7   7.9
           10           362.7     43.4   9.8
           11           405.8     46.0  12.9
           12           442.7     48.5  16.2
           13           479.4     51.2  23.6
           14           518.9     54.7  38.7
           15           558.1     57.8  75.2

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'logloss', 'tree_method':'exact', 'threads':8,
              'max_depth':max_depth}

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : {'binary_logloss'}, 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0,
              'max_depth': max_depth+1, 'num_leaves' : 2**max_depth}

"""
import pandas as pd
import time
time_begin = time.time()

from sklearn.datasets import make_classification, make_blobs, make_gaussian_quantiles

from utility import experiment_binary
from data_path import data_path

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'logloss', 'tree_method':'exact', 'threads':8}

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : {'binary_logloss'}, 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0}

params = []
times = []
n_classes = 2
n_train = 1000000
n_valid = 500000
n_all = n_train + n_valid
n_clusters_per_class = 8
n_rounds = 50
fname_header = "exp003_"
for n_features in [30, 50]:
    X, y = make_classification(n_samples=n_all, n_classes=n_classes,
                               n_features=n_features,
                               n_informative=n_features, n_redundant=0,
                               n_clusters_per_class=n_clusters_per_class,
                               shuffle=True, random_state=456)
    # Partially observable situation
    X = X[:,:28]
    for max_depth in range(5, 16):
        fname_footer = "n_features_%d_max_depth_%d.csv" % (n_features, max_depth)
        params_xgb['max_depth'] = max_depth
        params_lgb['max_depth'] = max_depth + 1
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_features':n_features, 'max_depth':max_depth})
        print('')
        print(params[-1])
        time_sec_lst = experiment_binary(X[:n_train], y[:n_train], X[-n_valid:], y[-n_valid:],
                                         params_xgb, params_lgb, n_rounds=n_rounds,
                                         fname_header=fname_header, fname_footer=fname_footer,
                                         n_skip=15)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=['XGB_CPU', 'XGB_GPU', 'LGBM']).join(pd.DataFrame(params))
df_time.to_csv("log/" + fname_header + "time.csv")

pd.set_option('display.precision', 1)
print("\n\nTime")
print(df_time.set_index(['n_features', 'max_depth']))

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
