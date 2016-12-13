"""
2016/12/13-14 not done
exp name  : exp002
desciption: Comparison btw XGB:CPU, XGB:GPU, and LightGBM on arificial datasets
fname     : xgb_vs_lgbm_001.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_rounds            : 50
  n_clusters_per_class: 8, 64
  n_train             : 10**5, 10**6, 10**7
  max_depth           : 5, 10, 15

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
n_valid = 500000
n_rounds = 50
fname_header = "exp002_"
for n_train in [10**5, 10**6, 10**7]:
    for n_clusters_per_class in [8, 64]:
        n_all = n_train + n_valid
        X, y = make_classification(n_samples=n_all, n_classes=n_classes, n_features=28,
                                   n_informative=10, n_redundant=10,
                                   n_clusters_per_class=n_clusters_per_class,
                                   shuffle=True, random_state=123)
        for max_depth in [5, 10, 15]:
            fname_footer = "n_train_%d_n_clusters_per_class_%d_max_depth_%d.csv" % (n_train, n_clusters_per_class, max_depth)
            params_xgb['max_depth'] = max_depth
            params_lgb['max_depth'] = max_depth + 1
            params_lgb['num_leaves'] = 2 ** max_depth
            params.append({'n_train':n_train, 'n_clusters_per_class':n_clusters_per_class, 'max_depth':max_depth})
            print('')
            print(params[-1])
            time_sec_lst = experiment_binary(X[:n_train], y[:n_train], X[-n_valid:], y[-n_valid:],
                                             params_xgb, params_lgb, n_rounds=n_rounds,
                                             fname_header=fname_header, fname_footer=fname_footer,
                                             n_skip=15)
            times.append(time_sec_lst)

pd.set_option('display.precision', 2)
print("\n\nTime")
print(pd.DataFrame(times, columns=['XGB_CPU', 'XGB_GPU', 'LGBM']).join(pd.DataFrame(params)).set_index(['n_train', 'n_clusters_per_class', 'max_depth']))

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
