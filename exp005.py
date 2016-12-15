"""
2016/12/15 1.9h
exp name  : exp005
desciption: Comparison of XGB:CPU and LightGBM on arificial datasets
fname     : exp005.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 10**6, 2*10**6
  n_features          : 28
  n_rounds            : 100
  n_clusters_per_class: 64
  max_depth           : 5, 10,11,12,13,14,15,16

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'logloss', 'tree_method':'exact', 'threads':8,
              'max_depth':max_depth}

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : {'binary_logloss'}, 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0,
              'max_depth': max_depth+1, 'num_leaves' : 2**max_depth}

                  Time(sec)              Ratio
                    XGB_CPU   LGBM XGB_CPU/LGB
n_train max_depth                             
1000000 5              80.1    5.3        15.2
        10            160.1   13.7        11.7
        11            179.3   16.9        10.6
        12            204.7   22.5         9.1
        13            214.1   29.5         7.3
        14            232.8   48.5         4.8
        15            246.8   85.4         2.9
        16            262.4  167.4         1.6
2000000 5             198.1    9.6        20.7
        10            416.0   24.7        16.8
        11            462.6   29.6        15.6
        12            491.0   37.7        13.0
        13            535.4   51.4        10.4
        14            583.0   71.3         8.2
        15            635.2  124.7         5.1
        16            679.2  242.8         2.8

Done: 6857.598001 seconds

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
n_valid = 5*10**5
n_clusters_per_class = 64
n_rounds = 100
fname_header = "exp005_"

for n_train in [10**6, 2*10**6]:
    n_all = n_train + n_valid
    X, y = make_classification(n_samples=n_all, n_classes=n_classes, n_features=28,
                               n_informative=10, n_redundant=10,
                               n_clusters_per_class=n_clusters_per_class,
                               shuffle=True, random_state=123)
    for max_depth in [5, 10,11,12,13,14,15,16]:
        fname_footer = "n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        params_xgb['max_depth'] = max_depth
        params_lgb['max_depth'] = max_depth + 1
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print('\n')
        print(params[-1])
        time_sec_lst = experiment_binary(X[:n_train], y[:n_train], X[-n_valid:], y[-n_valid:],
                                         params_xgb, params_lgb, n_rounds=n_rounds,
                                         use_gpu=False,
                                         fname_header=fname_header, fname_footer=fname_footer,
                                         n_skip=15)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=['XGB_CPU', 'LGBM']).join(pd.DataFrame(params))
df_time['XGB_CPU/LGB'] = df_time['XGB_CPU'] / df_time['LGBM']
df_time.set_index(['n_train', 'max_depth'], inplace=True)
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[[0,0,1,],[0,1,2]])
df_time.to_csv('log/' + fname_header + 'time.csv')

pd.set_option('display.precision', 1)
print(df_time)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
