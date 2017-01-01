"""
2017/1/1-2 11.2h
exp name  : exp009
desciption: Comparison of XGB, XGB_GPU and LightGBM on arificial datasets
fname     : exp009.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 0.5M, 1M, 2M
  n_valid             : n_train/4
  n_features          : 16, 32, 64
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 2,3,.. ,15
    The depth limit of grow_gpu is 15.

                              Time(sec)                  Ratio                
                               XGB_CPU XGB_GPU    LGB CPU/LGB GPU/LGB CPU/GPU
n_train n_features max_depth                                                 
500000  16         2               7.6     3.9    1.2     6.1     3.2     1.9
                   4              13.3     6.7    1.5     8.7     4.4     2.0
                   8              26.7    13.2    2.7     9.9     4.9     2.0
                   15             54.0    24.4   29.2     1.8     0.8     2.2
        32         2              13.0     6.2    1.5     8.4     4.0     2.1
                   4              24.7    11.1    2.0    12.1     5.4     2.2
                   8              50.4    21.9    4.6    10.9     4.7     2.3
                   15            101.2    41.4   42.9     2.4     1.0     2.4
        64         2              24.8    11.7    2.3    10.9     5.2     2.1
                   4              47.7    22.3    3.2    14.9     6.9     2.1
                   8              97.4    41.8    8.2    11.9     5.1     2.3
                   15            197.4    75.3   75.1     2.6     1.0     2.6
2000000 16         2              43.9    17.2    4.8     9.1     3.6     2.6
                   4              84.2    32.2    6.5    13.0     5.0     2.6
                   8             171.7    66.1   12.9    13.3     5.1     2.6
                   15            338.2   117.1   66.4     5.1     1.8     2.9
        32         2              84.0    29.7    6.3    13.4     4.7     2.8
                   4             162.6    56.7    8.7    18.8     6.6     2.9
                   8             332.7   113.7   19.4    17.1     5.9     2.9
                   15            664.5   205.6  110.8     6.0     1.9     3.2
        64         2             162.6    57.7    9.3    17.4     6.2     2.8
                   4             316.9   113.6   13.3    23.9     8.6     2.8
                   8             641.5   222.7   31.9    20.1     7.0     2.9
                   15           1321.4   390.6  198.4     6.7     2.0     3.4

"""
import pandas as pd
import time
time_begin = time.time()

from sklearn.datasets import make_classification

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
n_train = 10 ** 6
n_classes = 2
n_clusters_per_class = 8
n_rounds = 100
fname_header = "exp009_"

for n_train in [500000, 10**6, 2*10**6]:
    n_valid = n_train / 4
    n_all = n_train + n_valid
    for n_features in [16, 32, 64]:
        n_informative = n_redundant = n_features / 4
        X, y = make_classification(n_samples=n_all,
                                   n_classes=n_classes,
                                   n_features=n_features,
                                   n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_clusters_per_class=n_clusters_per_class,
                                   shuffle=True, random_state=123)
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_valid = X[n_train:]
        y_valid = y[n_train:]
        for max_depth in range(2, 16):
            fname_footer = "n_train_%d_n_features_%d_max_depth_%d.csv" % (n_train, n_features, max_depth)
            params_xgb['max_depth'] = max_depth
            params_lgb['max_depth'] = max_depth
            params_lgb['num_leaves'] = 2 ** max_depth
            params.append({'n_train':n_train, 'n_features':n_features, 'max_depth':max_depth})
            print('\n')
            print(params[-1])
            time_sec_lst = experiment_binary(X_train, y_train, X_valid, y_valid,
                                             params_xgb, params_lgb, n_rounds=n_rounds,
                                             use_gpu=True,
                                             fname_header=fname_header, fname_footer=fname_footer,
                                             n_skip=15)
            times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=['XGB_CPU', 'XGB_GPU', 'LGB']).join(pd.DataFrame(params))
df_time['CPU/LGB'] = df_time['XGB_CPU'] / df_time['LGB']
df_time['GPU/LGB'] = df_time['XGB_GPU'] / df_time['LGB']
df_time['CPU/GPU'] = df_time['XGB_CPU'] / df_time['XGB_GPU']
df_time.set_index(['n_train', 'n_features', 'max_depth'], inplace=True)
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[[0,0,0,1,1,1],[0,1,2,3,4,5]])
df_time.to_csv('log/' + fname_header + 'time.csv')

idx = pd.IndexSlice
df_time.loc[idx[[500000, 2000000],:,[2,4,8,15]]]

pd.set_option('display.precision', 1)
pd.set_option('display.width', 100)
print(df_time.loc[idx[[500000, 2000000],:,[2,4,8,15]]])

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
