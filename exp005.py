"""
2016/12/31
exp name  : exp005
desciption: Comparison of XGB, XGB_GPU and LightGBM on arificial datasets
fname     : exp005.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 10**6, 2*10**6
  n_valid             : n_train/4
  n_features          : 28
  n_rounds            : 100
  n_clusters_per_class: 64
  max_depth           : 5, 10,11,12,13,14,15,16

                  Time(sec)              Ratio
                    XGB_CPU    LGB XGB_CPU/LGB
n_train max_depth                             
1000000 5              76.2    4.8        16.0
        10            161.0   12.8        12.5
        11            179.3   16.1        11.1
        12            193.0   21.4         9.0
        13            209.4   29.7         7.0
        14            230.8   44.7         5.2
        15            243.2   80.9         3.0
        16            264.3  163.7         1.6
2000000 5             207.7    9.6        21.7
        10            443.3   25.1        17.7
        11            464.7   29.6        15.7
        12            496.5   38.3        13.0
        13            537.8   50.7        10.6
        14            582.0   72.2         8.1
        15            629.2  123.3         5.1
        16            676.1  236.3         2.9

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
n_classes = 2
n_clusters_per_class = 64
n_rounds = 100

n_rounds = 16
fname_header = "exp005_"

for n_train in [10**5, 2*10**5]:
    n_valid = n_train / 4
    n_all = n_train + n_valid
    X, y = make_classification(n_samples=n_all, n_classes=n_classes, n_features=28,
                               n_informative=10, n_redundant=10,
                               n_clusters_per_class=n_clusters_per_class,
                               shuffle=True, random_state=123)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_valid = X[n_train:]
    y_valid = y[n_train:]
    for max_depth in [5, 10]:#,11,12,13,14,15,16]:
        fname_footer = "n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        params_xgb['max_depth'] = max_depth
        params_lgb['max_depth'] = max_depth
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print('\n')
        print(params[-1])
        time_sec_lst = experiment_binary(X_train, y_train, X_valid, y_valid,
                                         params_xgb, params_lgb, n_rounds=n_rounds,
                                         use_gpu=True,
                                         fname_header=fname_header, fname_footer=fname_footer,
                                         n_skip=15)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=['XGB_CPU', 'XGB_GPU', 'LGB']).join(pd.DataFrame(params))
df_time['XGB_CPU/LGB'] = df_time['XGB_CPU'] / df_time['LGB']
df_time['XGB_GPU/LGB'] = df_time['XGB_GPU'] / df_time['LGB']
df_time.set_index(['n_train', 'max_depth'], inplace=True)
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[[0,0,0,1,1],[0,1,2,3,4]])
df_time.to_csv('log/' + fname_header + 'time.csv')

pd.set_option('display.precision', 1)
print(df_time)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
