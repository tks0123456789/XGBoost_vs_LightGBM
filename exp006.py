"""
2016/12/18 not done
exp name  : exp006
Objective : Does equal freq binning improve accuracy?
desciption: Comparison of XGB:CPU and LightGBM on arificial datasets
          : Same as exp005 except with equal freq binning
fname     : exp006.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
preprocess: Equal frequency binning
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 10**6, 2*10**6
  n_valid             : n_train/4
  n_features          : 28
  n_rounds            : 100
  n_clusters_per_class: 64
  max_depth           : 5, 10,11,12,13,14,15,16

"""
import pandas as pd
import time
time_begin = time.time()

from sklearn.datasets import make_classification

from utility import experiment_binary, equal_frequency_binning
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
fname_header = "exp006_"

for n_train in [10**6, 2*10**6]:
    n_valid = n_train / 4
    n_all = n_train + n_valid
    X, y = make_classification(n_samples=n_all, n_classes=n_classes, n_features=28,
                               n_informative=10, n_redundant=10,
                               n_clusters_per_class=n_clusters_per_class,
                               shuffle=True, random_state=123)
    X_train, bins_lst = equal_frequency_binning(X[:n_train], q=255)
    X_valid, bins_lst = equal_frequency_binning(X[n_train:], bins_lst=bins_lst)
    y_train = y[:n_train]
    y_valid = y[n_train:]
    for max_depth in [5, 10,11,12,13,14,15,16]:
        fname_footer = "n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        params_xgb['max_depth'] = max_depth
        params_lgb['max_depth'] = max_depth
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print('\n')
        print(params[-1])
        time_sec_lst = experiment_binary(X_train, y_train, X_valid, y_valid,
                                         params_xgb, params_lgb, n_rounds=n_rounds,
                                         use_gpu=False,
                                         fname_header=fname_header, fname_footer=fname_footer,
                                         n_skip=15)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=['XGB_CPU', 'LGB']).join(pd.DataFrame(params))
df_time['XGB_CPU/LGB'] = df_time['XGB_CPU'] / df_time['LGB']
df_time.set_index(['n_train', 'max_depth'], inplace=True)
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[[0,0,1,],[0,1,2]])
df_time.to_csv('log/' + fname_header + 'time.csv')

pd.set_option('display.precision', 1)
print(df_time)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
