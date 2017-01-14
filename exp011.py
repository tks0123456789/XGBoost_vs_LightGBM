"""
2017/1/14 1.23h
exp name  : exp011
desciption: Comparison of XGBoost and LightGBM on arificial datasets
fname     : exp011.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 0.5M, 1M, 2M
  n_valid             : n_train/4
  n_features          : 32
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10, 15, 20
  num_leaves          : 256, 1024, 4096

                             Time(sec)                       Ratio             
                              EQBIN_dw EQBIN_lg   LGB EQBIN_dw/LGB EQBIN_lg/LGB
n_train max_depth num_leaves                                                   
500000  5         256              6.8      6.6   2.7          2.5          2.4
                  1024             6.5      6.5   2.8          2.4          2.3
                  4096             6.6      6.4   2.9          2.3          2.2
        10        256             16.1     17.0   6.1          2.7          2.8
                  1024            26.9     26.2   7.1          3.8          3.7
                  4096            26.9     26.1   7.9          3.4          3.3
        15        256             16.3     18.4   6.7          2.4          2.7
                  1024            40.6     43.8  12.1          3.4          3.6
                  4096            84.0     86.7  19.5          4.3          4.4
        20        256             16.0     17.5   7.1          2.3          2.5
                  1024            40.7     45.4  13.0          3.1          3.5
                  4096           111.0    112.7  26.7          4.2          4.2
1000000 5         256             12.2     11.7   5.7          2.1          2.0
                  1024            12.2     11.6   5.7          2.1          2.0
                  4096            11.8     11.8   5.9          2.0          2.0
        10        256             23.7     25.3  12.5          1.9          2.0
                  1024            39.9     39.0  14.8          2.7          2.6
                  4096            39.0     39.1  15.5          2.5          2.5
        15        256             23.3     27.0  14.6          1.6          1.9
                  1024            51.2     56.0  22.4          2.3          2.5
                  4096           118.2    121.4  34.6          3.4          3.5
        20        256             23.0     26.2  14.9          1.5          1.8
                  1024            51.3     57.2  23.5          2.2          2.4
                  4096           136.7    152.0  44.4          3.1          3.4
2000000 5         256             21.5     21.9  11.6          1.9          1.9
                  1024            22.0     22.0  11.5          1.9          1.9
                  4096            22.2     22.3  11.6          1.9          1.9
        10        256             36.7     41.5  25.2          1.5          1.6
                  1024            61.2     60.1  30.4          2.0          2.0
                  4096            60.9     60.7  30.6          2.0          2.0
        15        256             35.8     43.2  29.1          1.2          1.5
                  1024            71.5     78.9  43.0          1.7          1.8
                  4096           157.6    161.9  61.2          2.6          2.6
        20        256             37.6     43.4  29.3          1.3          1.5
                  1024            70.6     78.7  43.5          1.6          1.8
                  4096           164.2    184.9  71.8          2.3          2.6

"""
import pandas as pd
import numpy as np
import time
time_begin = time.time()

from sklearn.datasets import make_classification

from utility import experiment_binary_gb
from data_path import data_path

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'tree_method':'exact', 'updater':'grow_colmaker',
              'eval_metric':'logloss',
              'silent':True, 'threads':8}

params_xgb_eqbin_d = params_xgb.copy()
params_xgb_eqbin_d.update({'tree_method':'hist',
                           'updater':'grow_fast_histmaker,prune',
                           'grow_policy':'depthwise',
                           'max_bin':255})

params_xgb_eqbin_l = params_xgb_eqbin_d.copy()
params_xgb_eqbin_l.update({'grow_policy':'lossguide'})

params_xgb_gpu = params_xgb.copy()
params_xgb_gpu.update({'updater':'grow_gpu'})

params_xgb_lst = [params_xgb_eqbin_d, params_xgb_eqbin_l]
model_str_lst = ['CPU', 'EQBIN_dw', 'EQBIN_lg', 'GPU']
model_str_lst = ['EQBIN_dw', 'EQBIN_lg']

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : 'binary_logloss', 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0}

params = []
times = []
n_classes = 2
n_clusters_per_class = 8
n_features = 32
n_informative = n_redundant = n_features // 4
n_rounds = 100
fname_header = "exp011_"

for n_train in [5*10**5, 10**6, 2*10**6]:
    n_valid = n_train // 4
    n_all = n_train + n_valid
    X, y = make_classification(n_samples=n_all, n_classes=n_classes,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               n_clusters_per_class=n_clusters_per_class,
                               shuffle=True, random_state=123)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_valid = X[n_train:]
    y_valid = y[n_train:]
    for max_depth in [5, 10, 15, 20]:
        for num_leaves in [256, 1024, 4096]:
            fname_footer = "n_%d_md_%d_nl_%d.csv" % (n_train, max_depth, num_leaves)
            for params_xgb in params_xgb_lst:
                params_xgb.update({'max_depth':max_depth, 'max_leaves':num_leaves})
            params_lgb.update({'max_depth':max_depth, 'num_leaves':num_leaves})
            params.append({'n_train':n_train, 'max_depth':max_depth, 'num_leaves':num_leaves})
            print('\n')
            print(params[-1])
            time_sec_lst = experiment_binary_gb(X_train, y_train, X_valid, y_valid,
                                                params_xgb_lst, model_str_lst, params_lgb,
                                                n_rounds=n_rounds,
                                                fname_header=fname_header, fname_footer=fname_footer,
                                                n_skip=15)
            times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_time.set_index(['n_train', 'max_depth', 'num_leaves'], inplace=True)
for model_str in model_str_lst:
    df_time[model_str + '/LGB'] = df_time[model_str] / df_time['LGB']
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[np.repeat([0, 1], [3, 2]), range(5)])
df_time.to_csv('log/' + fname_header + 'time.csv')

pd.set_option('display.precision', 1)
pd.set_option('display.width', 100)
print('\n')
print(df_time)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
