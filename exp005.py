"""
2016/12/31 2.5h
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
  max_depth           : 5,6,.. ,15
    The depth limit of grow_gpu is 15.

                  Time(sec)                      Ratio            
                    XGB_CPU XGB_GPU    LGB XGB_CPU/LGB XGB_GPU/LGB
n_train max_depth                                                 
1000000 5              77.8    28.4    4.8        16.3         5.9
        6              94.2    33.3    5.8        16.3         5.8
        7             107.1    37.8    7.1        15.0         5.3
        8             129.2    42.3    8.3        15.6         5.1
        9             139.4    46.9   10.2        13.6         4.6
        10            155.7    51.3   12.5        12.5         4.1
        11            182.0    55.8   16.1        11.3         3.5
        12            199.6    60.4   21.3         9.4         2.8
        13            216.1    64.9   29.1         7.4         2.2
        14            229.3    69.8   44.2         5.2         1.6
        15            248.7    75.1   83.5         3.0         0.9
2000000 5             199.7    63.4    9.5        21.0         6.7
        6             233.6    74.7   11.3        20.6         6.6
        7             274.6    85.6   13.8        19.8         6.2
        8             316.6    96.5   16.7        19.0         5.8
        9             362.5   107.2   20.3        17.8         5.3
        10            406.9   118.6   26.1        15.6         4.6
        11            449.6   128.8   30.5        14.8         4.2
        12            495.4   139.5   37.3        13.3         3.7
        13            537.0   150.2   49.1        10.9         3.1
        14            577.7   161.1   70.6         8.2         2.3
        15            627.6   172.3  121.0         5.2         1.4

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
fname_header = "exp005_"

for n_train in [10**6, 2*10**6]:
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
    for max_depth in range(5, 16):
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
