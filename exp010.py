"""
2017/1/14 1.25h
exp name  : exp010
desciption: Comparison of XGBoost and LightGBM on arificial datasets
fname     : exp010.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 0.5M, 1M, 2M
  n_valid             : n_train/4
  n_features          : 32
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10, 15
    The depth limit of grow_gpu is 15.

                  Time(sec)                                   Ratio  \
                        CPU EQBIN_dw EQBIN_lg    GPU    LGB CPU/LGB   
n_train max_depth                                                     
500000  5              32.5      6.6      6.5   14.3    2.7    12.2   
        10             66.8     26.3     26.1   26.6    7.3     9.2   
        15            102.8     95.6     93.7   41.3   41.1     2.5   
1000000 5              79.1     11.8     11.9   32.1    5.7    13.9   
        10            161.9     39.4     40.7   61.1   15.0    10.8   
        15            259.6    156.5    156.6   90.9   72.4     3.6   
2000000 5             214.4     21.5     22.0   71.4   11.3    19.0   
        10            456.1     60.9     58.5  140.1   31.1    14.7   
        15            742.0    237.8    236.3  206.1  122.4     6.1   

                                                     
                  EQBIN_dw/LGB EQBIN_lg/LGB GPU/LGB  
n_train max_depth                                    
500000  5                  2.5          2.4     5.4  
        10                 3.6          3.6     3.7  
        15                 2.3          2.3     1.0  
1000000 5                  2.1          2.1     5.7  
        10                 2.6          2.7     4.1  
        15                 2.2          2.2     1.3  
2000000 5                  1.9          1.9     6.3  
        10                 2.0          1.9     4.5  
        15                 1.9          1.9     1.7  

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
                           'updater':'grow_fast_histmaker',
                           'grow_policy':'depthwise',
                           'max_bin':255})

params_xgb_eqbin_l = params_xgb_eqbin_d.copy()
params_xgb_eqbin_l.update({'grow_policy':'lossguide'})

params_xgb_gpu = params_xgb.copy()
params_xgb_gpu.update({'updater':'grow_gpu'})

params_xgb_lst = [params_xgb, params_xgb_eqbin_d, params_xgb_eqbin_l, params_xgb_gpu]
model_str_lst = ['CPU', 'EQBIN_dw', 'EQBIN_lg', 'GPU']

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
fname_header = "exp010_"

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
    for max_depth in [5, 10, 15]:
        fname_footer = "n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        for params_xgb in params_xgb_lst:
            params_xgb.update({'max_depth':max_depth, 'max_leaves':2 ** max_depth})
        params_lgb['max_depth'] = max_depth
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print('\n')
        print(params[-1])
        time_sec_lst = experiment_binary_gb(X_train, y_train, X_valid, y_valid,
                                            params_xgb_lst, model_str_lst, params_lgb,
                                            n_rounds=n_rounds,
                                            fname_header=fname_header, fname_footer=fname_footer,
                                            n_skip=15)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_time.set_index(['n_train', 'max_depth'], inplace=True)
for model_str in model_str_lst:
    df_time[model_str + '/LGB'] = df_time[model_str] / df_time['LGB']
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[np.repeat([0, 1], [5, 4]), range(9)])
df_time.to_csv('log/' + fname_header + 'time.csv')

pd.set_option('display.precision', 1)
print('\n')
print(df_time)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
