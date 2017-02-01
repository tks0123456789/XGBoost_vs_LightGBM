"""
2017/2/1 1.25h
exp name  : exp010
desciption: Comparison of XGBoost and LightGBM on arificial datasets
fname     : exp010.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, Python 3.4.3
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
500000  5              32.1      6.5      6.4   14.4    2.7    12.1   
        10             64.6     26.5     26.1   26.8    7.1     9.1   
        15            103.4     94.2     93.7   41.4   42.1     2.5   
1000000 5              82.6     11.6     11.7   32.3    5.7    14.5   
        10            176.2     39.0     39.2   61.2   15.2    11.6   
        15            261.0    157.6    154.1   91.3   71.1     3.7   
2000000 5             218.3     21.8     22.0   71.6   11.2    19.4   
        10            420.8     60.3     60.3  140.2   30.6    13.8   
        15            743.3    230.5    234.4  206.5  119.3     6.2   

                                                     
                  EQBIN_dw/LGB EQBIN_lg/LGB GPU/LGB  
n_train max_depth                                    
500000  5                  2.5          2.4     5.4  
        10                 3.7          3.7     3.8  
        15                 2.2          2.2     1.0  
1000000 5                  2.0          2.1     5.7  
        10                 2.6          2.6     4.0  
        15                 2.2          2.2     1.3  
2000000 5                  1.9          2.0     6.4  
        10                 2.0          2.0     4.6  
        15                 1.9          2.0     1.7  

Logloss
                       CPU  EQBIN_dw  EQBIN_lg      GPU      LGB  EQBIN_lg-LGB  CPU-GPU
n_train max_depth                                                                      
500000  5          0.41211   0.40130   0.40130  0.40914  0.40684      -0.00554  0.00297
        10         0.27344   0.27221   0.27221  0.27172  0.27255      -0.00034  0.00172
        15         0.23451   0.23004   0.23004  0.23499  0.22925       0.00079 -0.00048
1000000 5          0.38572   0.38365   0.38365  0.38489  0.38332       0.00033  0.00083
        10         0.25409   0.25140   0.25140  0.25184  0.24728       0.00412  0.00225
        15         0.21071   0.20795   0.20795  0.21131  0.20929      -0.00134 -0.00059
2000000 5          0.38389   0.38377   0.38377  0.38338  0.38279       0.00097  0.00051
        10         0.23623   0.23234   0.23234  0.23034  0.23107       0.00127  0.00589
        15         0.18403   0.18010   0.18010  0.18393  0.18132      -0.00123  0.00010

Done: 4487.143693447113 seconds

"""
import pandas as pd
import numpy as np
import time
time_begin = time.time()

from sklearn.datasets import make_classification

from utility import experiment_binary_gb
from data_path import data_path

params_xgb_cpu = {'objective'       : 'binary:logistic',
                  'eval_metric'     : 'logloss',
                  'tree_method'     : 'exact',
                  'updater'         : 'grow_colmaker',
                  'eta'             : 0.1, #default=0.3
                  'lambda'          : 1, #default
                  'min_child_weight': 1, #default
                  'silent'          : True,
                  'threads'         : 8}

params_xgb_eqbin_d = params_xgb_cpu.copy()
params_xgb_eqbin_d.update({'tree_method': 'hist',
                           'updater'    : 'grow_fast_histmaker',
                           'grow_policy': 'depthwise',
                           'max_bin'    : 255,  #default=256
                       })

params_xgb_eqbin_l = params_xgb_eqbin_d.copy()
params_xgb_eqbin_l.update({'grow_policy': 'lossguide'})

params_xgb_gpu = params_xgb_cpu.copy()
params_xgb_gpu.update({'updater':'grow_gpu'})

params_xgb_lst = [params_xgb_cpu, params_xgb_eqbin_d, params_xgb_eqbin_l, params_xgb_gpu]
model_str_lst = ['CPU', 'EQBIN_dw', 'EQBIN_lg', 'GPU']

params_lgb = {'task'                   : 'train',
              'objective'              : 'binary',
              'metric'                 : 'binary_logloss',
              'learning_rate'          : 0.1, #default
              'lambda_l2'              : 1, #default=0
              'sigmoid'                : 1, #default
              'min_data_in_leaf'       : 1, #default=100
              'min_sum_hessian_in_leaf': 1, #default=10.0
              'max_bin'                : 255, #default
              'num_threads'            : 8,
              'verbose'                : 0,
}

params = []
times = []
valid_scores = []

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
            params_xgb.update({'max_depth':max_depth})
        params_xgb_eqbin_l.update({'max_leaves':2 ** max_depth})
        params_lgb.update({'max_depth':max_depth, 'num_leaves':2 ** max_depth})
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print('\n')
        print(params[-1])
        time_sec_s, sc_valid_s = experiment_binary_gb(X_train, y_train, X_valid, y_valid,
                                            params_xgb_lst, model_str_lst, params_lgb,
                                            n_rounds=n_rounds,
                                            fname_header=fname_header, fname_footer=fname_footer,
                                            n_skip=15)
        times.append(time_sec_s)
        valid_scores.append(sc_valid_s)

df_time = pd.DataFrame(times, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_time.set_index(['n_train', 'max_depth'], inplace=True)
for model_str in model_str_lst:
    df_time[model_str + '/LGB'] = df_time[model_str] / df_time['LGB']
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[np.repeat([0, 1], [5, 4]), range(9)])
df_time.to_csv('log/' + fname_header + 'time.csv')

df_valid_scores = pd.DataFrame(valid_scores, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_valid_scores.set_index(['n_train', 'max_depth'], inplace=True)
for A, B in [['EQBIN_lg', 'LGB'], ['CPU', 'GPU']]:
    df_valid_scores[A + "-" + B] = df_valid_scores[A] - df_valid_scores[B]

df_valid_scores.to_csv('log/' + fname_header + 'valid_scores.csv')

pd.set_option('display.precision', 1)
print('\n')
print(df_time)

pd.set_option('display.precision', 5)
pd.set_option('display.width', 100)
print('\nLogloss')
print(df_valid_scores)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
