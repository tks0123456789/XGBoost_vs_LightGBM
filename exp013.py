"""
2017/7/29 2.3h
exp name  : exp013
desciption: Comparison of XGBoost(hist_dw, hist_lg, hist_GPU, GPU and LightGBM on arificial datasets
XGBoost   : 0e06d18(2017/7/29)
LightGBM  : 2e82123(2017/7/28)
fname     : exp013.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, Python 3.4.3
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 0.5M, 1M, 2M, 4M
  n_valid             : n_train/4
  n_features          : 32
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10, 15

                  Time(sec)                                       Ratio                           \
                    hist_dw hist_lg hist_GPU     GPU    LGB hist_dw/LGB hist_lg/LGB hist_GPU/LGB   
n_train max_depth                                                                                  
500000  5               5.2     5.4      2.7    34.7    3.3         1.6         1.6          0.8   
        10             15.7    15.8      3.5    69.6    9.3         1.7         1.7          0.4   
        15             53.9    52.8     11.3   115.2   33.0         1.6         1.6          0.3   
1000000 5              10.4    10.6      4.6    88.3    6.8         1.5         1.6          0.7   
        10             25.8    26.3      6.7   178.4   17.0         1.5         1.5          0.4   
        15             89.8    89.2     15.9   296.6   60.9         1.5         1.5          0.3   
2000000 5              19.5    20.1      9.2   204.9   13.7         1.4         1.5          0.7   
        10             44.5    44.4     13.1   453.7   34.8         1.3         1.3          0.4   
        15            138.1   136.4     25.2   760.9   95.7         1.4         1.4          0.3   
4000000 5              38.3    38.7     18.7   520.5   28.2         1.4         1.4          0.7   
        10             75.6    72.4     25.5  1147.1   65.3         1.2         1.1          0.4   
        15            216.6   268.0     44.2  1867.2  177.4         1.2         1.5          0.2   

                           
                  GPU/LGB  
n_train max_depth          
500000  5            10.6  
        10            7.5  
        15            3.5  
1000000 5            13.0  
        10           10.5  
        15            4.9  
2000000 5            14.9  
        10           13.0  
        15            7.9  
4000000 5            18.5  
        10           17.6  
        15           10.5  

Logloss
                   hist_dw  hist_lg  hist_GPU      GPU      LGB  hist_lg-LGB
n_train max_depth                                                           
500000  5          0.40130  0.40130   0.40268  0.41211  0.41021     -0.00891
        10         0.27221  0.27221   0.27201  0.27344  0.26991      0.00230
        15         0.23004  0.23004   0.22946  0.23451  0.22714      0.00290
1000000 5          0.38365  0.38365   0.38455  0.38572  0.38339      0.00026
        10         0.25140  0.25140   0.25340  0.25409  0.25458     -0.00319
        15         0.20795  0.20795   0.20936  0.21071  0.20762      0.00033
2000000 5          0.38377  0.38377   0.38489  0.38389  0.38525     -0.00149
        10         0.23234  0.23234   0.22992  0.23623  0.23055      0.00179
        15         0.18010  0.18010   0.18229  0.18403  0.18206     -0.00196
4000000 5          0.43190  0.43190   0.42994  0.43203  0.42857      0.00333
        10         0.25177  0.25177   0.24653  0.25572  0.25278     -0.00101
        15         0.17420  0.17420   0.17979  0.17850  0.17661     -0.00242

Done: 8144.594601154327 seconds
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
params_xgb_gpu.update({'updater': 'grow_gpu'})

params_xgb_gpu_hist = params_xgb_cpu.copy()
params_xgb_gpu.update({'updater' : 'grow_gpu_hist',
                       'max_bin' : 255,  #default=256
                   })

params_xgb_lst = [params_xgb_eqbin_d,
                  params_xgb_eqbin_l,
                  params_xgb_gpu,
                  params_xgb_gpu_hist]

model_str_lst = ['hist_dw', 'hist_lg', 'hist_GPU', 'GPU']

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
fname_header = "exp013_"

for n_train in [5*10**5, 10**6, 2*10**6, 4*10**6]:
    n_valid = n_train // 4
    n_all = n_train + n_valid
    params_lgb['bin_construct_sample_cnt'] = n_train # default=50000
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
        num_leaves = 2 ** max_depth
        fname_footer = "n_%d_md_%d.csv" % (n_train, max_depth)
        for params_xgb in params_xgb_lst:
            params_xgb.update({'max_depth':max_depth, 'max_leaves':num_leaves})
        params_lgb.update({'max_depth':max_depth, 'num_leaves':num_leaves})
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print('\n')
        print(params[-1])
        time_sec_s, sc_valid_s = experiment_binary_gb(X_train, y_train, X_valid, y_valid,
                                                      params_xgb_lst, model_str_lst, params_lgb,
                                                      n_rounds=n_rounds,
                                                      fname_header=fname_header,
                                                      fname_footer=fname_footer,
                                                      n_skip=15)
        times.append(time_sec_s)
        valid_scores.append(sc_valid_s)

keys = ['n_train', 'max_depth']

df_times = pd.DataFrame(times, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_times = df_times.sort_values(keys).set_index(keys)
for model_str in model_str_lst:
    df_times[model_str + '/LGB'] = df_times[model_str] / df_times['LGB']
df_times.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_times.columns]],
                                 labels=[np.repeat([0, 1], [5, 4]), range(9)])
df_times.to_csv('log/' + fname_header + 'times.csv')

df_valid_scores = pd.DataFrame(valid_scores, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_valid_scores = df_valid_scores.sort_values(keys).set_index(keys)

A, B = model_str_lst[1], "LGB"
df_valid_scores[A + "-" + B] = df_valid_scores[A] - df_valid_scores[B]
df_valid_scores.to_csv('log/' + fname_header + 'valid_scores.csv')

pd.set_option('display.precision', 1)
pd.set_option('display.width', 100)
print('\n')
print(df_times)

pd.set_option('display.precision', 5)
pd.set_option('display.width', 100)
print('\nLogloss')
print(df_valid_scores)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
