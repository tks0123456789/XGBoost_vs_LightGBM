"""
2017/5/5 2.2h
exp name  : exp013
desciption: Comparison of XGBoost(hist_dw, hist_lg, hist_GPU, GPU and LightGBM on arificial datasets
XGBoost   : 197a9ea(2017/5/3)
LightGBM  : b54f60f(2017/5/5)
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
500000  5               4.7     4.6      3.4    35.0    3.0         1.6         1.5          1.1   
        10             14.3    14.4      5.5    73.6    7.9         1.8         1.8          0.7   
        15             50.3    51.1     32.9   105.3   26.5         1.9         1.9          1.2   
1000000 5               8.9     8.9      5.9    88.0    6.5         1.4         1.4          0.9   
        10             22.9    23.4      9.5   172.4   16.1         1.4         1.5          0.6   
        15             85.4    84.8     40.4   283.7   50.5         1.7         1.7          0.8   
2000000 5              17.2    17.0     11.1   206.2   12.8         1.3         1.3          0.9   
        10             38.5    37.7     17.4   437.5   31.8         1.2         1.2          0.5   
        15            127.3   126.1     54.8   667.8   84.3         1.5         1.5          0.7   
4000000 5              32.5    33.1     21.6   506.3   25.8         1.3         1.3          0.8   
        10             61.8    63.9     32.4  1105.8   62.2         1.0         1.0          0.5   
        15            196.2   193.9     83.5  1801.2  147.8         1.3         1.3          0.6   

                           
                  GPU/LGB  
n_train max_depth          
500000  5            11.6  
        10            9.3  
        15            4.0  
1000000 5            13.5  
        10           10.7  
        15            5.6  
2000000 5            16.1  
        10           13.7  
        15            7.9  
4000000 5            19.6  
        10           17.8  
        15           12.2  

Logloss
                   hist_dw  hist_lg  hist_GPU      GPU      LGB  hist_lg-LGB
n_train max_depth                                                           
500000  5          0.40130  0.40130   0.40548  0.41211  0.41316     -0.01186
        10         0.27221  0.27221   0.27476  0.27344  0.27060      0.00160
        15         0.23004  0.23004   0.23038  0.23451  0.22953      0.00050
1000000 5          0.38365  0.38365   0.38455  0.38572  0.38275      0.00090
        10         0.25140  0.25140   0.25078  0.25409  0.25241     -0.00101
        15         0.20795  0.20795   0.20799  0.21071  0.20931     -0.00136
2000000 5          0.38377  0.38377   0.38352  0.38389  0.38185      0.00192
        10         0.23234  0.23234   0.22951  0.23623  0.23017      0.00217
        15         0.18010  0.18010   0.18146  0.18403  0.18046     -0.00036
4000000 5          0.43190  0.43190   0.43105  0.43203  0.42751      0.00440
        10         0.25177  0.25177   0.25373  0.25572  0.25264     -0.00088
        15         0.17420  0.17420   0.17860  0.17850  0.17648     -0.00228

Done: 7753.560648918152 seconds
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
