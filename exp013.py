"""
2018-12-08
exp name  : exp013
desciption: Comparison of XGBoost(hist_dw, hist_lg, hist_GPU, GPU and LightGBM on arificial datasets
XGBoost   : 0.81
LightGBM  : 2.2.1
fname     : exp013.py
env       : i7 4790k, 32G, GTX1070, ubuntu 18.04.1 LTS, Python 3.6.6, nvidia driver:390.77
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 0.5M, 1M, 2M
  n_valid             : n_train/4
  n_features          : 32
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10, 15

                  Time(sec)                                     Ratio  \
                    hist_dw hist_lg hist_GPU    GPU   LGB hist_dw/LGB   
n_train max_depth                                                       
10000   5               0.7     0.6      0.4    0.3   0.1         5.1   
        10              2.7     2.7      2.4    0.6   0.7         3.8   
        15              5.0     4.9      4.7    0.9   2.5         2.0   
500000  5               4.7     4.6      1.2   10.8   3.1         1.5   
        10             15.5    15.5      8.2   25.9   8.5         1.8   
        15             55.6    55.0     37.1   40.4  31.3         1.8   
1000000 5               9.0     8.7      2.0   25.2   6.4         1.4   
        10             24.4    25.0     10.6   59.0  17.4         1.4   
        15             96.4    94.9     54.0   91.7  56.5         1.7   
2000000 5              17.1    17.1      3.6   56.7  12.8         1.3   
        10             40.9    40.8     14.1  131.8  34.0         1.2   
        15            146.0   143.1     75.5  205.0  99.3         1.5   

                                                    
                  hist_lg/LGB hist_GPU/LGB GPU/LGB  
n_train max_depth                                   
10000   5                 4.2          2.9     1.9  
        10                3.7          3.3     0.8  
        15                1.9          1.8     0.4  
500000  5                 1.5          0.4     3.5  
        10                1.8          1.0     3.1  
        15                1.8          1.2     1.3  
1000000 5                 1.4          0.3     3.9  
        10                1.4          0.6     3.4  
        15                1.7          1.0     1.6  
2000000 5                 1.3          0.3     4.4  
        10                1.2          0.4     3.9  
        15                1.4          0.8     2.1  

Logloss
                   hist_dw  hist_lg  hist_GPU      GPU      LGB  hist_lg-LGB
n_train max_depth                                                           
10000   5          0.42634  0.42634   0.42403  0.42770  0.42632      0.00003
        10         0.37524  0.37524   0.36959  0.37794  0.37447      0.00077
        15         0.36915  0.36915   0.37087  0.37898  0.37369     -0.00454
500000  5          0.40130  0.40130   0.41284  0.41409  0.41021     -0.00891
        10         0.27221  0.27221   0.26997  0.27457  0.27266     -0.00045
        15         0.22980  0.22980   0.22965  0.23235  0.22885      0.00095
1000000 5          0.38365  0.38365   0.38523  0.38572  0.38339      0.00026
        10         0.25140  0.25140   0.25059  0.25351  0.25074      0.00066
        15         0.20795  0.20795   0.21004  0.21061  0.20967     -0.00172
2000000 5          0.38377  0.38377   0.38493  0.38322  0.38533     -0.00157
        10         0.23234  0.23234   0.23147  0.23591  0.23249     -0.00015
        15         0.18010  0.18010   0.18078  0.18518  0.18143     -0.00133

Done: 2076.7 seconds
"""
import pandas as pd
import numpy as np
import time
t000 = time.time()

from sklearn.datasets import make_classification

from utility import experiment_binary_gb

params_xgb_cpu = {'objective'       : 'binary:logistic',
                  'eval_metric'     : 'logloss',
                  'tree_method'     : 'exact',
                  # 'updater'         : 'grow_colmaker',
                  'eta'             : 0.1, #default=0.3
                  'lambda'          : 1, #default
                  'min_child_weight': 1, #default
                  'silent'          : True,
                  'threads'         : 8}

params_xgb_eqbin_d = params_xgb_cpu.copy()
params_xgb_eqbin_d.update({'tree_method': 'hist',
                           # 'updater'    : 'grow_fast_histmaker',
                           'grow_policy': 'depthwise',
                           'max_bin'    : 255,  #default=256
})

params_xgb_eqbin_l = params_xgb_eqbin_d.copy()
params_xgb_eqbin_l.update({'grow_policy': 'lossguide'})

params_xgb_gpu_hist = params_xgb_cpu.copy()
params_xgb_gpu_hist.update({'tree_method' : 'gpu_hist',
                            'max_bin' : 255,  #default=256
})

params_xgb_gpu = params_xgb_cpu.copy()
params_xgb_gpu.update({'tree_method': 'gpu_exact'})

params_xgb_lst = [params_xgb_eqbin_d,
                  params_xgb_eqbin_l,
                  params_xgb_gpu_hist,
                  params_xgb_gpu]

model_str_lst = ['hist_dw', 'hist_lg', 'hist_GPU', 'GPU']

params_lgb = {'task'                   : 'train',
              'objective'              : 'binary',
              'metric'                 : 'binary_logloss',
              'learning_rate'          : 0.1, #default
              'lambda_l2'              : 1, #default=0
              'sigmoid'                : 1, #default
              'min_data_in_leaf'       : 1, #default=20
              'min_sum_hessian_in_leaf': 1, #default=1e-3
              'max_bin'                : 255, #default
              'num_threads'            : 8,
              'verbose'                : -1,
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

for n_train in [10**4, 5*10**5, 10**6, 2*10**6]:
    n_valid = n_train // 4
    n_all = n_train + n_valid
    params_lgb['bin_construct_sample_cnt'] = n_train  # default=50000
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
            params_xgb.update({'max_depth': max_depth, 'max_leaves': num_leaves})
        params_lgb.update({'max_depth': max_depth, 'num_leaves': num_leaves})
        params.append({'n_train': n_train, 'max_depth': max_depth})
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
df_times.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],df_times.columns.tolist()],
                                 labels=[np.repeat([0, 1], [5, 4]), range(9)])
df_times.to_csv('log/' + fname_header + 'times.csv')

df_valid_scores = pd.DataFrame(valid_scores, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_valid_scores = df_valid_scores.sort_values(keys).set_index(keys)

A, B = model_str_lst[1], "LGB"
df_valid_scores[A + "-" + B] = df_valid_scores[A] - df_valid_scores[B]
df_valid_scores.to_csv('log/' + fname_header + 'valid_scores.csv')

pd.set_option('precision', 1)
pd.set_option('max_columns', 100)
print('\n')
print(df_times)

pd.set_option('precision', 5)
pd.set_option('max_columns', 100)
print('\nLogloss')
print(df_valid_scores)

print('\nDone: {:.1f} seconds'.format(time.time() - t000))
