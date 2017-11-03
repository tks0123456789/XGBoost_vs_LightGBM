"""
2017/11/03 2.14h
exp name  : exp013
desciption: Comparison of XGBoost(hist_dw, hist_lg, hist_GPU, GPU and LightGBM on arificial datasets
XGBoost   : a8f670d(2017/11/02)
LightGBM  : 7a166fb(2017/11/01)
fname     : exp013.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, CUDA V8.0.61, Python 3.4.3
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
500000  5               5.4     5.2      2.3    32.7    3.2         1.7         1.6          0.7   
        10             15.9    15.9      3.4    65.0    8.2         1.9         1.9          0.4   
        15             52.6    52.0     18.6   103.4   30.6         1.7         1.7          0.6   
1000000 5              10.6    10.2      4.1    78.7    6.7         1.6         1.5          0.6   
        10             26.2    25.6      6.1   167.6   17.7         1.5         1.4          0.3   
        15             89.0    87.1     23.1   256.8   58.4         1.5         1.5          0.4   
2000000 5              19.3    19.9      8.1   201.2   13.5         1.4         1.5          0.6   
        10             42.0    41.6     11.8   446.2   34.0         1.2         1.2          0.3   
        15            135.2   133.3     32.1   682.3   97.1         1.4         1.4          0.3   
4000000 5              36.9    37.2     16.1   501.8   27.4         1.3         1.4          0.6   
        10             73.8    71.0     22.4  1112.9   63.5         1.2         1.1          0.4   
        15            209.3   207.9     50.3  1801.0  161.4         1.3         1.3          0.3   

                           
                  GPU/LGB  
n_train max_depth          
500000  5            10.3  
        10            7.9  
        15            3.4  
1000000 5            11.8  
        10            9.5  
        15            4.4  
2000000 5            14.9  
        10           13.1  
        15            7.0  
4000000 5            18.3  
        10           17.5  
        15           11.2  

Logloss
                   hist_dw  hist_lg  hist_GPU      GPU      LGB  hist_lg-LGB
n_train max_depth                                                           
500000  5          0.40130  0.40130   0.41027  0.41211  0.41021     -0.00891
        10         0.27221  0.27221   0.26919  0.27344  0.27293     -0.00072
        15         0.23004  0.23004   0.23097  0.23451  0.23118     -0.00115
1000000 5          0.38365  0.38365   0.38332  0.38572  0.38339      0.00026
        10         0.25140  0.25140   0.25133  0.25409  0.24857      0.00283
        15         0.20795  0.20795   0.20930  0.21071  0.20863     -0.00068
2000000 5          0.38377  0.38377   0.38352  0.38389  0.38525     -0.00149
        10         0.23234  0.23234   0.23245  0.23623  0.23336     -0.00103
        15         0.18010  0.18010   0.18206  0.18403  0.17999      0.00010
4000000 5          0.43190  0.43190   0.43215  0.43203  0.42857      0.00333
        10         0.25177  0.25177   0.25156  0.25572  0.24902      0.00274
        15         0.17420  0.17420   0.17662  0.17850  0.17744     -0.00324

Done: 7711.9 seconds
"""
import pandas as pd
import numpy as np
import time
t000 = time.time()

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

print('\nDone: {:.1f} seconds'.format(time.time() - t000))
