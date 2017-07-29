"""
2017/7/29 1.3h
exp name  : exp014
desciption: Comparison of XGBoost(hist_dw, hist_lg, hist_GPU, GPU and LightGBM on arificial datasets
XGBoost   : 0e06d18(2017/7/29)
LightGBM  : 2e82123(2017/7/28)
fname     : exp014.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, Python 3.4.3
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 1,2,4,8,16,32,64 * 10K
  n_valid             : n_train/4
  n_features          : 256
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10

                  Time(sec)                                      Ratio                           \
                    hist_dw hist_lg hist_GPU    GPU    LGB hist_dw/LGB hist_lg/LGB hist_GPU/LGB   
n_train max_depth                                                                                 
10000   5               6.2     6.1      0.9    4.0    1.2         5.2         5.2          0.8   
        10             39.2    38.2      2.8    7.8    7.7         5.1         5.0          0.4   
20000   5               7.4     7.2      1.1    9.3    1.8         4.2         4.1          0.6   
        10             61.3    60.5      3.5   18.4   12.6         4.9         4.8          0.3   
40000   5               8.8     9.5      1.6   19.9    2.4         3.6         3.9          0.7   
        10             81.4    85.9      4.6   40.3   18.9         4.3         4.6          0.2   
80000   5              12.8    12.3      2.8   41.1    4.2         3.1         2.9          0.7   
        10            108.4   108.6      6.9   83.8   27.4         4.0         4.0          0.3   
160000  5              20.9    20.6      4.9   81.6    7.4         2.8         2.8          0.7   
        10            142.0   139.0     11.3  167.5   39.1         3.6         3.6          0.3   
320000  5              34.5    38.3      9.2  163.1   14.1         2.5         2.7          0.7   
        10            185.9   172.2     20.2  339.6   63.7         2.9         2.7          0.3   
640000  5              62.6    60.4     17.4  390.7   27.6         2.3         2.2          0.6   
        10            236.0   228.6     37.6  814.1  104.3         2.3         2.2          0.4   

                           
                  GPU/LGB  
n_train max_depth          
10000   5             3.4  
        10            1.0  
20000   5             5.3  
        10            1.5  
40000   5             8.2  
        10            2.1  
80000   5             9.8  
        10            3.1  
160000  5            11.0  
        10            4.3  
320000  5            11.6  
        10            5.3  
640000  5            14.2  
        10            7.8  

Logloss
                   hist_dw  hist_lg  hist_GPU      GPU      LGB  hist_lg-LGB
n_train max_depth                                                           
10000   5          0.51700  0.51700   0.51747  0.51580  0.51501      0.00200
        10         0.48824  0.48824   0.48797  0.48589  0.48747      0.00078
20000   5          0.48498  0.48498   0.48879  0.49112  0.48647     -0.00149
        10         0.41889  0.41889   0.41667  0.42195  0.41875      0.00014
40000   5          0.48737  0.48737   0.48637  0.48928  0.48720      0.00017
        10         0.38736  0.38736   0.39077  0.38908  0.38680      0.00055
80000   5          0.48883  0.48883   0.48891  0.49044  0.49114     -0.00231
        10         0.36545  0.36545   0.36418  0.37158  0.36813     -0.00268
160000  5          0.47945  0.47945   0.47948  0.48007  0.47908      0.00037
        10         0.34355  0.34355   0.34438  0.34903  0.34747     -0.00392
320000  5          0.47197  0.47197   0.47197  0.47336  0.47204     -0.00007
        10         0.32412  0.32412   0.32399  0.32506  0.32309      0.00104
640000  5          0.47742  0.47742   0.47636  0.47874  0.47593      0.00149
        10         0.32615  0.32615   0.32618  0.32693  0.32677     -0.00062

Done: 4676.4 seconds
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
n_features = 256
n_informative = n_redundant = n_features // 4
n_rounds = 100
fname_header = "exp014_"

N = 10**4
for n_train in [N, 2*N, 4*N, 8*N, 16*N, 32*N, 64*N]:
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
    for max_depth in [5, 10]:
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
