"""
2017/11/03 1.18h
exp name  : exp014
desciption: Comparison of XGBoost(hist_dw, hist_lg, hist_GPU, GPU and LightGBM on arificial datasets
XGBoost   : a8f670d(2017/11/02)
LightGBM  : 7a166fb(2017/11/01)
fname     : exp014.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, CUDA V8.0.61, Python 3.4.3
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 1,2,4,8,16,32,64 * 10K
  n_valid             : n_train/4
  n_features          : 256
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10

                  Time(sec)                                     Ratio                           \
                    hist_dw hist_lg hist_GPU    GPU   LGB hist_dw/LGB hist_lg/LGB hist_GPU/LGB   
n_train max_depth                                                                                
10000   5               6.4     5.8      1.1    4.0   0.9         7.2         6.6          1.3   
        10             37.4    37.3      5.0    7.8   6.7         5.6         5.6          0.7   
20000   5               7.2     7.8      1.3    8.8   1.3         5.6         6.1          1.0   
        10             64.9    57.4      5.9   17.0  10.4         6.2         5.5          0.6   
40000   5               9.6     8.8      1.8   18.2   2.1         4.6         4.2          0.9   
        10             79.6    81.9      7.0   37.7  15.3         5.2         5.4          0.5   
80000   5              12.0    12.9      3.1   37.4   3.7         3.2         3.5          0.8   
        10            108.1   108.5      9.7   76.9  22.8         4.7         4.8          0.4   
160000  5              18.8    17.8      5.5   75.5   6.9         2.7         2.6          0.8   
        10            132.8   140.2     14.5  156.7  33.3         4.0         4.2          0.4   
320000  5              28.4    28.4     10.3  150.9  13.3         2.1         2.1          0.8   
        10            164.3   171.2     25.2  316.8  53.9         3.0         3.2          0.5   
640000  5              55.7    48.8     19.7  308.8  26.4         2.1         1.9          0.7   
        10            218.8   229.8     44.4  647.7  92.5         2.4         2.5          0.5   

                           
                  GPU/LGB  
n_train max_depth          
10000   5             4.5  
        10            1.2  
20000   5             6.9  
        10            1.6  
40000   5             8.7  
        10            2.5  
80000   5            10.1  
        10            3.4  
160000  5            10.9  
        10            4.7  
320000  5            11.3  
        10            5.9  
640000  5            11.7  
        10            7.0  

Logloss
                   hist_dw  hist_lg  hist_GPU      GPU      LGB  hist_lg-LGB
n_train max_depth                                                           
10000   5          0.51700  0.51700   0.51901  0.51580  0.51524      0.00176
        10         0.48824  0.48824   0.47722  0.48589  0.48564      0.00260
20000   5          0.48498  0.48498   0.48992  0.49112  0.48647     -0.00149
        10         0.41889  0.41889   0.42292  0.42195  0.41891     -0.00002
40000   5          0.48737  0.48737   0.48796  0.48928  0.48720      0.00017
        10         0.38736  0.38736   0.38987  0.38908  0.38666      0.00070
80000   5          0.48883  0.48883   0.48966  0.49044  0.49113     -0.00230
        10         0.36545  0.36545   0.36644  0.37158  0.36813     -0.00268
160000  5          0.47945  0.47945   0.47941  0.48007  0.47908      0.00037
        10         0.34355  0.34355   0.34750  0.34903  0.34593     -0.00238
320000  5          0.47197  0.47197   0.47197  0.47336  0.47204     -0.00007
        10         0.32412  0.32412   0.32395  0.32506  0.32315      0.00097
640000  5          0.47742  0.47742   0.47714  0.47874  0.47593      0.00149
        10         0.32615  0.32615   0.32648  0.32693  0.32528      0.00087

Done: 4244.7 seconds
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
