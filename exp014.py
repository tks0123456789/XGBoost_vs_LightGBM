"""
2018-10-13 
exp name  : exp014
desciption: Comparison of XGBoost(hist_dw, hist_lg, hist_GPU, GPU and LightGBM on arificial datasets
XGBoost   : 0.80
LightGBM  : 2.2.1
fname     : exp014.py
env       : i7 4790k, 32G, GTX1070, ubuntu 16.04.5 LTS, CUDA V9.0, Python 3.6.6
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 1,2,4,8,16,32,64 * 10K
  n_valid             : n_train/4
  n_features          : 256
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10

                  Time(sec)                                     Ratio  \
                    hist_dw hist_lg hist_GPU    GPU   LGB hist_dw/LGB   
n_train max_depth                                                       
10000   5               6.0     6.0      0.9    4.1   1.0         6.3   
        10             43.2    42.2      5.9    8.2   6.4         6.8   
20000   5               7.5     7.6      0.9    8.5   1.4         5.5   
        10             61.0    63.5      7.7   18.2   9.7         6.3   
40000   5               9.3     9.5      1.2   18.6   2.3         4.1   
        10             83.8    84.5     10.2   39.1  14.9         5.6   
80000   5              12.1    12.5      1.5   38.3   3.8         3.2   
        10            100.0   109.3     11.5   79.2  20.8         4.8   
160000  5              19.1    18.5      2.2   76.9   7.1         2.7   
        10            140.9   134.5     15.1  162.9  33.9         4.2   
320000  5              31.4    26.6      3.6  163.6  13.6         2.3   
        10            178.8   165.0     18.6  344.4  56.0         3.2   
640000  5              44.0    44.6      6.3  374.7  26.9         1.6   
        10            210.6   206.7     23.7  776.6  91.6         2.3   

                                                    
                  hist_lg/LGB hist_GPU/LGB GPU/LGB  
n_train max_depth                                   
10000   5                 6.2          1.0     4.3  
        10                6.6          0.9     1.3  
20000   5                 5.5          0.7     6.2  
        10                6.5          0.8     1.9  
40000   5                 4.2          0.5     8.2  
        10                5.7          0.7     2.6  
80000   5                 3.3          0.4    10.1  
        10                5.3          0.6     3.8  
160000  5                 2.6          0.3    10.9  
        10                4.0          0.4     4.8  
320000  5                 2.0          0.3    12.0  
        10                2.9          0.3     6.1  
640000  5                 1.7          0.2    14.0  
        10                2.3          0.3     8.5  

Logloss
                   hist_dw  hist_lg  hist_GPU      GPU      LGB  hist_lg-LGB
n_train max_depth                                                           
10000   5          0.53911  0.53911   0.53822  0.54564  0.53780      0.00131
        10         0.47608  0.47608   0.47249  0.47945  0.48651     -0.01044
20000   5          0.49771  0.49771   0.50148  0.50566  0.49986     -0.00215
        10         0.40974  0.40974   0.41694  0.42189  0.41443     -0.00469
40000   5          0.49625  0.49625   0.49621  0.49787  0.49582      0.00043
        10         0.37420  0.37420   0.38280  0.38021  0.37856     -0.00436
80000   5          0.50465  0.50465   0.50723  0.50824  0.50578     -0.00113
        10         0.37047  0.37047   0.37348  0.37458  0.37506     -0.00459
160000  5          0.49950  0.49950   0.49611  0.49876  0.49913      0.00037
        10         0.34682  0.34682   0.34887  0.35071  0.34703     -0.00021
320000  5          0.50055  0.50055   0.50006  0.50148  0.49925      0.00129
        10         0.34555  0.34555   0.34589  0.34804  0.34567     -0.00012
640000  5          0.49771  0.49771   0.49666  0.49778  0.49727      0.00044
        10         0.33634  0.33634   0.33521  0.33614  0.33437      0.00197

Done: 4435.7 seconds
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

params_xgb_gpu = params_xgb_cpu.copy()
params_xgb_gpu.update({'tree_method': 'gpu_exact'})

params_xgb_gpu_hist = params_xgb_cpu.copy()
params_xgb_gpu.update({'tree_method' : 'gpu_hist',
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
