"""
2018-12-08
exp name  : exp014
desciption: Comparison of XGBoost(hist_dw, hist_lg, hist_GPU, GPU and LightGBM on arificial datasets
XGBoost   : 0.81
LightGBM  : 2.2.1
fname     : exp014.py
env       : i7 4790k, 32G, GTX1070, ubuntu 18.04.1 LTS, Python 3.6.6, nvidia driver:390.77
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 1,2,4,8,16,32 * 10K
  n_valid             : n_train/4
  n_features          : 256
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10

                  Time(sec)                                     Ratio  \
                    hist_dw hist_lg hist_GPU    GPU   LGB hist_dw/LGB   
n_train max_depth                                                       
10000   5               6.6     6.4      0.9    1.2   1.0         6.6   
        10             45.7    44.5      7.0    2.8   6.6         7.0   
20000   5               7.7     7.8      1.0    2.3   1.4         5.6   
        10             64.0    61.7      9.0    5.5   9.9         6.5   
40000   5               9.9     9.9      1.1    4.4   2.2         4.4   
        10             84.8    89.5     12.0   10.9  15.1         5.6   
80000   5              13.2    13.3      1.4    9.0   3.8         3.5   
        10            101.8   106.1     13.2   22.0  21.0         4.8   
160000  5              18.8    20.5      2.0   18.5   7.2         2.6   
        10            143.7   143.5     16.9   45.3  34.1         4.2   
320000  5              34.3    31.3      3.3   45.1  13.8         2.5   
        10            183.6   181.9     20.4  108.0  56.6         3.2   

                                                    
                  hist_lg/LGB hist_GPU/LGB GPU/LGB  
n_train max_depth                                   
10000   5                 6.4          0.9     1.2  
        10                6.8          1.1     0.4  
20000   5                 5.6          0.7     1.6  
        10                6.3          0.9     0.6  
40000   5                 4.4          0.5     2.0  
        10                5.9          0.8     0.7  
80000   5                 3.5          0.4     2.4  
        10                5.0          0.6     1.0  
160000  5                 2.9          0.3     2.6  
        10                4.2          0.5     1.3  
320000  5                 2.3          0.2     3.3  
        10                3.2          0.4     1.9  

Logloss
                   hist_dw  hist_lg  hist_GPU      GPU      LGB  hist_lg-LGB
n_train max_depth                                                           
10000   5          0.53911  0.53911   0.53822  0.54611  0.53780      0.00131
        10         0.47608  0.47608   0.47249  0.47544  0.48651     -0.01044
20000   5          0.49771  0.49771   0.50148  0.50595  0.49986     -0.00215
        10         0.40974  0.40974   0.41694  0.41970  0.41443     -0.00469
40000   5          0.49625  0.49625   0.49621  0.49955  0.49582      0.00043
        10         0.37602  0.37602   0.37829  0.37714  0.37856     -0.00254
80000   5          0.50465  0.50465   0.50723  0.50790  0.50578     -0.00113
        10         0.37047  0.37047   0.37347  0.37579  0.37506     -0.00459
160000  5          0.49950  0.49950   0.49611  0.49784  0.49913      0.00037
        10         0.34682  0.34682   0.34887  0.35013  0.34703     -0.00021
320000  5          0.50055  0.50055   0.50075  0.50148  0.49925      0.00129
        10         0.34555  0.34555   0.34644  0.34583  0.34567     -0.00012

Done: 1997.2 seconds
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
n_features = 256
n_informative = n_redundant = n_features // 4
n_rounds = 100
fname_header = "exp014_"

N = 10**4
for n_train in [N, 2*N, 4*N, 8*N, 16*N, 32*N]:
    n_valid = n_train // 4
    n_all = n_train + n_valid
    params_lgb['bin_construct_sample_cnt'] =  n_train # default=50000
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
