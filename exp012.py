"""
2017/4/15 4.72h
exp name  : exp012
desciption: Comparison of XGBoost and LightGBM on arificial datasets
XGBoost   : 8222755
LightGBM  : 9224a9d
fname     : exp012.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, Python 3.4.3
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 1, 2, 4, 8, 16, 32 * 10000
  n_valid             : n_train/4
  n_features          : 256
  n_rounds            : 100
  n_clusters_per_class: 8
  max_depth           : 5, 10, 15, 20
  num_leaves          : 32, 256, 1024, 4096

                             Time(sec)                        Ratio             
                              EQBIN_dw EQBIN_lg    LGB EQBIN_dw/LGB EQBIN_lg/LGB
n_train num_leaves max_depth                                                    
10000   32         5               6.1      6.0    0.9          7.0          6.9
                   10              6.5      7.0    1.5          4.4          4.7
                   15              6.4      6.6    1.6          4.1          4.3
                   20              6.4      6.7    1.6          4.0          4.2
        256        10             36.2     37.4    5.4          6.7          7.0
                   15             40.0     38.5    7.1          5.7          5.5
                   20             40.1     39.4    7.4          5.4          5.3
        1024       10             39.8     39.0    5.8          6.9          6.7
                   15             51.5     51.1    7.9          6.6          6.5
                   20             54.3     55.4    7.9          6.9          7.0
        4096       15             53.8     53.2    8.1          6.6          6.5
                   20             57.8     52.9    8.3          6.9          6.4
20000   32         5               7.5      7.0    1.2          6.2          5.8
                   10              7.4      7.7    2.0          3.7          3.9
                   15              7.8      7.8    2.1          3.8          3.8
                   20              7.7      7.8    2.1          3.7          3.7
        256        10             41.8     47.1    7.9          5.3          6.0
                   15             44.6     46.2   11.0          4.1          4.2
                   20             44.2     45.1   11.5          3.9          3.9
        1024       10             59.4     58.9    9.0          6.6          6.5
                   15             96.8     96.2   15.3          6.3          6.3
                   20             95.9     94.5   15.6          6.1          6.1
        4096       15            100.2    106.7   15.8          6.3          6.7
                   20            101.9     99.5   16.2          6.3          6.1
40000   32         5               8.8      8.8    2.0          4.4          4.4
                   10              8.9      9.1    3.2          2.7          2.8
                   15              8.9      9.4    3.4          2.6          2.8
                   20              8.7      9.4    3.3          2.7          2.9
        256        10             46.2     46.4   11.1          4.2          4.2
                   15             45.9     49.7   14.6          3.1          3.4
                   20             49.1     48.8   15.3          3.2          3.2
        1024       10             81.8     77.4   13.8          5.9          5.6
                   15            142.0    144.4   28.4          5.0          5.1
                   20            164.9    147.7   30.5          5.4          4.8
        4096       15            177.9    178.7   30.7          5.8          5.8
                   20            195.3    206.9   32.8          6.0          6.3
80000   32         5              11.8     11.3    3.5          3.4          3.2
                   10             11.7     12.1    5.6          2.1          2.2
                   15             11.9     12.3    5.8          2.0          2.1
                   20             11.6     12.2    5.9          2.0          2.1
        256        10             50.0     56.8   15.5          3.2          3.7
                   15             50.0     54.5   20.0          2.5          2.7
                   20             50.5     54.9   21.6          2.3          2.5
        1024       10            101.1    113.3   21.5          4.7          5.3
                   15            170.6    173.9   46.2          3.7          3.8
                   20            168.4    176.8   50.7          3.3          3.5
        4096       15            324.5    315.5   58.1          5.6          5.4
                   20            370.6    366.0   64.3          5.8          5.7
160000  32         5              17.3     16.6    6.3          2.7          2.6
                   10             17.1     18.3    9.9          1.7          1.9
                   15             16.4     18.4   10.4          1.6          1.8
                   20             17.0     18.0   10.3          1.6          1.7
        256        10             58.1     60.1   23.5          2.5          2.6
                   15             58.4     63.4   31.3          1.9          2.0
                   20             58.4     63.8   33.8          1.7          1.9
        1024       10            127.6    125.9   31.3          4.1          4.0
                   15            184.8    192.8   64.2          2.9          3.0
                   20            181.1    196.2   71.8          2.5          2.7
        4096       15            481.3    466.9   99.7          4.8          4.7
                   20            559.2    562.9  120.7          4.6          4.7
320000  32         5              26.8     26.3   12.3          2.2          2.1
                   10             26.7     29.1   18.6          1.4          1.6
                   15             26.3     29.3   19.5          1.4          1.5
                   20             26.7     29.7   19.4          1.4          1.5
        256        10             72.3     77.7   40.0          1.8          1.9
                   15             73.0     80.0   52.6          1.4          1.5
                   20             72.5     81.4   57.0          1.3          1.4
        1024       10            164.6    167.7   51.4          3.2          3.3
                   15            208.3    216.5   94.7          2.2          2.3
                   20            201.7    222.0  104.0          1.9          2.1
        4096       15            631.5    639.1  178.8          3.5          3.6
                   20            678.2    728.9  225.6          3.0          3.2

Logloss
                              EQBIN_dw  EQBIN_lg      LGB  EQBIN_lg-LGB
n_train num_leaves max_depth                                           
10000   32         5           0.51700   0.51700  0.51909      -0.00209
                   10          0.51539   0.48934  0.49083      -0.00149
                   15          0.51539   0.49345  0.49544      -0.00199
                   20          0.51539   0.49345  0.49742      -0.00396
        256        10          0.48099   0.47040  0.47638      -0.00598
                   15          0.46976   0.48407  0.48380       0.00026
                   20          0.47147   0.47651  0.49213      -0.01562
        1024       10          0.48824   0.48824  0.47971       0.00854
                   15          0.50802   0.50802  0.49974       0.00828
                   20          0.50235   0.50235  0.49903       0.00332
        4096       15          0.50802   0.50802  0.49974       0.00828
                   20          0.50235   0.50235  0.49903       0.00332
20000   32         5           0.48498   0.48498  0.48516      -0.00017
                   10          0.49006   0.45384  0.45863      -0.00479
                   15          0.49006   0.46019  0.45703       0.00316
                   20          0.49006   0.46019  0.45703       0.00316
        256        10          0.41712   0.40652  0.41063      -0.00411
                   15          0.41567   0.40439  0.40229       0.00210
                   20          0.41567   0.39952  0.40271      -0.00320
        1024       10          0.41889   0.41889  0.41902      -0.00013
                   15          0.43804   0.43988  0.43125       0.00863
                   20          0.43803   0.43803  0.42928       0.00875
        4096       15          0.43495   0.43495  0.42577       0.00918
                   20          0.44044   0.44044  0.43420       0.00624
40000   32         5           0.48737   0.48737  0.48700       0.00037
                   10          0.48358   0.45278  0.45187       0.00091
                   15          0.48358   0.45054  0.44949       0.00105
                   20          0.48358   0.45054  0.44949       0.00105
        256        10          0.40162   0.38528  0.38674      -0.00146
                   15          0.39870   0.36281  0.36311      -0.00029
                   20          0.39870   0.36013  0.36244      -0.00230
        1024       10          0.38736   0.38736  0.39074      -0.00338
                   15          0.37971   0.38583  0.38620      -0.00038
                   20          0.38107   0.38688  0.38827      -0.00139
        4096       15          0.39998   0.39998  0.40012      -0.00014
                   20          0.40725   0.40725  0.40450       0.00274
80000   32         5           0.48883   0.48883  0.48900      -0.00017
                   10          0.48883   0.45942  0.45688       0.00254
                   15          0.48883   0.45828  0.45526       0.00302
                   20          0.48883   0.45768  0.45526       0.00242
        256        10          0.39833   0.37319  0.37024       0.00295
                   15          0.39833   0.34429  0.34723      -0.00295
                   20          0.39833   0.33923  0.34058      -0.00136
        1024       10          0.36545   0.36545  0.36638      -0.00093
                   15          0.34449   0.33274  0.33050       0.00225
                   20          0.34595   0.32861  0.32985      -0.00125
        4096       15          0.35353   0.35394  0.35266       0.00128
                   20          0.36075   0.36347  0.35864       0.00482
160000  32         5           0.47945   0.47945  0.47910       0.00035
                   10          0.47967   0.44586  0.44694      -0.00108
                   15          0.47967   0.44452  0.44542      -0.00090
                   20          0.47967   0.44437  0.44707      -0.00271
        256        10          0.38776   0.35757  0.35594       0.00163
                   15          0.38776   0.32587  0.32600      -0.00013
                   20          0.38776   0.31888  0.31919      -0.00031
        1024       10          0.34355   0.34355  0.34585      -0.00230
                   15          0.32481   0.29398  0.29164       0.00234
                   20          0.32481   0.28267  0.28414      -0.00147
        4096       15          0.30413   0.30583  0.30423       0.00160
                   20          0.30449   0.30583  0.30554       0.00029
320000  32         5           0.47197   0.47197  0.47262      -0.00065
                   10          0.47197   0.43967  0.44064      -0.00097
                   15          0.47197   0.43980  0.43918       0.00062
                   20          0.47197   0.43980  0.43895       0.00085
        256        10          0.37130   0.33812  0.33905      -0.00093
                   15          0.37130   0.31415  0.31414       0.00002
                   20          0.37130   0.30707  0.30718      -0.00010
        1024       10          0.32412   0.32412  0.32133       0.00279
                   15          0.30822   0.26802  0.26670       0.00132
                   20          0.30822   0.25330  0.25390      -0.00059
        4096       15          0.25883   0.25535  0.25477       0.00057
                   20          0.25541   0.24699  0.24534       0.00165

Done: 16988.32298707962 seconds
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

params_xgb_lst = [params_xgb_eqbin_d, params_xgb_eqbin_l]
model_str_lst = ['EQBIN_dw', 'EQBIN_lg']

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
fname_header = "exp012_"

N = 10 ** 4
for n_train in [N, 2*N, 4*N, 8*N, 16*N, 32*N]:
    print(n_train)
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
    for max_depth in [5, 10, 15, 20]:
        for num_leaves in [32, 256, 1024, 4096]:
            if num_leaves > 2 ** max_depth:
                continue
            fname_footer = "n_%d_md_%d_nl_%d.csv" % (n_train, max_depth, num_leaves)
            for params_xgb in params_xgb_lst:
                params_xgb.update({'max_depth':max_depth, 'max_leaves':num_leaves})
            params_lgb.update({'max_depth':max_depth, 'num_leaves':num_leaves})
            params.append({'n_train':n_train, 'max_depth':max_depth, 'num_leaves':num_leaves})
            print('\n')
            print(params[-1])
            time_sec_s, sc_valid_s = experiment_binary_gb(X_train, y_train, X_valid, y_valid,
                                                params_xgb_lst, model_str_lst, params_lgb,
                                                n_rounds=n_rounds,
                                                fname_header=fname_header, fname_footer=fname_footer,
                                                n_skip=15)
            times.append(time_sec_s)
            valid_scores.append(sc_valid_s)

keys = ['n_train', 'num_leaves', 'max_depth']

df_times = pd.DataFrame(times, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_times = df_times.sort_values(keys).set_index(keys)
for model_str in model_str_lst:
    df_times[model_str + '/LGB'] = df_times[model_str] / df_times['LGB']
df_times.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_times.columns]],
                                labels=[np.repeat([0, 1], [3, 2]), range(5)])
df_times.to_csv('log/' + fname_header + 'times.csv')

df_valid_scores = pd.DataFrame(valid_scores, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_valid_scores = df_valid_scores.sort_values(keys).set_index(keys)

A, B = model_str_lst[1], "LGB"
df_valid_scores[A + "-" + B] = df_valid_scores[A] - df_valid_scores[B]
df_valid_scores.to_csv('log/' + fname_header + 'valid_scores.csv')

pd.set_option("display.max_rows",75)

pd.set_option('display.precision', 1)
pd.set_option('display.width', 100)
print('\n')
print(df_times)

pd.set_option('display.precision', 5)
pd.set_option('display.width', 100)
print('\nLogloss')
print(df_valid_scores)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
