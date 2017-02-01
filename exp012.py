"""
2017/2/1 8h
exp name  : exp012
desciption: Comparison of XGBoost and LightGBM on arificial datasets
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
10000   32         5              14.1     11.7    0.7         20.2         16.8
                   10             12.7     13.2    1.2         10.7         11.1
                   15             12.5     13.6    1.2         10.3         11.1
                   20             12.3     13.2    1.2         10.1         10.8
        256        10             62.3     64.9    4.6         13.5         14.0
                   15             72.3     69.4    6.0         12.0         11.5
                   20             71.8     71.1    6.2         11.7         11.5
        1024       10             68.4     69.0    5.2         13.2         13.3
                   15             88.6     89.0    6.9         12.9         12.9
                   20             94.1     93.8    7.0         13.4         13.4
        4096       15             89.9     90.7    8.2         11.0         11.1
                   20             95.0     92.6    7.9         12.0         11.7
20000   32         5              13.5     13.2    1.0         13.5         13.2
                   10             13.8     14.6    1.6          8.7          9.2
                   15             13.7     14.5    1.6          8.4          8.9
                   20             13.5     14.6    1.6          8.3          9.0
        256        10             79.5     82.8    6.5         12.2         12.7
                   15             81.0     88.6    8.9          9.1         10.0
                   20             80.0     89.1    9.1          8.8          9.8
        1024       10            107.7    105.0    7.8         13.8         13.5
                   15            166.9    160.2   13.5         12.4         11.9
                   20            171.8    165.4   14.3         12.0         11.5
        4096       15            171.0    168.4   14.8         11.6         11.4
                   20            178.0    177.2   15.0         11.9         11.8
40000   32         5              15.5     16.2    1.7          9.3          9.7
                   10             15.6     17.3    2.6          6.1          6.7
                   15             15.6     17.4    2.7          5.9          6.6
                   20             15.6     16.6    2.8          5.5          5.9
        256        10             85.3     95.4    9.2          9.3         10.4
                   15             87.1    100.1   12.0          7.2          8.3
                   20             85.7     99.2   12.8          6.7          7.8
        1024       10            146.2    145.0   12.1         12.1         12.0
                   15            258.7    260.5   25.0         10.3         10.4
                   20            267.8    267.0   26.6         10.0         10.0
        4096       15            321.5    318.7   28.3         11.3         11.3
                   20            346.5    342.4   30.4         11.4         11.3
80000   32         5              18.8     18.8    3.1          6.1          6.1
                   10             18.8     20.0    4.8          3.9          4.1
                   15             19.2     20.0    5.1          3.8          3.9
                   20             19.9     22.3    5.1          3.9          4.4
        256        10             93.6    100.5   13.4          7.0          7.5
                   15             93.9    105.7   17.3          5.4          6.1
                   20             93.5    106.9   18.6          5.0          5.8
        1024       10            190.5    191.4   18.5         10.3         10.3
                   15            312.4    350.8   40.3          7.8          8.7
                   20            314.1    349.8   43.5          7.2          8.0
        4096       15            604.4    585.2   54.0         11.2         10.8
                   20            633.7    623.1   59.6         10.6         10.5
160000  32         5              25.4     24.8    5.6          4.5          4.4
                   10             24.8     26.7    8.8          2.8          3.0
                   15             25.1     27.1    9.2          2.7          2.9
                   20             25.3     27.0    9.2          2.7          2.9
        256        10            104.0    112.3   20.6          5.0          5.4
                   15            104.7    118.1   27.6          3.8          4.3
                   20            105.1    124.6   29.9          3.5          4.2
        1024       10            238.0    239.2   27.9          8.5          8.6
                   15            339.7    385.8   55.8          6.1          6.9
                   20            340.9    394.8   62.1          5.5          6.4
        4096       15            866.7    865.5   91.4          9.5          9.5
                   20           1002.5    993.4  110.3          9.1          9.0
320000  32         5              37.0     37.1   11.1          3.3          3.3
                   10             36.8     39.9   16.8          2.2          2.4
                   15             37.5     39.1   17.5          2.1          2.2
                   20             37.0     39.6   17.5          2.1          2.3
        256        10            124.4    131.7   36.4          3.4          3.6
                   15            124.1    137.2   47.7          2.6          2.9
                   20            124.0    146.7   51.8          2.4          2.8
        1024       10            301.8    300.5   46.1          6.5          6.5
                   15            371.0    416.3   81.2          4.6          5.1
                   20            387.5    426.7   92.4          4.2          4.6
        4096       15           1165.6   1209.1  151.4          7.7          8.0
                   20           1297.9   1348.4  188.3          6.9          7.2

Logloss
                              EQBIN_dw  EQBIN_lg      LGB  EQBIN_lg-LGB
n_train num_leaves max_depth                                           
10000   32         5           0.51700   0.51700  0.51909      -0.00209
                   10          0.51539   0.48934  0.49080      -0.00146
                   15          0.51539   0.49346  0.49538      -0.00192
                   20          0.51539   0.49346  0.49735      -0.00390
        256        10          0.48099   0.47040  0.47634      -0.00594
                   15          0.46976   0.48406  0.47903       0.00504
                   20          0.47147   0.47651  0.47724      -0.00074
        1024       10          0.48824   0.48824  0.47765       0.01059
                   15          0.50802   0.50802  0.50071       0.00731
                   20          0.50235   0.50235  0.49290       0.00946
        4096       15          0.50802   0.50802  0.50071       0.00731
                   20          0.50235   0.50235  0.49290       0.00946
20000   32         5           0.48499   0.48499  0.48518      -0.00019
                   10          0.49006   0.45384  0.45863      -0.00479
                   15          0.49006   0.46019  0.45703       0.00316
                   20          0.49006   0.46019  0.45703       0.00316
        256        10          0.41712   0.40652  0.41064      -0.00412
                   15          0.41567   0.40439  0.40229       0.00210
                   20          0.41567   0.39952  0.40272      -0.00320
        1024       10          0.41889   0.41889  0.41131       0.00758
                   15          0.43804   0.43988  0.43620       0.00369
                   20          0.43803   0.43803  0.42998       0.00805
        4096       15          0.43495   0.43495  0.43246       0.00249
                   20          0.44044   0.44044  0.43890       0.00154
40000   32         5           0.48737   0.48737  0.48699       0.00037
                   10          0.48358   0.45278  0.45187       0.00091
                   15          0.48358   0.45054  0.44949       0.00105
                   20          0.48358   0.45054  0.44949       0.00105
        256        10          0.40162   0.38528  0.38675      -0.00147
                   15          0.39870   0.36281  0.36310      -0.00028
                   20          0.39870   0.36013  0.36245      -0.00232
        1024       10          0.38736   0.38736  0.38814      -0.00079
                   15          0.37971   0.38583  0.38874      -0.00292
                   20          0.38107   0.38688  0.38779      -0.00091
        4096       15          0.39998   0.39998  0.39628       0.00369
                   20          0.40724   0.40724  0.40493       0.00231
80000   32         5           0.48883   0.48883  0.48900      -0.00017
                   10          0.48883   0.45942  0.45688       0.00254
                   15          0.48883   0.45828  0.45526       0.00302
                   20          0.48883   0.45768  0.45526       0.00242
        256        10          0.39833   0.37319  0.37025       0.00294
                   15          0.39833   0.34429  0.34724      -0.00295
                   20          0.39833   0.33923  0.34058      -0.00136
        1024       10          0.36546   0.36546  0.36565      -0.00020
                   15          0.34449   0.33274  0.33049       0.00225
                   20          0.34595   0.32861  0.32985      -0.00124
        4096       15          0.35353   0.35394  0.35175       0.00219
                   20          0.36075   0.36347  0.35864       0.00483
160000  32         5           0.47945   0.47945  0.47910       0.00035
                   10          0.47967   0.44586  0.44694      -0.00108
                   15          0.47967   0.44452  0.44542      -0.00090
                   20          0.47967   0.44437  0.44707      -0.00271
        256        10          0.38776   0.35757  0.35594       0.00164
                   15          0.38776   0.32587  0.32600      -0.00013
                   20          0.38776   0.31888  0.31919      -0.00031
        1024       10          0.34355   0.34355  0.34766      -0.00411
                   15          0.32481   0.29398  0.29165       0.00234
                   20          0.32481   0.28267  0.28415      -0.00147
        4096       15          0.30413   0.30583  0.30451       0.00132
                   20          0.30449   0.30583  0.30461       0.00122
320000  32         5           0.47197   0.47197  0.47262      -0.00065
                   10          0.47197   0.43967  0.44064      -0.00097
                   15          0.47197   0.43980  0.43918       0.00062
                   20          0.47197   0.43980  0.43895       0.00085
        256        10          0.37130   0.33812  0.33905      -0.00093
                   15          0.37130   0.31415  0.31414       0.00002
                   20          0.37130   0.30707  0.30718      -0.00010
        1024       10          0.32412   0.32412  0.32422      -0.00010
                   15          0.30822   0.26802  0.26670       0.00132
                   20          0.30822   0.25330  0.25390      -0.00059
        4096       15          0.25883   0.25535  0.25518       0.00017
                   20          0.25541   0.24698  0.24624       0.00075

Done: 28713.359662771225 seconds

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
