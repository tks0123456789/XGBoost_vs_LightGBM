"""
2017/4/15 2.38h
exp name  : exp011
desciption: Comparison of XGBoost and LightGBM on arificial datasets
XGBoost   : 8222755
LightGBM  : 9224a9d
fname     : exp011.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, Python 3.4.3
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 0.5M, 1M, 2M
  n_valid             : n_train/4
  n_features          : 32
  n_rounds            : 200
  n_clusters_per_class: 8
  max_depth           : 5, 10, 15, 20
  num_leaves          : 32, 256, 1024, 4096, 16384

                             Time(sec)                        Ratio             
                              EQBIN_dw EQBIN_lg    LGB EQBIN_dw/LGB EQBIN_lg/LGB
n_train num_leaves max_depth                                                    
500000  32         5               8.4      8.0    5.5          1.5          1.5
                   10              8.4      8.0    6.3          1.3          1.3
                   15              8.3      8.0    6.4          1.3          1.3
                   20              8.3      8.1    6.2          1.3          1.3
        256        10             18.3     17.7   11.7          1.6          1.5
                   15             18.2     17.3   12.1          1.5          1.4
                   20             18.0     16.8   12.2          1.5          1.4
        1024       10             26.1     25.7   14.1          1.8          1.8
                   15             42.6     41.4   23.3          1.8          1.8
                   20             43.4     42.2   25.0          1.7          1.7
        4096       15             77.2     76.3   35.5          2.2          2.1
                   20            112.6    113.8   54.8          2.1          2.1
        16384      15             84.3     83.8   40.1          2.1          2.1
                   20            149.4    148.7   67.7          2.2          2.2
1000000 32         5              16.2     15.7   11.3          1.4          1.4
                   10             16.0     15.5   13.3          1.2          1.2
                   15             15.7     15.8   13.5          1.2          1.2
                   20             16.3     15.5   13.5          1.2          1.2
        256        10             29.2     28.3   23.4          1.3          1.2
                   15             28.3     27.1   23.7          1.2          1.1
                   20             28.9     26.7   24.3          1.2          1.1
        1024       10             42.2     40.9   29.0          1.5          1.4
                   15             56.0     53.5   40.0          1.4          1.3
                   20             57.0     52.9   39.5          1.4          1.3
        4096       15            114.3    111.2   62.8          1.8          1.8
                   20            142.0    149.0   90.0          1.6          1.7
        16384      15            134.1    132.4   72.9          1.8          1.8
                   20            260.0    253.2  133.4          1.9          1.9
2000000 32         5              30.2     30.5   23.1          1.3          1.3
                   10             30.1     28.7   26.4          1.1          1.1
                   15             30.4     29.2   26.4          1.2          1.1
                   20             30.4     28.7   26.2          1.2          1.1
        256        10             48.6     51.7   47.8          1.0          1.1
                   15             48.9     45.7   48.4          1.0          0.9
                   20             48.3     44.6   49.0          1.0          0.9
        1024       10             70.0     68.5   59.3          1.2          1.2
                   15             82.9     80.7   69.4          1.2          1.2
                   20             83.2     71.9   68.0          1.2          1.1
        4096       15            159.0    157.1  116.4          1.4          1.3
                   20            179.8    181.0  131.8          1.4          1.4
        16384      15            207.1    203.9  132.1          1.6          1.5
                   20            397.4    381.5  236.0          1.7          1.6

Logloss
                              EQBIN_dw  EQBIN_lg      LGB  EQBIN_lg-LGB
n_train num_leaves max_depth                                           
500000  32         5           0.36296   0.36296  0.36228       0.00067
                   10          0.36084   0.32989  0.33306      -0.00317
                   15          0.36084   0.32803  0.32541       0.00262
                   20          0.36084   0.33081  0.33304      -0.00223
        256        10          0.26726   0.25364  0.25515      -0.00151
                   15          0.26803   0.25459  0.25560      -0.00101
                   20          0.26803   0.25180  0.25049       0.00131
        1024       10          0.24845   0.24845  0.24849      -0.00004
                   15          0.23223   0.22722  0.22681       0.00040
                   20          0.23134   0.22405  0.22303       0.00102
        4096       15          0.22261   0.22023  0.22109      -0.00086
                   20          0.22044   0.21882  0.21887      -0.00005
        16384      15          0.22148   0.22148  0.21995       0.00153
                   20          0.22389   0.22389  0.22266       0.00123
1000000 32         5           0.34270   0.34270  0.34309      -0.00039
                   10          0.34286   0.31639  0.31487       0.00152
                   15          0.34286   0.31226  0.31393      -0.00167
                   20          0.34286   0.31172  0.31282      -0.00110
        256        10          0.24967   0.23941  0.23956      -0.00015
                   15          0.25168   0.23304  0.23539      -0.00235
                   20          0.25168   0.23121  0.23212      -0.00092
        1024       10          0.22757   0.22757  0.22549       0.00208
                   15          0.21827   0.21009  0.20857       0.00153
                   20          0.21609   0.20795  0.20859      -0.00064
        4096       15          0.20322   0.20123  0.20092       0.00031
                   20          0.20022   0.19817  0.19807       0.00010
        16384      15          0.20134   0.20134  0.20146      -0.00012
                   20          0.20619   0.20639  0.20622       0.00016
2000000 32         5           0.33979   0.33979  0.33423       0.00555
                   10          0.33893   0.31382  0.30907       0.00474
                   15          0.33893   0.30837  0.30984      -0.00147
                   20          0.33893   0.31194  0.31050       0.00144
        256        10          0.23744   0.21472  0.21724      -0.00252
                   15          0.23744   0.21310  0.21347      -0.00037
                   20          0.23744   0.21204  0.20879       0.00325
        1024       10          0.20274   0.20274  0.19997       0.00277
                   15          0.19503   0.18647  0.18898      -0.00251
                   20          0.19324   0.18807  0.18549       0.00258
        4096       15          0.17666   0.17378  0.17400      -0.00022
                   20          0.17502   0.16870  0.16935      -0.00064
        16384      15          0.17191   0.17191  0.17236      -0.00045
                   20          0.17095   0.17057  0.17110      -0.00053

Done: 8556.412877321243 seconds
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
n_features = 32
n_informative = n_redundant = n_features // 4
n_rounds = 200
fname_header = "exp011_"

for n_train in [5*10**5, 10**6, 2*10**6]:
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
        for num_leaves in [32, 256, 1024, 4096, 16384]:
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

pd.set_option('display.precision', 1)
pd.set_option('display.width', 100)
print('\n')
print(df_times)

pd.set_option('display.precision', 5)
pd.set_option('display.width', 100)
print('\nLogloss')
print(df_valid_scores)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
