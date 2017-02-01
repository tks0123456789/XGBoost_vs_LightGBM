"""
2017/2/1 3.65h
exp name  : exp011
desciption: Comparison of XGBoost and LightGBM on arificial datasets
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
n_train max_depth num_leaves                                                    
500000  5         32              12.2     11.9    5.2          2.3          2.3
        10        32              12.5     11.9    6.0          2.1          2.0
                  256             31.1     31.4   10.9          2.8          2.9
                  1024            47.7     46.5   14.0          3.4          3.3
        15        32              12.1     12.1    6.1          2.0          2.0
                  256             31.8     30.7   11.1          2.9          2.8
                  1024            79.1     81.5   21.7          3.7          3.8
                  4096           145.1    143.2   34.8          4.2          4.1
                  16384          156.7    155.5   49.8          3.1          3.1
        20        32              12.5     12.0    6.0          2.1          2.0
                  256             31.6     30.9   11.6          2.7          2.7
                  1024            78.9     83.9   22.4          3.5          3.7
                  4096           209.5    214.0   53.2          3.9          4.0
                  16384          267.9    263.6   88.0          3.0          3.0
1000000 5         32              22.6     22.1   10.8          2.1          2.1
        10        32              22.4     21.6   12.9          1.7          1.7
                  256             46.5     47.1   22.9          2.0          2.1
                  1024            73.4     77.4   27.1          2.7          2.9
        15        32              22.3     21.4   12.9          1.7          1.7
                  256             46.2     45.2   22.8          2.0          2.0
                  1024           101.0    114.0   37.5          2.7          3.0
                  4096           210.4    208.8   61.7          3.4          3.4
                  16384          245.4    243.5   86.3          2.8          2.8
        20        32              22.4     21.6   13.1          1.7          1.7
                  256             45.2     44.2   23.3          1.9          1.9
                  1024            99.2    101.3   37.1          2.7          2.7
                  4096           259.8    285.0   87.5          3.0          3.3
                  16384          467.3    450.9  165.3          2.8          2.7
2000000 5         32              42.2     40.4   22.5          1.9          1.8
        10        32              41.2     37.7   25.7          1.6          1.5
                  256             74.2     74.4   46.8          1.6          1.6
                  1024           115.0    109.5   57.8          2.0          1.9
        15        32              39.7     38.0   25.5          1.6          1.5
                  256             70.9     69.0   47.2          1.5          1.5
                  1024           133.4    134.1   65.4          2.0          2.1
                  4096           279.1    280.5  109.9          2.5          2.6
                  16384          366.9    360.0  149.6          2.5          2.4
        20        32              41.3     37.4   25.4          1.6          1.5
                  256             70.3     68.5   47.5          1.5          1.4
                  1024           134.5    128.6   64.7          2.1          2.0
                  4096           318.8    334.2  128.9          2.5          2.6
                  16384          700.2    676.2  274.1          2.6          2.5

Logloss
                              EQBIN_dw  EQBIN_lg      LGB  EQBIN_lg-LGB
n_train max_depth num_leaves                                           
500000  5         32           0.36296   0.36296  0.36228       0.00067
        10        32           0.36084   0.32989  0.33306      -0.00317
                  256          0.26726   0.25364  0.25515      -0.00151
                  1024         0.24845   0.24845  0.24307       0.00538
        15        32           0.36084   0.32803  0.32541       0.00262
                  256          0.26803   0.25459  0.25560      -0.00101
                  1024         0.23223   0.22722  0.22681       0.00041
                  4096         0.22261   0.22023  0.22158      -0.00135
                  16384        0.22148   0.22148  0.22219      -0.00071
        20        32           0.36084   0.33081  0.33304      -0.00223
                  256          0.26803   0.25180  0.25049       0.00131
                  1024         0.23134   0.22405  0.22303       0.00102
                  4096         0.22044   0.21882  0.21976      -0.00094
                  16384        0.22389   0.22389  0.22326       0.00063
1000000 5         32           0.34270   0.34270  0.34309      -0.00039
        10        32           0.34286   0.31639  0.31487       0.00152
                  256          0.24967   0.23941  0.23956      -0.00015
                  1024         0.22757   0.22757  0.22908      -0.00152
        15        32           0.34286   0.31226  0.31393      -0.00167
                  256          0.25168   0.23304  0.23539      -0.00235
                  1024         0.21827   0.21009  0.20857       0.00153
                  4096         0.20322   0.20123  0.20094       0.00029
                  16384        0.20134   0.20134  0.20092       0.00041
        20        32           0.34286   0.31172  0.31282      -0.00110
                  256          0.25168   0.23121  0.23212      -0.00092
                  1024         0.21609   0.20795  0.20859      -0.00064
                  4096         0.20022   0.19817  0.19797       0.00020
                  16384        0.20619   0.20639  0.20644      -0.00005
2000000 5         32           0.33979   0.33979  0.33423       0.00555
        10        32           0.33893   0.31382  0.30907       0.00474
                  256          0.23744   0.21472  0.21724      -0.00252
                  1024         0.20274   0.20274  0.19960       0.00314
        15        32           0.33893   0.30837  0.30984      -0.00147
                  256          0.23744   0.21310  0.21347      -0.00037
                  1024         0.19503   0.18647  0.18898      -0.00251
                  4096         0.17666   0.17378  0.17419      -0.00040
                  16384        0.17191   0.17191  0.17282      -0.00091
        20        32           0.33893   0.31194  0.31050       0.00144
                  256          0.23744   0.21204  0.20879       0.00325
                  1024         0.19324   0.18807  0.18549       0.00258
                  4096         0.17502   0.16870  0.16935      -0.00064
                  16384        0.17095   0.17057  0.17068      -0.00011

Done: 13129.097624540329 seconds

"""
import pandas as pd
import numpy as np
import time
time_begin = time.time()

from sklearn.datasets import make_classification

from utility import experiment_binary_gb
from data_path import data_path

params_xgb = {'objective'       : 'binary:logistic',
              'eval_metric'     : 'logloss',
              'tree_method'     : 'exact',
              'updater'         : 'grow_colmaker',
              'eta'             : 0.1, #default=0.3
              'lambda'          : 1, #default
              'min_child_weight': 1, #default
              'silent'          : True,
              'threads'         : 8}

params_xgb_eqbin_d = params_xgb.copy()
params_xgb_eqbin_d.update({'tree_method': 'hist',
                           'updater'    : 'grow_fast_histmaker',
                           'grow_policy': 'depthwise',
                           'max_bin'    : 255,  #default=256
                       })

params_xgb_eqbin_l = params_xgb_eqbin_d.copy()
params_xgb_eqbin_l.update({'grow_policy': 'lossguide'})

params_xgb_gpu = params_xgb.copy()
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

df_times = pd.DataFrame(times, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_times.set_index(['n_train', 'max_depth', 'num_leaves'], inplace=True)
for model_str in model_str_lst:
    df_times[model_str + '/LGB'] = df_times[model_str] / df_times['LGB']
df_times.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_times.columns]],
                                labels=[np.repeat([0, 1], [3, 2]), range(5)])
df_times.to_csv('log/' + fname_header + 'times.csv')

df_valid_scores = pd.DataFrame(valid_scores, columns=model_str_lst+['LGB']).join(pd.DataFrame(params))
df_valid_scores.set_index(['n_train', 'max_depth', 'num_leaves'], inplace=True)

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
