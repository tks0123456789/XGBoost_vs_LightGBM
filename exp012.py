"""
2017/1/23 2.8h
exp name  : exp012
desciption: The same as exp011 except the metric is AUC
fname     : exp012.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS
preprocess: None
result    : AUC, Feature importance, Leaf counts, Time
params:
  n_train             : 0.5M, 1M, 2M
  n_valid             : n_train/4
  n_features          : 32
  n_rounds            : 200
  n_clusters_per_class: 8
  max_depth           : 5, 10, 15, 20
  num_leaves          : 32, 256, 1024, 4096

                             Time(sec)                        Ratio             
                              EQBIN_dw EQBIN_lg    LGB EQBIN_dw/LGB EQBIN_lg/LGB
n_train max_depth num_leaves                                                    
500000  5         32              19.8     19.2   14.3          1.4          1.3
        10        32              19.9     19.5   15.1          1.3          1.3
                  256             39.2     39.2   20.3          1.9          1.9
                  1024            55.2     54.3   23.0          2.4          2.4
        15        32              19.9     19.5   15.3          1.3          1.3
                  256             39.4     38.5   20.1          2.0          1.9
                  1024            85.6     87.5   30.1          2.8          2.9
                  4096           151.1    150.0   44.1          3.4          3.4
        20        32              19.8     19.3   14.5          1.4          1.3
                  256             39.3     37.8   19.8          2.0          1.9
                  1024            87.6     91.0   30.8          2.8          3.0
                  4096           211.0    216.5   62.7          3.4          3.5
1000000 5         32              37.7     37.6   30.2          1.2          1.2
        10        32              37.6     37.4   32.7          1.1          1.1
                  256             61.0     62.4   42.9          1.4          1.5
                  1024            89.6     87.4   50.2          1.8          1.7
        15        32              37.9     37.5   32.6          1.2          1.1
                  256             61.4     61.3   46.4          1.3          1.3
                  1024           116.3    118.6   60.6          1.9          2.0
                  4096           227.5    226.5   88.9          2.6          2.5
        20        32              37.7     37.1   35.4          1.1          1.0
                  256             62.1     60.6   43.7          1.4          1.4
                  1024           116.0    115.5   57.1          2.0          2.0
                  4096           276.1    301.2  107.9          2.6          2.8
2000000 5         32              75.4     74.2   71.5          1.1          1.0
        10        32              74.2     72.8   74.8          1.0          1.0
                  256            107.7    110.4   96.1          1.1          1.1
                  1024           146.1    144.1  106.5          1.4          1.4
        15        32              74.7     73.2   74.8          1.0          1.0
                  256            107.4    104.8   96.1          1.1          1.1
                  1024           171.0    168.4  115.4          1.5          1.5
                  4096           313.8    317.2  159.8          2.0          2.0
        20        32              74.2     72.7   74.4          1.0          1.0
                  256            107.6    104.1   97.1          1.1          1.1
                  1024           170.8    162.9  114.7          1.5          1.4
                  4096           353.9    378.8  178.9          2.0          2.1

AUC
                              EQBIN_dw  EQBIN_lg     LGB  EQBIN_lg-LGB
n_train max_depth num_leaves                                          
500000  5         32            0.9263    0.9263  0.9262    1.5158e-04
        10        32            0.9274    0.9395  0.9382    1.2360e-03
                  256           0.9587    0.9626  0.9622    4.2344e-04
                  1024          0.9638    0.9638  0.9644   -5.3343e-04
        15        32            0.9274    0.9398  0.9408   -1.0146e-03
                  256           0.9585    0.9625  0.9622    3.1177e-04
                  1024          0.9676    0.9687  0.9688   -1.2582e-04
                  4096          0.9697    0.9704  0.9702    1.5840e-04
        20        32            0.9274    0.9390  0.9384    5.8636e-04
                  256           0.9585    0.9632  0.9635   -3.2049e-04
                  1024          0.9679    0.9694  0.9697   -2.6984e-04
                  4096          0.9713    0.9726  0.9723    3.5725e-04
1000000 5         32            0.9330    0.9330  0.9327    2.9837e-04
        10        32            0.9331    0.9435  0.9438   -2.4784e-04
                  256           0.9640    0.9667  0.9666    9.0437e-05
                  1024          0.9695    0.9695  0.9694    1.1518e-04
        15        32            0.9331    0.9449  0.9443    5.8412e-04
                  256           0.9635    0.9683  0.9677    5.7254e-04
                  1024          0.9715    0.9733  0.9736   -3.3313e-04
                  4096          0.9745    0.9749  0.9749    5.9958e-06
        20        32            0.9331    0.9450  0.9447    2.6357e-04
                  256           0.9635    0.9688  0.9686    2.4953e-04
                  1024          0.9720    0.9737  0.9735    1.3542e-04
                  4096          0.9752    0.9763  0.9764   -7.9870e-05
2000000 5         32            0.9357    0.9357  0.9378   -2.1197e-03
        10        32            0.9362    0.9464  0.9475   -1.1332e-03
                  256           0.9682    0.9734  0.9729    5.3319e-04
                  1024          0.9760    0.9760  0.9762   -2.5484e-04
        15        32            0.9362    0.9478  0.9474    3.4074e-04
                  256           0.9682    0.9739  0.9738    5.5413e-05
                  1024          0.9774    0.9790  0.9785    4.3423e-04
                  4096          0.9805    0.9810  0.9809    8.2714e-05
        20        32            0.9362    0.9471  0.9475   -4.2095e-04
                  256           0.9682    0.9741  0.9748   -7.1769e-04
                  1024          0.9778    0.9787  0.9791   -4.7622e-04
                  4096          0.9808    0.9818  0.9817    1.1680e-04

"""
import pandas as pd
import numpy as np
import time
time_begin = time.time()

from sklearn.datasets import make_classification

from utility import experiment_binary_gb
from data_path import data_path

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'auc',
              'silent':True, 'threads':8}

params_xgb_eqbin_d = params_xgb.copy()
params_xgb_eqbin_d.update({'tree_method':'hist',
                           'grow_policy':'depthwise',
                           'max_bin':255})

params_xgb_eqbin_l = params_xgb_eqbin_d.copy()
params_xgb_eqbin_l.update({'grow_policy':'lossguide'})

params_xgb_lst = [params_xgb_eqbin_d, params_xgb_eqbin_l]
model_str_lst = ['EQBIN_dw', 'EQBIN_lg']

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : 'auc', 'sigmoid': 1.0, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0}

params = []
times = []
valid_scores = []
n_classes = 2
n_clusters_per_class = 8
n_features = 32
n_informative = n_redundant = n_features // 4
n_rounds = 200
fname_header = "exp012_"

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

pd.set_option('display.precision', 4)
pd.set_option('display.width', 100)
print('\nAUC')
print(df_valid_scores)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
