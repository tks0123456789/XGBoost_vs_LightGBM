"""
2017/1/23 2.12h
exp name  : exp011
desciption: Comparison of XGBoost and LightGBM on arificial datasets
fname     : exp011.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS
preprocess: None
result    : Logloss, Feature importance, Leaf counts, Time
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
500000  5         32              12.4     11.9    5.2          2.4          2.3
        10        32              12.2     11.8    6.0          2.0          2.0
                  256             30.8     31.7   11.5          2.7          2.8
                  1024            47.1     46.9   14.3          3.3          3.3
        15        32              12.0     11.9    6.3          1.9          1.9
                  256             31.1     30.7   11.5          2.7          2.7
                  1024            79.0     80.8   22.0          3.6          3.7
                  4096           143.1    143.9   35.3          4.1          4.1
        20        32              12.1     11.6    6.1          2.0          1.9
                  256             31.5     30.4   11.8          2.7          2.6
                  1024            80.3     83.8   24.2          3.3          3.5
                  4096           205.7    210.3   52.8          3.9          4.0
1000000 5         32              22.6     22.2   10.8          2.1          2.0
        10        32              22.0     21.2   13.0          1.7          1.6
                  256             44.9     45.6   22.7          2.0          2.0
                  1024            72.2     71.0   27.2          2.6          2.6
        15        32              21.8     21.7   13.0          1.7          1.7
                  256             44.0     44.7   22.7          1.9          2.0
                  1024            99.4    102.5   37.5          2.6          2.7
                  4096           210.0    207.1   62.3          3.4          3.3
        20        32              21.7     21.8   13.1          1.7          1.7
                  256             45.0     47.9   23.3          1.9          2.1
                  1024           100.8    101.0   37.3          2.7          2.7
                  4096           261.9    285.9   88.7          3.0          3.2
2000000 5         32              41.1     40.0   22.5          1.8          1.8
        10        32              39.9     38.9   25.8          1.5          1.5
                  256             74.2     76.6   46.7          1.6          1.6
                  1024           112.4    112.7   57.8          1.9          2.0
        15        32              40.1     38.9   25.8          1.6          1.5
                  256             71.3     70.4   46.7          1.5          1.5
                  1024           131.9    135.6   65.6          2.0          2.1
                  4096           279.2    283.4  109.6          2.5          2.6
        20        32              40.4     37.6   25.7          1.6          1.5
                  256             71.6     72.0   47.7          1.5          1.5
                  1024           133.6    128.0   65.1          2.1          2.0
                  4096           318.7    344.3  129.2          2.5          2.7

Logloss
                              EQBIN_dw  EQBIN_lg     LGB  EQBIN_lg-LGB
n_train max_depth num_leaves                                          
500000  5         32            0.3630    0.3630  0.3623        0.0007
        10        32            0.3608    0.3299  0.3331       -0.0032
                  256           0.2673    0.2536  0.2551       -0.0015
                  1024          0.2484    0.2484  0.2431        0.0054
        15        32            0.3608    0.3280  0.3254        0.0026
                  256           0.2680    0.2546  0.2556       -0.0010
                  1024          0.2322    0.2272  0.2268        0.0004
                  4096          0.2226    0.2202  0.2216       -0.0014
        20        32            0.3608    0.3308  0.3330       -0.0022
                  256           0.2680    0.2518  0.2505        0.0013
                  1024          0.2313    0.2240  0.2230        0.0010
                  4096          0.2204    0.2188  0.2198       -0.0009
1000000 5         32            0.3427    0.3427  0.3431       -0.0004
        10        32            0.3429    0.3164  0.3149        0.0015
                  256           0.2497    0.2394  0.2396       -0.0001
                  1024          0.2276    0.2276  0.2291       -0.0015
        15        32            0.3429    0.3123  0.3139       -0.0017
                  256           0.2517    0.2330  0.2354       -0.0024
                  1024          0.2183    0.2101  0.2086        0.0015
                  4096          0.2032    0.2012  0.2009        0.0003
        20        32            0.3429    0.3117  0.3128       -0.0011
                  256           0.2517    0.2312  0.2321       -0.0009
                  1024          0.2161    0.2080  0.2086       -0.0006
                  4096          0.2002    0.1982  0.1980        0.0002
2000000 5         32            0.3398    0.3398  0.3342        0.0056
        10        32            0.3389    0.3138  0.3091        0.0047
                  256           0.2374    0.2147  0.2172       -0.0025
                  1024          0.2027    0.2027  0.1996        0.0031
        15        32            0.3389    0.3084  0.3098       -0.0015
                  256           0.2374    0.2131  0.2135       -0.0004
                  1024          0.1950    0.1865  0.1890       -0.0025
                  4096          0.1767    0.1738  0.1742       -0.0004
        20        32            0.3389    0.3119  0.3105        0.0014
                  256           0.2374    0.2120  0.2088        0.0033
                  1024          0.1932    0.1881  0.1855        0.0026
                  4096          0.1750    0.1687  0.1693       -0.0006

"""
import pandas as pd
import numpy as np
import time
time_begin = time.time()

from sklearn.datasets import make_classification

from utility import experiment_binary_gb
from data_path import data_path

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'logloss',
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
              'metric' : 'binary_logloss', 'sigmoid': 1.0, 'num_threads':8,
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
print('\nLogloss')
print(df_valid_scores)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
