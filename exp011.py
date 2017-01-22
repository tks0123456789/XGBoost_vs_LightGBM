"""
2017/1/23 
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
500000  5         32              12.5     11.9    4.9          2.5          2.4
        10        32              12.2     11.8    5.9          2.1          2.0
                  256             31.4     31.7   11.1          2.8          2.8
                  1024            47.5     46.2   13.5          3.5          3.4
        15        32              11.9     11.7    5.9          2.0          2.0
                  256             31.5     30.8   11.2          2.8          2.7
                  1024            78.3     81.9   21.5          3.6          3.8
                  4096           143.9    144.3   36.1          4.0          4.0
        20        32              12.3     11.8    5.9          2.1          2.0
                  256             31.6     30.2   11.4          2.8          2.6
                  1024            80.1     82.7   22.7          3.5          3.6
                  4096           205.4    210.3   53.0          3.9          4.0
1000000 5         32              22.4     21.7   10.2          2.2          2.1
        10        32              22.9     21.5   12.4          1.9          1.7
                  256             46.9     46.9   22.0          2.1          2.1
                  1024            73.5     69.5   26.7          2.8          2.6
        15        32              21.7     21.6   12.4          1.8          1.7
                  256             45.6     43.8   22.6          2.0          1.9
                  1024            97.6    103.7   36.3          2.7          2.9
                  4096           209.8    212.1   63.4          3.3          3.3
        20        32              22.4     21.3   12.5          1.8          1.7
                  256             44.6     43.7   22.3          2.0          2.0
                  1024            99.9     99.8   36.5          2.7          2.7
                  4096           262.6    284.9   87.9          3.0          3.2
2000000 5         32              40.4     40.6   20.8          1.9          1.9
        10        32              40.5     38.7   24.3          1.7          1.6
                  256             70.6     76.2   45.9          1.5          1.7
                  1024           108.9    110.3   56.2          1.9          2.0
        15        32              39.2     38.6   25.1          1.6          1.5
                  256             71.1     70.3   46.2          1.5          1.5
                  1024           133.8    134.3   66.8          2.0          2.0
                  4096           280.9    283.0  108.0          2.6          2.6
        20        32              40.7     38.5   23.7          1.7          1.6
                  256             70.9     70.1   45.9          1.5          1.5
                  1024           134.7    128.6   64.1          2.1          2.0
                  4096           321.0    336.8  130.1          2.5          2.6


                              EQBIN_dw  EQBIN_lg     LGB
n_train max_depth num_leaves                            
500000  5         32            0.3630    0.3630  0.3621
        10        32            0.3608    0.3299  0.3313
                  256           0.2673    0.2536  0.2772
                  1024          0.2484    0.2484  0.2753
        15        32            0.3608    0.3280  0.3311
                  256           0.2680    0.2546  0.2762
                  1024          0.2322    0.2272  0.2724
                  4096          0.2226    0.2202  0.2899
        20        32            0.3608    0.3308  0.3310
                  256           0.2680    0.2518  0.2742
                  1024          0.2313    0.2240  0.2722
                  4096          0.2204    0.2188  0.3258
1000000 5         32            0.3427    0.3427  0.3520
        10        32            0.3429    0.3164  0.3216
                  256           0.2497    0.2394  0.2616
                  1024          0.2276    0.2276  0.2532
        15        32            0.3429    0.3123  0.3213
                  256           0.2517    0.2330  0.2542
                  1024          0.2183    0.2101  0.2465
                  4096          0.2032    0.2012  0.2580
        20        32            0.3429    0.3117  0.3213
                  256           0.2517    0.2312  0.2550
                  1024          0.2161    0.2080  0.2471
                  4096          0.2002    0.1982  0.2846
2000000 5         32            0.3398    0.3398  0.3367
        10        32            0.3389    0.3138  0.3081
                  256           0.2374    0.2147  0.2348
                  1024          0.2027    0.2027  0.2197
        15        32            0.3389    0.3084  0.3054
                  256           0.2374    0.2131  0.2264
                  1024          0.1950    0.1865  0.2102
                  4096          0.1767    0.1738  0.2143
        20        32            0.3389    0.3119  0.3129
                  256           0.2374    0.2120  0.2259
                  1024          0.1932    0.1881  0.2111
                  4096          0.1750    0.1687  0.2207


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
