"""
2017/1/23 2.1h
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
500000  5         32              12.3     11.8    5.2          2.4          2.3
        10        32              12.4     11.9    6.1          2.0          1.9
                  256             31.4     31.7   11.4          2.8          2.8
                  1024            47.6     46.9   13.9          3.4          3.4
        15        32              12.3     12.0    6.3          2.0          1.9
                  256             31.7     30.8   11.5          2.8          2.7
                  1024            79.1     81.3   22.4          3.5          3.6
                  4096           144.8    142.3   35.5          4.1          4.0
        20        32              12.4     11.5    6.0          2.1          1.9
                  256             31.9     30.4   11.4          2.8          2.7
                  1024            80.1     82.9   22.7          3.5          3.7
                  4096           206.0    210.1   51.7          4.0          4.1
1000000 5         32              22.5     21.9   10.8          2.1          2.0
        10        32              21.6     21.8   12.8          1.7          1.7
                  256             44.7     47.3   22.3          2.0          2.1
                  1024            72.6     70.2   27.2          2.7          2.6
        15        32              22.3     21.6   13.0          1.7          1.7
                  256             46.1     44.4   22.5          2.0          2.0
                  1024            99.8    102.2   37.3          2.7          2.7
                  4096           210.1    208.3   62.8          3.3          3.3
        20        32              21.5     21.5   13.1          1.6          1.6
                  256             45.7     44.2   23.5          1.9          1.9
                  1024            98.8    100.6   37.0          2.7          2.7
                  4096           259.8    284.9   89.2          2.9          3.2
2000000 5         32              40.5     40.8   22.4          1.8          1.8
        10        32              40.5     39.5   25.7          1.6          1.5
                  256             71.0     78.4   46.4          1.5          1.7
                  1024           111.3    108.2   57.0          2.0          1.9
        15        32              39.2     39.2   25.6          1.5          1.5
                  256             71.0     70.7   46.6          1.5          1.5
                  1024           132.5    134.5   65.8          2.0          2.0
                  4096           275.0    283.7  109.7          2.5          2.6
        20        32              41.2     38.1   25.6          1.6          1.5
                  256             73.2     73.8   47.5          1.5          1.6
                  1024           136.8    128.5   64.8          2.1          2.0
                  4096           316.7    333.8  129.6          2.4          2.6

Logloss
                              EQBIN_dw  EQBIN_lg     LGB  EQBIN_lg-LGB
n_train max_depth num_leaves                                          
500000  5         32            0.3630    0.3630  0.3604    2.5272e-03
        10        32            0.3608    0.3299  0.3331   -3.2004e-03
                  256           0.2673    0.2536  0.2780   -2.4332e-02
                  1024          0.2484    0.2484  0.2745   -2.6060e-02
        15        32            0.3608    0.3280  0.3281   -9.5740e-05
                  256           0.2680    0.2546  0.2771   -2.2507e-02
                  1024          0.2322    0.2272  0.2722   -4.4943e-02
                  4096          0.2226    0.2202  0.2896   -6.9379e-02
        20        32            0.3608    0.3308  0.3322   -1.3850e-03
                  256           0.2680    0.2518  0.2733   -2.1478e-02
                  1024          0.2313    0.2240  0.2709   -4.6852e-02
                  4096          0.2204    0.2188  0.3243   -1.0551e-01
1000000 5         32            0.3427    0.3427  0.3517   -8.9887e-03
        10        32            0.3429    0.3164  0.3249   -8.4835e-03
                  256           0.2497    0.2394  0.2624   -2.2939e-02
                  1024          0.2276    0.2276  0.2552   -2.7667e-02
        15        32            0.3429    0.3123  0.3229   -1.0639e-02
                  256           0.2517    0.2330  0.2581   -2.5036e-02
                  1024          0.2183    0.2101  0.2456   -3.5524e-02
                  4096          0.2032    0.2012  0.2597   -5.8515e-02
        20        32            0.3429    0.3117  0.3218   -1.0052e-02
                  256           0.2517    0.2312  0.2554   -2.4167e-02
                  1024          0.2161    0.2080  0.2470   -3.9025e-02
                  4096          0.2002    0.1982  0.2844   -8.6268e-02
2000000 5         32            0.3398    0.3398  0.3336    6.1835e-03
        10        32            0.3389    0.3138  0.3086    5.2613e-03
                  256           0.2374    0.2147  0.2340   -1.9233e-02
                  1024          0.2027    0.2027  0.2223   -1.9586e-02
        15        32            0.3389    0.3084  0.3083    9.4393e-05
                  256           0.2374    0.2131  0.2293   -1.6185e-02
                  1024          0.1950    0.1865  0.2138   -2.7289e-02
                  4096          0.1767    0.1738  0.2145   -4.0704e-02
        20        32            0.3389    0.3119  0.3070    4.9439e-03
                  256           0.2374    0.2120  0.2251   -1.3069e-02
                  1024          0.1932    0.1881  0.2117   -2.3654e-02
                  4096          0.1750    0.1687  0.2200   -5.1285e-02

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
