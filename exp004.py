"""
2016/12/15 2.8h
exp name  : exp004
desciption: Comparison XGB:CPU and LightGBM on Higgs data
fname     : exp004.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
result    : AUC, Feature importance, Leaf counts, Time
params:
  n_rounds : 100
  n_train  : 10**5, 10**6, 10**7
  max_depth: 5, 10, 15

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'auc', 'tree_method':'exact', 'threads':8,
              'max_depth':max_depth}

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : {'auc'}, 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0,
              'max_depth': max_depth+1, 'num_leaves' : 2**max_depth}

Time(sec)
                    XGB_CPU   LGBM
n_train  max_depth                
100000   5              8.7    5.5
         10            14.2    7.4
         15            20.9   32.4
1000000  5             68.5   17.3
         10           143.8   29.2
         15           218.4  111.4
10000000 5           1207.0  188.9
         10          2658.8  295.6
         15          4274.5  582.4

Ratio
                    XGB_CPU  LGBM
n_train  max_depth               
100000   5             1.56     1
         10            1.93     1
         15            0.64     1
1000000  5             3.95     1
         10            4.93     1
         15            1.96     1
10000000 5             6.39     1
         10            8.99     1
         15            7.34     1

Done: 10107.3541229 seconds

"""
import pandas as pd
import time
time_begin = time.time()

from utility import experiment_binary
from data_path import data_path

# https://archive.ics.uci.edu/ml/datasets/HIGGS
dtrain = pd.read_csv(data_path+'HIGGS.csv', header=None).values
print ('finish loading from csv ')

X = dtrain[:,1:]
y = dtrain[:,0]

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'auc', 'tree_method':'exact', 'threads':8}

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : {'auc'}, 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0}

params = []
times = []
n_valid = 500000
n_rounds = 100
fname_header = "exp004_"
for n_train in [10**5, 10**6, 10**7]:
    for max_depth in [5, 10, 15]:
        fname_footer = "n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        params_xgb['max_depth'] = max_depth
        params_lgb['max_depth'] = max_depth + 1
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print(params[-1])
        time_sec_lst = experiment_binary(X[:n_train], y[:n_train], X[-n_valid:], y[-n_valid:],
                                         params_xgb, params_lgb, n_rounds=n_rounds, use_gpu=False,
                                         fname_header=fname_header, fname_footer=fname_footer,
                                         n_skip=15)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=['XGB_CPU', 'LGBM']).join(pd.DataFrame(params))
df_time.to_csv("log/" + fname_header + "time.csv")

pd.set_option('display.precision', 1)
print("\n\nTime(sec)")
print(df_time.set_index(['n_train', 'max_depth']))

print('\nRatio')
df_time['XGB_CPU'] /= df_time['LGBM']
df_time['LGBM'] = 1
pd.set_option('display.precision', 2)
print(df_time)
    
print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
