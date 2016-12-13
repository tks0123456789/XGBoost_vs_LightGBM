"""
2016/12/13 7.3h
exp name  : exp001
desciption: Comparison btw XGB:CPU, XGB:GPU, and LightGBM on Higgs data
fname     : xgb_vs_lgbm_001.py
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

Time
                    XGB_CPU  XGB_GPU   LGBM
n_train  max_depth                         
100000   5             55.0     37.7    5.4
         10            80.3     42.0    7.4
         15           111.6     47.7   32.5
1000000  5            274.8    115.8   17.4
         10           521.8    143.5   29.4
         15           830.5    171.2  107.6
10000000 5           2774.8    954.3  188.9
         10          5959.8   1209.1  296.1
         15          9809.8   1455.8  585.2

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
fname_header = "exp001_"
for n_train in [10**5, 10**6, 10**7]:
    for max_depth in [5, 10, 15]:
        fname_footer = "n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        params_xgb['max_depth'] = max_depth
        params_lgb['max_depth'] = max_depth + 1
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print(params[-1])
        time_sec_lst = experiment_binary(X[:n_train], y[:n_train], X[-n_valid:], y[-n_valid:],
                                         params_xgb, params_lgb, n_rounds=n_rounds,
                                         fname_header=fname_header, fname_footer=fname_footer,
                                         n_skip=15)
        times.append(time_sec_lst)

pd.set_option('display.precision', 1)
print("\n\nTime")
print(pd.DataFrame(times, columns=['XGB_CPU', 'XGB_GPU', 'LGBM']).join(pd.DataFrame(params)).set_index(['n_train', 'max_depth']))

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
