"""
2016/12/16 13.6h
exp name  : exp004
desciption: Comparison XGB:CPU and LightGBM on Higgs data
fname     : exp004.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
result    : AUC, Feature importance, Leaf counts, Time
params:
  n_rounds : 500
  n_train  : 10K, 0.1M, 1M, 10M
  max_depth: 5, 10, 15

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'auc', 'tree_method':'exact', 'threads':8,
              'max_depth':max_depth}

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : {'auc'}, 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0,
              'max_depth': max_depth, 'num_leaves' : 2**max_depth}

                   Time(sec)               Ratio
                     XGB_CPU    LGBM XGB_CPU/LGB
n_train  max_depth                              
10000    5              23.6    22.8         1.0
         10             25.3    26.6         1.0
         15             26.7    34.7         0.8
100000   5              42.6    27.2         1.6
         10             67.5    34.5         2.0
         15             96.8    87.9         1.1
1000000  5             328.4    85.1         3.9
         10            627.9   123.5         5.1
         15            949.7   436.5         2.2
10000000 5            6020.8   945.0         6.4
         10          13259.4  1318.3        10.1
         15          21068.2  2251.3         9.4

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
              'eval_metric':'auc', 'tree_method':'exact', 'threads':8,
              'silent':1}

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : {'auc'}, 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0}

params = []
times = []
n_valid = 500000
n_rounds = 500
fname_header = "exp004_"
for n_train in [10**4, 10**5, 10**6, 10**7]:
    for max_depth in [5, 10, 15]:
        fname_footer = "n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        params_xgb['max_depth'] = max_depth
        params_lgb['max_depth'] = max_depth
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print(params[-1])
        time_sec_lst = experiment_binary(X[:n_train], y[:n_train], X[-n_valid:], y[-n_valid:],
                                         params_xgb, params_lgb, n_rounds=n_rounds, use_gpu=False,
                                         fname_header=fname_header, fname_footer=fname_footer,
                                         n_skip=100)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=['XGB_CPU', 'LGBM']).join(pd.DataFrame(params))
df_time['XGB_CPU/LGB'] = df_time['XGB_CPU'] / df_time['LGBM']
df_time.set_index(['n_train', 'max_depth'], inplace=True)
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[[0,0,1,],[0,1,2]])
df_time.to_csv('log/' + fname_header + 'time.csv')

pd.set_option('display.precision', 1)
print('\n')
print(df_time)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
