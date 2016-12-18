"""
2016/12/18 12.4h
exp name  : exp008
desciption: Comparison XGB:CPU and LightGBM on Higgs data
          : Training only
fname     : exp008.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
result    : Feature importance, Leaf counts, Time
params:
  n_rounds : 500
  n_train  : 10K, 0.1M, 1M, 10M
  max_depth: 5, 10, 15

                   Time(sec)           Ratio
                         XGB     LGB XGB/LGB
n_train  max_depth                          
10000    5               2.0     0.3     6.2
         10              4.1     1.6     2.6
         15              5.3     8.5     0.6
100000   5              19.1     1.5    13.0
         10             42.4     6.8     6.2
         15             71.0    57.3     1.2
1000000  5             277.2    18.3    15.1
         10            634.3    53.5    11.9
         15            997.8   370.3     2.7
10000000 5            5543.9   199.3    27.8
         10          12795.7   561.9    22.8
         15          20575.2  1488.6    13.8

"""
import pandas as pd
import time
time_begin = time.time()

from utility import experiment_binary_train
from data_path import data_path

# https://archive.ics.uci.edu/ml/datasets/HIGGS
dtrain = pd.read_csv(data_path+'HIGGS.csv', header=None).values
print ('finish loading from csv ')

X = dtrain[:,1:]
y = dtrain[:,0]

params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'lambda':1,
              'eval_metric':'logloss', 'tree_method':'exact', 'threads':8,
              'silent':1}

params_lgb = {'task':'train', 'objective':'binary', 'learning_rate':0.1, 'lambda_l2':1,
              'metric' : {'binary_logloss'}, 'sigmoid': 0.5, 'num_threads':8,
              'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
              'verbose' : 0}

params = []
times = []
n_rounds = 500
fname_header = "exp008_"
for n_train in [10**4, 10**5, 10**6, 10**7]:
    for max_depth in [5, 10, 15]:
        fname_footer = "n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        params_xgb['max_depth'] = max_depth
        params_lgb['max_depth'] = max_depth
        params_lgb['num_leaves'] = 2 ** max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print(params[-1])
        time_sec_lst = experiment_binary_train(X[:n_train], y[:n_train],
                                               params_xgb, params_lgb, n_rounds=n_rounds,
                                               fname_header=fname_header,
                                               fname_footer=fname_footer,
                                               n_skip=100)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=['XGB', 'LGB']).join(pd.DataFrame(params))
df_time['XGB/LGB'] = df_time['XGB'] / df_time['LGB']
df_time.set_index(['n_train', 'max_depth'], inplace=True)
df_time.columns = pd.MultiIndex(levels=[['Time(sec)', 'Ratio'],[df_time.columns]],
                                labels=[[0,0,1,],[0,1,2]])
df_time.to_csv('log/' + fname_header + 'time.csv')

pd.set_option('display.precision', 1)
print('\n')
print(df_time)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
