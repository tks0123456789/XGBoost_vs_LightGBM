"""
2016/12/15 1.4h
exp name  : exp006
desciption: Comparison of XGB:CPU and LightGBM on arificial datasets
fname     : exp006.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.4LTS
result    : Logloss, Feature importance, Leaf counts, Time
params:
  n_train             : 1M, 2M, 4M
  n_features          : 28
  n_rounds            : 100
  n_clusters_per_class: 64
  max_depth-1         : no limit, [11,13,15], 20
  num_leaves          : 2**11, 2**13, 2**15

                              no_depth_limit  depth  depth_8  depth_18
n_train num_leaves max_depth                                          
1000000 2048       12                   29.5   17.1      8.9      27.0
        8192       14                   80.1   30.6      9.4      61.6
        32768      16                  382.1   82.1     11.7     154.6
2000000 2048       12                   50.2   30.2     16.1      48.1
        8192       14                  114.1   52.4     16.9      98.3
        32768      16                  490.0  127.2     19.0     234.6
4000000 2048       12                   96.4   60.4     33.7      90.5
        8192       14                  176.5  102.3     34.7     163.2
        32768      16                  592.5  217.7     37.4     368.4

"""
import pandas as pd
import numpy as np
import time
time_begin = time.time()

from sklearn.datasets import make_classification, make_blobs, make_gaussian_quantiles

from utility import exp_binary_lgbm
from data_path import data_path

no_depth_limit ={'task':'train', 'objective':'binary',
                 'learning_rate':0.1, 'lambda_l2':1,
                 'metric' : {'binary_logloss'},
                 'sigmoid': 0.5, 'num_threads':8,
                 'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
                 'verbose' : 0}
depth = no_depth_limit.copy()
depth_limit_8 = no_depth_limit.copy()
depth_limit_8['max_depth'] = 9
depth_limit_18 = no_depth_limit.copy()
depth_limit_18['max_depth'] = 19

params_lgb_lst = [no_depth_limit, depth, depth_limit_8, depth_limit_18]

params = []
times = []
n_classes = 2
n_valid = 5*10**5
n_clusters_per_class = 64
n_rounds = 100
fname_header = "exp006_"
model_str_lst = ['no_depth_limit', 'depth', 'depth_8', 'depth_18']
for n_train in [10**6, 2*10**6, 4*10**6]:
    n_all = n_train + n_valid
    X, y = make_classification(n_samples=n_all, n_classes=n_classes, n_features=28,
                               n_informative=10, n_redundant=10,
                               n_clusters_per_class=n_clusters_per_class,
                               shuffle=True, random_state=123*6)
    for num_leaves in [2**11, 2**13, 2**15]:
        fname_footer = "n_train_%d_num_leaves_%d.csv" % (n_train, num_leaves)
        max_depth = int(np.log2(num_leaves)) + 1
        params_lgb_lst[0]['num_leaves'] = num_leaves
        params_lgb_lst[1]['num_leaves'] = num_leaves
        params_lgb_lst[1]['max_depth'] = max_depth # tree depth limit is max_depth - 1
        params_lgb_lst[2]['num_leaves'] = num_leaves
        params_lgb_lst[3]['num_leaves'] = num_leaves
        params.append({'n_train':n_train, 'num_leaves':num_leaves, 'max_depth':max_depth})
        print('\n')
        print(params[-1])
        time_sec_lst = exp_binary_lgbm(X[:n_train], y[:n_train], X[-n_valid:], y[-n_valid:],
                                       params_lgb_lst, model_str_lst=model_str_lst,
                                       metric='logloss', n_rounds=n_rounds,
                                       fname_header=fname_header, fname_footer=fname_footer,
                                       n_skip=15)
        times.append(time_sec_lst)

df_time = pd.DataFrame(times, columns=model_str_lst).join(pd.DataFrame(params))
df_time.set_index(['n_train', 'num_leaves', 'max_depth'], inplace=True)
df_time.to_csv('log/' + fname_header + 'time.csv')

pd.set_option('display.precision', 1)
print(df_time)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
