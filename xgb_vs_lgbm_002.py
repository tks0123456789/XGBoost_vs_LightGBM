"""
Comparison btw XGB:CPU, XGB:GPU, and LightGBM
xgb_vs_lgbm_002.py
2016/12/13
params:
  n_rounds: 100
    params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'max_depth':max_depth, 'lambda':1,
                  'eval_metric':'auc', 'tree_method':'exact', 'threads':8}
    params_lgb = {'task': 'train', 'objective': 'binary', 'max_depth': max_depth+1, 'lambda_l2':1,
                  'learning_rate' : 0.1, 'sigmoid': 0.5, 'num_leaves' : 2**max_depth,
                  'metric' : {'auc'},
                  'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
                  'num_threads':8, 'verbose' : 0}

Time
                    XGB_CPU  XGB_GPU   LGBM
n_train  max_depth                         
100000   5             54.4     38.0    5.4
         10            80.7     42.2    7.4
         15           111.6     47.7   33.0
1000000  5            272.8    115.0   17.3
         10           535.2    143.9   29.1
         15           839.9    171.7  108.9
10000000 5           2794.2    954.3  189.1
         10          5944.8   1210.7  295.8
         15          9841.6   1455.9  584.1

"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import json

from data_path import data_path
import time

time_begin = time.time()
np.random.seed(123)

# binary classification
# #positive==#negative
def make_gaussian_data(n_row, n_col, class_sep=1):
    class_diff = np.random.rand(n_col)
    class_diff /= np.sqrt((class_diff ** 2).sum())

    mean_positive = np.repeat(0, n_col)
    mean_negative = mean_positive + class_sep * class_diff
    
    X = np.vstack((np.random.multivariate_normal(mean=mean_positive,
                                                 cov=np.identity(n_col), size=n_row/2),
                   np.random.multivariate_normal(mean=mean_negative,
                                                 cov=np.identity(n_col), size=n_row/2)))
    y = np.repeat((1,0), n_row/2)
    idx = np.arange(n_row)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y

def experiment(X_train, y_train, X_valid, y_valid, n_rounds=10, max_depth=5,
               log_footer=None, n_skip=10):
    t000 = time.time()
    print("\nmax_depth:%d\n" % max_depth)
    
    df_auc_train = pd.DataFrame(index=range(n_rounds))
    df_auc_valid = pd.DataFrame(index=range(n_rounds))
    feature_names = ['f%d' % i for i in range(X_train.shape[1])]
    feat_imps_dict = {}
    leaf_cnts_dict = {}
    time_sec_lst = []

    # XGBoost
    xgmat_train = xgb.DMatrix(X_train, label=y_train)
    watchlist = [(xgmat_train,'train'), (xgmat_valid, 'valid')]
    xgmat_valid = xgb.DMatrix(X_valid, label=y_valid)
    
    params_xgb = {'objective':'binary:logistic', 'eta':0.1, 'max_depth':max_depth, 'lambda':1,
                  'eval_metric':'auc', 'tree_method':'exact', 'threads':8}

    # XGB:CPU
    evals_result = {}
    print ("training xgboost - cpu tree construction")
    params_xgb['updater'] = 'grow_colmaker'
    t0 = time.time()
    bst = xgb.train(params_xgb, xgmat_train, n_rounds, watchlist,
                    evals_result=evals_result, verbose_eval=False)
    time_sec_lst.append(time.time() - t0)
    print ("XGBoost CPU: %s seconds" % (str(time_sec_lst[-1])))
    df_auc_train['XGB_CPU'] = evals_result['train']['auc']
    df_auc_valid['XGB_CPU'] = evals_result['valid']['auc']
    feat_imps_dict['XGB_CPU'] = pd.Series(bst.get_score(importance_type='gain'), index=feature_names)
    dmp = bst.get_dump()
    leaf_cnts_dict['XGB_CPU'] = [tree.count('leaf') for tree in dmp]

    # XGB:GPU
    evals_result = {}
    print ("training xgboost - gpu tree construction")
    params_xgb['updater'] = 'grow_gpu'
    t0 = time.time()
    bst = xgb.train(params_xgb, xgmat_train, n_rounds, watchlist,
                    evals_result=evals_result, verbose_eval=False)
    time_sec_lst.append(time.time() - t0)
    print ("XGBoost GPU: %s seconds" % (str(time_sec_lst[-1])))
    df_auc_train['XGB_GPU'] = evals_result['train']['auc']
    df_auc_valid['XGB_GPU'] = evals_result['valid']['auc']
    feat_imps_dict['XGB_GPU'] = pd.Series(bst.get_score(importance_type='gain'), index=feature_names)
    dmp = bst.get_dump()
    leaf_cnts_dict['XGB_GPU'] = [tree.count('leaf') for tree in dmp]

    # LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    watchlist = [lgb_train, lgb_valid]
    print ("training LightGBM")
    params_lgb = {'task': 'train', 'objective': 'binary', 'max_depth': max_depth+1, 'lambda_l2':1,
                  'learning_rate' : 0.1, 'sigmoid': 0.5, 'num_leaves' : 2**(max_depth+1),
                  'metric' : {'auc'},
                  'min_data_in_leaf': 1, 'min_sum_hessian_in_leaf': 1,
                  'num_threads':8, 'verbose' : 0}
    evals_result = {}
    t0 = time.time()
    gbm = lgb.train(params_lgb, lgb_train, num_boost_round=n_rounds,
                    valid_sets=watchlist, evals_result=evals_result, verbose_eval=False)
    time_sec_lst.append(time.time() - t0)
    print ("LightGBM: %s seconds" % (str(time_sec_lst[-1])))
    df_auc_train['LGBM'] = evals_result['training']['auc']
    df_auc_valid['LGBM'] = evals_result['valid_1']['auc']
    feat_imps_dict['LGBM'] = gbm.feature_importance("gain")
    model_json = gbm.dump_model()
    tree_lst = [str(tree['tree_structure']) for tree in model_json['tree_info']]
    leaf_cnts_dict['LGBM'] = [tree.count('leaf_value') for tree in tree_lst]

    print('\nAUC train')
    print(df_auc_train.iloc[::n_skip,])
    if X_valid is not None:
        print('\nAUC valid')
        print(df_auc_valid.iloc[::n_skip,])

    print('\nLeaf counts')
    df_leaf_cnts = pd.DataFrame(leaf_cnts_dict, columns=['XGB_CPU', 'XGB_GPU', 'LGBM'])
    print(df_leaf_cnts.iloc[::n_skip,])

    df_feat_imps = pd.DataFrame(feat_imps_dict,
                                index=feature_names,
                                columns=['XGB_CPU', 'XGB_GPU', 'LGBM']).fillna(0)
    df_feat_imps /= df_feat_imps.sum(0)
    df_feat_imps = df_feat_imps.sort_values('XGB_CPU', ascending=False)
    print('\nFeature importance')
    print(df_feat_imps.head(5))
    if log_footer is not None:
        footer = log_footer + "n_train_%d_n_rounds_%d_max_depth_%d.csv" \
                 % (n_train, n_rounds, max_depth)
        df_auc_train.to_csv('log/' + 'Auc_Train_' + footer)
        df_auc_valid.to_csv('log/' + 'Auc_Valid_' + footer)
        df_leaf_cnts.to_csv('log/' + 'Leaf_cnts_' + footer)
        df_feat_imps.to_csv('log/' + 'Feat_imps_' + footer)
    return(time_sec_lst)

params = []
times = []
n_rounds = 30
for n_train in [2*10**5, 4*10**5, 8*10**5]:#, 5*10**5, 10**6, 5*10**6]:
    for n_col in [10]:
        for class_sep in [1]:
            print("\nn_train:%d, n_col:%d, class_sep:%d" % (n_train, n_col, class_sep))
            n_valid = int(0.1 * n_train)
            X, y = make_gaussian_data(n_row=n_train + n_valid,
                                      n_col=n_col,
                                      class_sep=class_sep)
            for max_depth in range(10, 16):
                params.append({'n_train':n_train,
                               'n_col':n_col,
                               'class_sep':class_sep,
                               'max_depth':max_depth})
                time_sec_lst = experiment(X[:n_train], y[:n_train], X[n_train:], y[n_train:],
                                          n_rounds, max_depth, n_skip=5, log_footer=None)
                times.append(time_sec_lst)

pd.set_option('display.precision', 1)
print("\n\nTime")
print(pd.DataFrame(times, columns=['XGB_CPU', 'XGB_GPU', 'LGBM']).join(pd.DataFrame(params)).set_index(['n_train', 'n_col', 'class_sep', 'max_depth']))
pd.set_option('display.precision', 4)

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
