#
import time
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import json

def experiment_binary_auc(X_train, y_train, X_valid, y_valid, n_rounds=10, max_depth=5,
                          log_header=None, n_skip=10):
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
    if log_header is not None:
        footer = "n_train_%d_n_rounds_%d_max_depth_%d.csv" % (n_train, n_rounds, max_depth)
        df_auc_train.to_csv('log/' header + 'Auc_Train_' + footer)
        df_auc_valid.to_csv('log/' header + 'Auc_Valid_' + footer)
        df_leaf_cnts.to_csv('log/' header + 'Leaf_cnts_' + footer)
        df_feat_imps.to_csv('log/' header + 'Feat_imps_' + footer)
    return(time_sec_lst)
