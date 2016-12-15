#
import time
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import json

def experiment_binary(X_train, y_train, X_valid, y_valid,
                      params_xgb, params_lgb, n_rounds=10, use_gpu=True,
                      fname_header=None, fname_footer=None, n_skip=10):
    t000 = time.time()

    metric = params_xgb['eval_metric']

    df_score_train = pd.DataFrame(index=range(n_rounds))
    df_score_valid = pd.DataFrame(index=range(n_rounds))
    feature_names = ['f%d' % i for i in range(X_train.shape[1])]
    feat_imps_dict = {}
    leaf_cnts_dict = {}
    time_sec_lst = []

    # XGBoost
    xgmat_train = xgb.DMatrix(X_train, label=y_train)
    xgmat_valid = xgb.DMatrix(X_valid, label=y_valid)
    watchlist = [(xgmat_train,'train'), (xgmat_valid, 'valid')]

    # XGB:CPU
    params_xgb['updater'] = 'grow_colmaker'
    print("training xgboost - cpu tree construction")
    evals_result = {}
    t0 = time.time()
    bst = xgb.train(params_xgb, xgmat_train, n_rounds, watchlist,
                    evals_result=evals_result, verbose_eval=False)
    time_sec_lst.append(time.time() - t0)
    print("XGBoost CPU: %s seconds" % (str(time_sec_lst[-1])))
    df_score_train['XGB_CPU'] = evals_result['train'][metric]
    df_score_valid['XGB_CPU'] = evals_result['valid'][metric]
    feat_imps_dict['XGB_CPU'] = pd.Series(bst.get_score(importance_type='gain'), index=feature_names)
    dmp = bst.get_dump()
    leaf_cnts_dict['XGB_CPU'] = [tree.count('leaf') for tree in dmp]
    
    if use_gpu:
        # XGB:GPU
        params_xgb['updater'] = 'grow_gpu'
        print("training xgboost - gpu tree construction")
        evals_result = {}
        t0 = time.time()
        bst = xgb.train(params_xgb, xgmat_train, n_rounds, watchlist,
                        evals_result=evals_result, verbose_eval=False)
        time_sec_lst.append(time.time() - t0)
        print("XGBoost GPU: %s seconds" % (str(time_sec_lst[-1])))
        df_score_train['XGB_GPU'] = evals_result['train'][metric]
        df_score_valid['XGB_GPU'] = evals_result['valid'][metric]
        feat_imps_dict['XGB_GPU'] = pd.Series(bst.get_score(importance_type='gain'), index=feature_names)
        dmp = bst.get_dump()
        leaf_cnts_dict['XGB_GPU'] = [tree.count('leaf') for tree in dmp]

    # LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    watchlist = [lgb_train, lgb_valid]
    print ("training LightGBM")
    evals_result = {}
    t0 = time.time()
    gbm = lgb.train(params_lgb, lgb_train, num_boost_round=n_rounds,
                    valid_sets=watchlist, evals_result=evals_result, verbose_eval=False)
    time_sec_lst.append(time.time() - t0)
    print ("LightGBM: %s seconds" % (str(time_sec_lst[-1])))
    df_score_train['LGBM'] = evals_result['training'][metric]
    df_score_valid['LGBM'] = evals_result['valid_1'][metric]
    feat_imps_dict['LGBM'] = gbm.feature_importance("gain")
    model_json = gbm.dump_model()
    tree_lst = [str(tree['tree_structure']) for tree in model_json['tree_info']]
    leaf_cnts_dict['LGBM'] = [tree.count('leaf_value') for tree in tree_lst]

    print('\n%s train' % metric)
    print(df_score_train.iloc[::n_skip,])
    print('\n%s valid' % metric)
    print(df_score_valid.iloc[::n_skip,])

    if use_gpu:
        columns = ['XGB_CPU', 'XGB_GPU', 'LGBM']
    else:
        columns = ['XGB_CPU', 'LGBM']
    print('\nLeaf counts')
    df_leaf_cnts = pd.DataFrame(leaf_cnts_dict, columns=columns)
    print(df_leaf_cnts.iloc[::n_skip,])

    df_feat_imps = pd.DataFrame(feat_imps_dict,
                                index=feature_names,
                                columns=columns).fillna(0)
    df_feat_imps /= df_feat_imps.sum(0)
    df_feat_imps = df_feat_imps.sort_values('XGB_CPU', ascending=False)
    print('\nFeature importance sorted by XGB_CPU') # added sorted by XGB_CPU after exp001
    print(df_feat_imps.head(5))
    if fname_header is not None:
        df_score_train.to_csv('log/' + fname_header + 'Score_Train_' + fname_footer)
        df_score_valid.to_csv('log/' + fname_header + 'Score_Valid_' + fname_footer)
        df_leaf_cnts.to_csv('log/' + fname_header + 'Leaf_cnts_' + fname_footer)
        df_feat_imps.to_csv('log/' + fname_header + 'Feat_imps_' + fname_footer)
    return(time_sec_lst)

def exp_binary_lgbm(X_train, y_train, X_valid, y_valid,
                    params_lgb_lst, model_str_lst, metric,
                    n_rounds=10,
                    fname_header=None, fname_footer=None, n_skip=10):
    t000 = time.time()

    df_score_train = pd.DataFrame(index=range(n_rounds))
    df_score_valid = pd.DataFrame(index=range(n_rounds))
    feature_names = ['f%d' % i for i in range(X_train.shape[1])]
    feat_imps_dict = {}
    leaf_cnts_dict = {}
    time_sec_lst = []

    # LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    watchlist = [lgb_train, lgb_valid]
    print ("training LightGBM")
    for params_lgb,  model_str in zip(params_lgb_lst, model_str_lst):
        evals_result = {}
        t0 = time.time()
        gbm = lgb.train(params_lgb, lgb_train, num_boost_round=n_rounds,
                        valid_sets=watchlist, evals_result=evals_result, verbose_eval=False)
        time_sec_lst.append(time.time() - t0)
        print ("LightGBM: %s seconds" % (str(time_sec_lst[-1])))
        df_score_train[model_str] = evals_result['training'][metric]
        df_score_valid[model_str] = evals_result['valid_1'][metric]
        model_json = gbm.dump_model()
        tree_lst = [str(tree['tree_structure']) for tree in model_json['tree_info']]
        leaf_cnts_dict[model_str] = [tree.count('leaf_value') for tree in tree_lst]

    print('\n%s train' % metric)
    print(df_score_train.iloc[::n_skip,])
    print('\n%s valid' % metric)
    print(df_score_valid.iloc[::n_skip,])

    print('\nLeaf counts')
    df_leaf_cnts = pd.DataFrame(leaf_cnts_dict, columns=model_str_lst)
    print(df_leaf_cnts.iloc[::n_skip,])

    if fname_header is not None:
        df_score_train.to_csv('log/' + fname_header + 'Score_Train_' + fname_footer)
        df_score_valid.to_csv('log/' + fname_header + 'Score_Valid_' + fname_footer)
        df_leaf_cnts.to_csv('log/' + fname_header + 'Leaf_cnts_' + fname_footer)
    return(time_sec_lst)
