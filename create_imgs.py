import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
for n_train in [10**4, 10**5, 10**6, 10**7]:
    leaf_cnts = []
    auc_valid = []
    loss_valid = []
    for max_depth in [5, 10, 15]:
        footer = "_n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        fname_leaf_cnts = "log/exp004_" + "Leaf_cnts" + footer
        leaf_cnts.append(pd.read_csv(fname_leaf_cnts, index_col=0))
        fname_auc_valid = "log/exp004_" + "Score_Valid" + footer
        auc_valid.append(pd.read_csv(fname_auc_valid, index_col=0))
        fname_loss_valid = "log/exp007_" + "Score_Valid" + footer
        loss_valid.append(pd.read_csv(fname_loss_valid, index_col=0))
        leaf_cnts[-1].columns = auc_valid[-1].columns = ['XGB', 'LGB']
        
    # Leaf cnts
    fig, axes = plt.subplots(3, 1, figsize=(19.8, 10))
    plt.subplots_adjust(top=0.93)
    plt.suptitle("Higgs, Leaf counts, n_train:10**%d" % np.log10(n_train), fontsize=16)
    for i in range(3):
        leaf_cnts[i].plot(title="max_depth:%d" % (5*(i+1)),
                          ylim=(0, 1.2 * leaf_cnts[i].max().max()),
                          ax=axes[i])
        axes[i].set_ylabel("Leaf counts")
    axes[2].set_xlabel("Boosting iterations")
    plt.savefig('img/Higgs_leafcnts_%d.png' % n_train)
    
    # Logloss
    fig, axes = plt.subplots(3, 1, figsize=(19.8, 10))
    plt.subplots_adjust(top=0.93)
    plt.suptitle("Higgs, Logloss, n_train:10**%d" % np.log10(n_train), fontsize=16)
    for i in range(3):
        loss_valid[i].plot(title="max_depth:%d" % (5*(i+1)), ax=axes[i])
        axes[i].set_ylabel("logloss")
    axes[2].set_xlabel("Boosting iterations")
    plt.savefig('img/Higgs_logloss_%d.png' % n_train)

    # AUC
    fig, axes = plt.subplots(3, 1, figsize=(19.8, 10))
    plt.subplots_adjust(top=0.93)
    plt.suptitle("Higgs, AUC, n_train:10**%d" % np.log10(n_train), fontsize=16)
    for i in range(3):
        auc_valid[i].plot(title="max_depth:%d" % (5*(i+1)), ax=axes[i])
        axes[i].set_ylabel("logloss")
    axes[2].set_xlabel("Boosting iterations")
    plt.savefig('img/Higgs_auc_%d.png' % n_train)
"""

# Artificial datasets
N = 10**4
exp_lst = [{'header'      :'exp011_',
            'metric'      :'Logloss',
            'n_train_s'   :[5*10**5, 10**6, 2*10**6],
            'num_leaves_s':[32, 256, 1024, 4096, 16384],
            'max_depth_s' :[5, 10, 15, 20],
        },
           {'header'      :'exp012_',
            'metric'      :'Logloss',
            'n_train_s'   :[N, 2*N, 4*N, 8*N, 16*N, 32*N],
            'num_leaves_s':[32, 256, 1024, 4096],
            'max_depth_s' :[5, 10, 15, 20],
        }]
for exp_dict in exp_lst:
    header = exp_dict['header']
    metric = exp_dict['metric']
    for n_train in exp_dict['n_train_s']:
        for num_leaves in exp_dict['num_leaves_s']:
            leaf_cnts = []
            loss_valid = []
            max_depth_lst = []
            for max_depth in exp_dict['max_depth_s']:
                if num_leaves > 2 ** max_depth:
                    continue
                max_depth_lst.append(max_depth)
                fname_footer = "_n_%d_md_%d_nl_%d.csv" % (n_train, max_depth, num_leaves)
                fname_leaf_cnts = "log/" + header + "Leaf_cnts" + fname_footer
                leaf_cnts.append(pd.read_csv(fname_leaf_cnts, index_col=0))
                fname_loss_valid = "log/" + header + "Score_Valid" + fname_footer
                loss_valid.append(pd.read_csv(fname_loss_valid, index_col=0))
            max_depth_cnts = len(max_depth_lst)
            
            # Leaf cnts
            fig, axes = plt.subplots(max_depth_cnts, 1, figsize=(19.8, 10))
            plt.subplots_adjust(top=0.93)
            if n_train > 10**6/2:
                str_n_train = "%.1fM" % (n_train/10**6)
            else:
                str_n_train = "%dK" % int(n_train/1000)
                
            dname = header[:-1] + ", n_train:%s, num_leaves:%d" % \
                    (str_n_train, num_leaves)
            plt.suptitle(dname, fontsize=16)
            for i, max_depth in enumerate(max_depth_lst):
                leaf_cnts[i].plot(title="max_depth:%d" % max_depth,
                                  ylim=(0, 1.2 * leaf_cnts[i].max().max()),
                                  ax=axes[i])
                if i < max_depth_cnts - 1:
                    axes[i].set_xticks([])
            
                axes[i].set_ylabel("Leaf counts")
                axes[max_depth_cnts - 1].set_xlabel("Boosting iterations")
                plt.savefig("img/" + header + "leafcnts_%s_num_leaves_%d.png" % \
                            (str_n_train, num_leaves))
    
            # Score
            fig, axes = plt.subplots(max_depth_cnts, 1, figsize=(19.8, 10))
            plt.subplots_adjust(top=0.93)
            plt.suptitle(dname, fontsize=16)
            for i, max_depth in enumerate(max_depth_lst):
                loss_valid[i].plot(title="max_depth:%d" % max_depth, ax=axes[i])
                if i < max_depth_cnts - 1:
                    axes[i].set_xticks([])
                axes[i].set_ylabel(metric)
            axes[max_depth_cnts - 1].set_xlabel("Boosting iterations")
            plt.savefig("img/" + header + metric + "_%s_num_leaves_%d.png" % \
                        (str_n_train, num_leaves))
