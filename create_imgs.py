import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Artificial datasets
for n_train in [10**6, 2*10**6]:
    leaf_cnts = []
    loss_valid = []
    for max_depth in [5, 10, 15]:
        footer = "_n_train_%d_max_depth_%d.csv" % (n_train, max_depth)
        fname_leaf_cnts = "log/exp00%d_" + "Leaf_cnts" + footer
        leaf_cnts.append(pd.read_csv(fname_leaf_cnts % 5, index_col=0))
        leaf_cnts[-1]["XGB_Eq_binning"] = pd.read_csv(fname_leaf_cnts % 6, index_col=0)['XGB'].values
        fname_loss_valid = "log/exp00%d_" + "Score_Valid" + footer
        loss_valid.append(pd.read_csv(fname_loss_valid % 5, index_col=0))
        loss_valid[-1]["XGB_Eq_binning"] = pd.read_csv(fname_loss_valid % 6, index_col=0)['XGB'].values
    # Leaf cnts
    fig, axes = plt.subplots(3, 1, figsize=(19.8, 10))
    plt.subplots_adjust(top=0.93)
    plt.suptitle("Leaf counts, Aritificial dataset, n_train:%dM" % (n_train/10**6), fontsize=16)
    for i in range(3):
        leaf_cnts[i].plot(title="max_depth:%d" % (5*(i+1)),
                          ylim=(0, 1.2 * leaf_cnts[i].max().max()),
                          ax=axes[i])
        axes[i].set_ylabel("Leaf counts")
    axes[2].set_xlabel("Boosting iterations")
    plt.savefig("img/Adata_leafcnts_%dM.png" % (n_train/10**6))
    
    # Logloss
    fig, axes = plt.subplots(3, 1, figsize=(19.8, 10))
    plt.subplots_adjust(top=0.93)
    plt.suptitle("Logloss, Aritificial dataset, n_train:%dM" % (n_train/10**6), fontsize=16)
    for i in range(3):
        loss_valid[i].plot(title="max_depth:%d" % (5*(i+1)), ax=axes[i])
        axes[i].set_ylabel("logloss")
    axes[2].set_xlabel("Boosting iterations")
    plt.savefig("img/Adata_logloss_%dM.png" % (n_train/10**6))
