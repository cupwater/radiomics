# coding: utf8
#!/bin/bash
'''
Author: Peng Bo
Date: 2022-07-25 13:30:40
LastEditTime: 2022-08-27 22:53:06
Description: 

'''

import os
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold, ensemble


def tsne_visualize(X, y):
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    # visualize the distributions of landmarks
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


def feature_engineer(raw_features, labels, proportion=0.99):
    # select those important features
    model = ensemble.AdaBoostClassifier(random_state=100)
    model.fit(raw_features, labels.tolist())
    importances = model.feature_importances_.tolist()
    importances_v = sorted(enumerate(importances), key=lambda x:-x[1])
    cum_porp = 0
    selected_idxs = []
    for idx, score in importances_v:
        cum_porp += score
        selected_idxs.append(idx)
        if cum_porp>proportion:
            break
    return selected_idxs


def read_data(data_path):
    # read from csv
    df = pd.read_csv(data_path)
    mean = df.mean()[23:]
    new_df = df.fillna(df.mean())
    features = []
    labels   = []
    for data in new_df.itertuples(index=False):
        for idx, v in enumerate(data[23:]):
            if np.isnan(v):
                pdb.set_trace()
                data[23+idx] = 0
        features.append(data[23:])
        labels.append(data[0])
    features = np.array(features).reshape(len(features), -1)
    return features, np.array(labels) - 1


if __name__ == "__main__":
    folder = 'radiomics_data/data2/'
    ratio = 0.7
    train_idx = np.random.choice(list(range(175)), size=int(175*ratio), replace=False)
    train_idx.sort()
    test_idx  = list(set(list(range(175))) - set(train_idx))
    test_idx.sort()

    file_list = [
        'RT-M', 'RT-C', 'RT-P',
        'RA-M', 'RA-C', 'RA-P',
        'RD-M', 'RD-C', 'RD-P'
    ]
    
    for fname in file_list:
        features, labels = read_data(os.path.join(folder, fname+'.csv'))
        normalized_features = features.copy()
        for idx in range(features.shape[1]):
            _max, _min = np.max(features[:,idx]), np.min(features[:,idx])
            normalized_features[:,idx] = (features[:,idx] - _min) / (_max - _min)
        selected_idxs = feature_engineer(normalized_features, labels)
        select_features = normalized_features[:, selected_idxs] 
        
        # write the results into file
        feat_name = os.path.join(folder, fname+ '.feat')
        np.savetxt(feat_name, select_features, fmt='%.3f')
        meta_name = os.path.join(folder, fname + '.meta')
        np.savetxt(meta_name, labels, fmt='%d')

        # split into train part and test part
        train_feat = select_features[train_idx, :]
        train_meta = labels[train_idx]
        np.savetxt(os.path.join(folder, fname + '_train.feat'), train_feat, fmt='%.3f')
        np.savetxt(os.path.join(folder, fname + '_train.meta'), train_meta, fmt='%d')

        test_feat  = select_features[test_idx, :]
        test_meta  = labels[test_idx]
        np.savetxt(os.path.join(folder, fname + '_test.feat'), test_feat, fmt='%.3f')
        np.savetxt(os.path.join(folder, fname + '_test.meta'), test_meta, fmt='%d')