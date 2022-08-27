'''
Author: Peng Bo
Date: 2022-08-27 21:54:51
LastEditTime: 2022-08-27 22:51:09
Description: 

'''
import os
import numpy as np
from os import path as osp

from sklearn import svm
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":
    root = 'radiomics_data/data2/'

    models_list = [
        svm.SVC(),
        KNeighborsClassifier(n_neighbors=4),
        ensemble.RandomForestClassifier(random_state=100),
        GaussianNB()
    ]

    file_list = [
        'RT-M', 'RT-C', 'RT-P',
        'RA-M', 'RA-C', 'RA-P',
        'RD-M', 'RD-C', 'RD-P'
    ]
    
    print('file \t svm \t KNN \t RF \t Bayes')
    for fname in file_list:
        train_X = np.loadtxt(osp.join(root, fname + '_train.feat'))
        train_y = np.loadtxt(osp.join(root, fname + '_train.meta'))
        test_X = np.loadtxt(osp.join(root, fname + '_test.feat'))
        test_y = np.loadtxt(osp.join(root, fname + '_test.meta'))
        res_list = []
        for m_idx, model in enumerate(models_list):
            model.fit(train_X, train_y)
            pred = model.predict(test_X)
            res_list.append((pred == test_y).sum()*1.0/test_y.shape[0])
        res_str = [str(round(v, 3)) for v in res_list]
        print('\t'.join([fname] + res_str))