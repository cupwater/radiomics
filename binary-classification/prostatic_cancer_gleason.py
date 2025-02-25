import csv
import datetime
import os

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.svm
import yaml
import time

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from scipy.stats import levene, ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, scale, normalize, label_binarize
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

seed = 4396


def featurs_deal(csv_file):
    print('reading data from ', csv_file)

    data_ori = pd.read_csv(csv_file)
    header_row = next(csv.reader(open(csv_file)))
    str_idx = []
    # find the columns index where the type of cell is 'string'
    for idx, i in enumerate(data_ori.iloc[1, :]):
        if isinstance(i, str):
            str_idx.append(idx)

    data = data_ori
    for i in str_idx:
        data = data.drop(columns=[header_row[i]])

    # low level cases and high level cases in data are 59 and 116
    # make 'low:high' in testset is 1:1
    # (59-33):(116-90) = 26:26 in testset,
    # 33:90 in trainset.
    data_L = data[:][data['Label'] == 1]
    data_H = data[:][data['Label'] == 2]

    # shuffle
    data_L = data_L.sample(frac=1.0, random_state=seed)
    data_H = data_H.sample(frac=1.0, random_state=seed)

    data_train_L = data_L.iloc[:33, :]
    data_train_H = data_H.iloc[:90, :]

    data_test_L = data_L.iloc[33:, :]
    data_test_H = data_H.iloc[90:, :]

    # T test
    # index = []
    # for colName in data.columns[:]:
    #     if levene(data_train_L[colName], data_train_H[colName])[1] > 0.05:
    #         if ttest_ind(data_train_L[colName], data_train_H[colName])[1] < 0.05:
    #             # 独立样本T检验结果具有显著性差异(p > 0.05), 特征索引colName加入index
    #             index.append(colName)
    #     else:
    #         if ttest_ind(data_train_L[colName], data_train_H[colName], equal_var=False)[1] < 0.05:
    #             index.append(colName)
    #
    # print(len(index), 'features obtained after T test: ', index)

    # concat LABEL1 and LABEL2
    # data_train_L = data_train_L[index]
    # data_train_H = data_train_H[index]
    data_train = pd.concat([data_train_L, data_train_H])
    # shuffle
    data_train = shuffle(data_train)
    x_train = data_train[data_train.columns[1:]]

    # data_test_L = data_test_L[index]
    # data_test_H = data_test_H[index]
    data_test = pd.concat([data_test_L, data_test_H])
    data_test = shuffle(data_test)
    x_test = data_test[data_test.columns[1:]]

    # keep columns-name and normalize
    columns = x_train.columns
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train, columns=columns)
    x_test = pd.DataFrame(x_test, columns=columns)
    y_train = data_train['Label']
    y_test = data_test['Label']

    # 原本标签1&2 变成0&1
    y_train = label_binarize(y_train, classes=[1, 2])
    y_train = y_train.ravel()
    y_test = label_binarize(y_test, classes=[1, 2])
    y_test = y_test.ravel()

    # drop columns include 'NaN'
    is_NaN = x_train.isnull().any()  # False - 无缺失值
    if sum(is_NaN[is_NaN == True]) > 0:
        # print index name
        # NaN_index = is_NaN[is_NaN == True].index
        # print(NaN_index)

        # drop these columns include 'NaN'
        # not_NaN_index = is_NaN[is_NaN == False].index
        # x_train = x_train[not_NaN_index]
        # x_test = x_test[not_NaN_index]

        # fill 'NaN' with mean()
        x_train.fillna(x_train.mean(), inplace=True)
        x_test.fillna(x_test.mean(), inplace=True)

    # oversampling
    sm = SMOTE(random_state=seed)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

    # shuffle the resampled trainset
    x_train_res.insert(loc=0, column='Label', value=y_train_res)
    x_train_res = x_train_res.sample(frac=1.0, random_state=seed)
    x_train = x_train_res[x_train_res.columns[1:]]
    y_train = x_train_res['Label'].ravel()

    return x_train, y_train, x_test, y_test


def features_reduction(data, filter, based=None):
    x_train, y_train, x_test, y_test = data

    # lasso
    if based == None:
        lassoCV = LassoCV(max_iter=10000, cv=5, random_state=seed).fit(x_train, y_train)
        coef = pd.Series(lassoCV.coef_, index=x_train.columns)
        print('α - %.4f , %s %d ' % (lassoCV.alpha_, 'Lasso picked', sum(coef != 0)))

        # Draw MSE with different alphas
        plt.figure()
        # 随着alpha值的变化，均方误差的变化曲线
        plt.plot(lassoCV.alphas_, lassoCV.mse_path_, ':')
        # 验证过程中，随着alpha值的变化，均方误差的平均曲线
        plt.plot(lassoCV.alphas_, lassoCV.mse_path_.mean(axis=-1),
                 label='Average MSE Across Folds', linewidth=2, color='cyan')
        # 每次验证系统认为的最合适的alpha值
        plt.axvline(lassoCV.alpha_, linestyle='--', label='CV Estimate of Best alpha', color='blue')
        plt.semilogx()
        plt.legend()
        plt.xlabel('alpha')
        plt.ylabel('Mean Square Error')
        plt.axis('tight')
        plt.savefig('../output/lassoCV_picked_{}.png'.format(sum(coef != 0)))
        plt.show()

        index = coef[coef != 0].index
        x_train = x_train[index]
        x_test = x_test[index]

    # rfe
    else:
        if based == 'LR':
            model = LogisticRegression(random_state=seed)
            rfe = RFE(estimator=model, n_features_to_select=10)
            selector = rfe.fit(x_train, y_train)

            support = pd.Series(selector.support_, index=x_train.columns)
            index = support[support == True].index
            x_train = x_train[index]
            x_test = x_test[index]


        elif based == 'SVM':
            model = SVC(kernel='linear', probability=True, random_state=seed)
            rfe = RFE(estimator=model, n_features_to_select=10)
            selector = rfe.fit(x_train, y_train)

            # method 1:
            support = pd.Series(selector.support_, index=x_train.columns)
            index = support[support == True].index
            x_train = x_train[index]
            x_test = x_test[index]

        elif based == 'PCA':
            print('applying PCA ...')
            pca = PCA(n_components=0.95)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)

        # Spearman rank correlation test
        x_train_df = pd.DataFrame(x_train)
        corr_matrix = x_train_df.corr(method='spearman').abs()
        # draw heatmap
        sns.set(font_scale=1.0)
        f, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr_matrix, cmap="YlOrRd", square=True, ax=ax)
        f.tight_layout()
        plt.savefig("correlation_matrix.png", dpi=800)

    return x_train, y_train, x_test, y_test


def classify(model, data):
    # StratifiedKFold - 分层抽样：训练集，测试集中各类别样本的比例与原始数据集中相同
    kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x_train, y_train, x_test, y_test = data

    classifier = None
    parameters_grid = None
    if model == 'RF':
        classifier = RandomForestClassifier(n_jobs=-1)

        parameters_grid = {
            'n_estimators': [int(x) for x in np.arange(30,50)],
            # 'max_depth': [3, 4, 5, 6, 7, 8],
            # 'max_features': ["auto", "sqat", "log2"],
            # 'class_weight': [None, "balanced"]
        }

    elif model == 'KNN':
        classifier = KNeighborsClassifier(n_jobs=-1)

        parameters_grid = {
            "n_neighbors": np.arange(4,10),
            # "leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        }

    elif model == 'LR':
        classifier = LogisticRegression(n_jobs=-1)

        parameters_grid = {
            # "penalty": ['l1', 'l2'],
            "C": [0.1, 0.5 ,1 ,2, 3, 4, 5, 6, 7, 8, 9],
            # "class_weight": [None, 'balanced']
        }


    elif model == 'SVM':
        classifier = SVC(probability=True, gamma='auto')

        parameters_grid = {
            # "kernel": ['linear', 'rbf', 'poly'],
            "C": [0.1, 0.5 ,1 ,2, 3, 4, 5, 6, 7, 8, 9],
        }


    elif model == 'Bayes':
        classifier = GaussianNB()

        parameters_grid = {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }

    if parameters_grid is not None:
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters_grid,
                                   cv=kflod,
                                   scoring='accuracy')
        grid_result = grid_search.fit(x_train, y_train)
        print('Best ' + model + ' Train Score: %.4f using %s' % (grid_result.best_score_, grid_search.best_params_))
        print('Best ' + model + ' Model: ', grid_search.best_estimator_)
        pre_proba = grid_search.best_estimator_.predict_proba(x_test)
        pre = grid_result.best_estimator_.predict(x_test)
        # print(x_test)
        # print(y_test)
        # print(pre_proba)
        # print(pre)
        accuracy, precision, recall, f1, auc = accuracy_score(y_test, pre), \
                                               precision_score(y_test, pre), \
                                               recall_score(y_test, pre), \
                                               f1_score(y_test, pre), \
                                               roc_auc_score(y_test, pre_proba[:, 1])

        print(' accuracy:%.2f \n precision:%.2f \n recall:%.2f \n f1:%.2f \n auc:%.2f' % (
        accuracy, precision, recall, f1, auc))
        # test_score = accuracy
        # test_score = grid_search.best_estimator_.fit(x_train, y_train).score(x_test, y_test)
        # print(model + ' Test Score: ', test_score)

        return accuracy, precision, recall, f1, auc


def create_log(config, filter, model, csv_name, result, based=None):
    file = os.path.join(config['file_path']['log'], 'log.txt')
    if not os.path.isfile(file):
        f = open(file, mode="w", encoding="utf-8")
        f.write('%s%s%s%s%s%s%s%s\n' % ('features'.center(15, ' '),
                                        'model'.center(15, ' '),
                                        'select method'.center(15, ' '),
                                        'accuracy'.center(15, ' '),
                                        'precision'.center(15, ' '),
                                        'recall'.center(15, ' '),
                                        'f1'.center(15, ' '),
                                        'auc'.center(15, ' ')))
        f.close()

    with open(file, 'a') as f:
        method = filter
        if filter == 'RFE':
            method = filter + '(' + based + ')'
        f.write('%s%s%s%s%s%s%s%s%s\n' % (csv_name.center(15, ' '),
                                          model.center(15, ' '),
                                          method.center(15, ' '),
                                          str(np.around(result[0], 4)).center(15, ' '),
                                          str(np.around(result[1], 4)).center(15, ' '),
                                          str(np.around(result[2], 4)).center(15, ' '),
                                          str(np.around(result[3], 4)).center(15, ' '),
                                          str(np.around(result[4], 4)).center(15, ' '),
                                          time.strftime('%Y-%m-%d %H:%M', time.localtime()).center(20, ' ')))


def main(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    modality = config['object']['series']
    roi = config['object']['roi']
    classifier = config['classifier']

    csv_file = 'R' + modality + '-' + roi + '.csv'
    dealt_data = featurs_deal(os.path.join(config['file_path']['csv'], csv_file))

    classifier = [x for x in classifier.values()]
    filter = [x for x in classifier[0].split(',')]
    model = [x for x in classifier[1].split(',')]
    based = [x for x in classifier[2].split(',')]

    for fi in filter:
        if fi == 'RFE':
            for b in based:
                reduced_data = features_reduction(dealt_data, fi, b)
                for m in model:
                    print('using %s with %s based on %s ...' % (m, fi, b))
                    t0 = datetime.datetime.now()
                    result = classify(m, reduced_data)
                    print('spend time: {}s'.format((datetime.datetime.now() - t0).seconds))
                    create_log(config, fi, m, csv_file, result, based=b)
        else:
            reduced_data = features_reduction(dealt_data, fi)
            for m in model:
                print('using %s with %s ...' % (m, fi))
                t0 = datetime.datetime.now()
                result = classify(m, reduced_data)
                print('spend time: {}s'.format((datetime.datetime.now() - t0).seconds))
                create_log(config, fi, m, csv_file, result)


if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Prostatic Cancer Classifier on Gleason')
    parser.add_argument('--config-file', type=str, default='./config.yaml')
    args = parser.parse_args()
    main(args.config_file)
