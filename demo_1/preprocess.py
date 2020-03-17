import gc
from datetime import timedelta

import pandas as pd
from joblib import dump, load
from demo_1.level_1_helper_functions import clip_data, get_mean_of_cyl_values
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

from demo_1 import Configs


def preprocess_train(train, model=None):
    if model == 'xgb':

        train = train.dropna(axis=1, how='all', inplace=False)

        print('Length of train : ' + str(train.shape))

        train = clip_data(train)

        train = train.resample('H').mean()

        train = get_mean_of_cyl_values(train)
        train = train.rolling(window=50).mean()
        print('Train shape :' + str(train.shape))

        xposition = [u'2016-05-02 00:00:00', u'2016-11-28 00:00:00', u'2017-06-21 00:00:00', u'2018-02-07 00:00:00',
                     u'2018-02-21 00:00:00', u'2018-02-21 00:00:00',
                     u'2018-02-22 00:00:00']

        WINDOW = 30 * 24
        train['label'] = 0

        for i in xposition:
            train['label'].loc[i] = 1

        idx = train.index[train['label'] == 1]
        for j in idx:
            train['label'].loc[j - timedelta(hours=WINDOW): j] = 1

        return train

    elif model == 'lstm':

        train = train.dropna(axis=1, how='all', inplace=False)

        train = clip_data(train)

        train = get_mean_of_cyl_values(train)

        train_max = train.resample('H').max().add_suffix('_max')
        train_min = train.resample('H').min().add_suffix('_min')
        train_std = train.resample('H').std().add_suffix('_std')
        train = train.resample('H').mean()

        train = pd.concat([train, train_max], axis=1, sort=False)
        train = pd.concat([train, train_min], axis=1, sort=False)
        train = pd.concat([train, train_std], axis=1, sort=False)
        del train_max, train_min,
        gc.collect()

        train = train.rolling(window=150).mean()

        print('Train shape :' + str(train.shape))

        vds_5 = train

        del train
        gc.collect()

        vds_5 = vds_5.replace(to_replace=0, value=1)

        vds_5 = vds_5.pct_change(periods=1, fill_method='ffill')
        #
        vds_5 = vds_5.fillna(method='ffill')
        vds_5 = vds_5.fillna(method='bfill')

        vds_55 = normalize(vds_5)
        vds_55 = scale(vds_55)

        n_components_ = 50
        pca = FastICA(n_components=n_components_)

        dump(pca, 'pca.joblib')

        pca2_results = pca.fit_transform(vds_55)
        # n_comp=pca.n_components_
        n_comp = n_components_
        print('Number of componeds : ' + str(n_comp))
        # print(pca2_results)
        # print (len(pca2_results[:, 1]))

        for i in range(0, n_comp):
            vds_5['pca_' + str(i)] = 0
            # print(len(vds_5['pca_' + str(i)]))
            # print(len(pca2_results[:, i]))
            vds_5['pca_' + str(i)] = pca2_results[:, i]

        pca_columns = [x for x in vds_5.columns if x.startswith('pca')]
        vds_5 = vds_5[pca_columns]
        return vds_5

    elif model == 'svm':

        train = train.dropna(axis=1, how='all', inplace=False)

        train = clip_data(train)

        train_max = train.resample('H').max().add_suffix('_max')
        train_min = train.resample('H').min().add_suffix('_min')
        train_std = train.resample('H').std().add_suffix('_std')
        train = train.resample('H').mean()

        train = pd.concat([train, train_max], axis=1, sort=False)
        train = pd.concat([train, train_min], axis=1, sort=False)
        train = pd.concat([train, train_std], axis=1, sort=False)
        del train_max, train_min,
        gc.collect()

        train = get_mean_of_cyl_values(train)

        vds_5 = train
        print('VDS_5 shape :' + str(vds_5.shape))

        vds_5 = vds_5.replace(to_replace=0, value=1)

        vds_5 = vds_5.pct_change(periods=1, fill_method='ffill')

        vds_5 = vds_5.fillna(method='ffill')
        vds_5 = vds_5.fillna(method='bfill')

        return vds_5


def dirty_preprocess_test(test, model=None):
    test = test.dropna(axis=1, how='all', inplace=False)

    print('Length of test : ' + str(test.shape))

    if model == 'xgb':
        # test = clip_data(test)
        test = test.resample('H').mean()
        # test = test.rolling(window=50).mean()

        test = get_mean_of_cyl_values(test)

        return test

    elif model == 'lstm':

        # test = clip_data(test)
        test = get_mean_of_cyl_values(test)

        test_max = test.resample('H').max().add_suffix('_max')
        test_min = test.resample('H').min().add_suffix('_min')
        test_std = test.resample('H').std().add_suffix('_std')
        test = test.resample('H').mean()

        test = pd.concat([test, test_max], axis=1, sort=False)
        test = pd.concat([test, test_min], axis=1, sort=False)
        test = pd.concat([test, test_std], axis=1, sort=False)
        del test_max, test_min,
        gc.collect()

        # test = test.rolling(window=150).mean()

        test = test.replace(to_replace=0, value=1)

        test = test.pct_change(periods=1, fill_method='ffill')
        #
        test = test.fillna(method='ffill')
        test = test.fillna(method='bfill')

        vds_55 = normalize(test)
        vds_55 = scale(vds_55)

        n_components_ = 50

        pca = load('pca.joblib')
        pca2_results = pca.fit_transform(vds_55)

        n_comp = n_components_
        print('Number of componeds : ' + str(n_comp))
        # print(pca2_results)
        # print (len(pca2_results[:, 1]))

        for i in range(0, n_comp):
            test['pca_' + str(i)] = 0
            test['pca_' + str(i)] = pca2_results[:, i]

        pca_columns = [x for x in test.columns if x.startswith('pca')]
        test = test[pca_columns]

        return test

    elif model == 'svm':
        test = test.dropna(axis=1, how='all', inplace=False)

        print('Length of test : ' + str(test.shape))

        print('Length of test : ' + str(test.shape))

        # for column in test.columns:
        #     if column in list(Configs.CLIP_DICT.keys()):
        #         print(test[column].head())
        #         print(Configs.CLIP_DICT[column])
        #         test[column] = test[column].clip(lower=Configs.CLIP_DICT[column][0],
        #                                          upper=Configs.CLIP_DICT[column][1])
        #         print(test[column].head())

        test_max = test.resample('H').max().add_suffix('_max')
        test_min = test.resample('H').min().add_suffix('_min')
        test_std = test.resample('H').std().add_suffix('_std')
        test = test.resample('H').mean()

        test = pd.concat([test, test_max], axis=1, sort=False)
        test = pd.concat([test, test_min], axis=1, sort=False)
        test = pd.concat([test, test_std], axis=1, sort=False)
        del test_max, test_min,
        gc.collect()

        test = get_mean_of_cyl_values(test)

        vds_5 = test
        print('VDS_5 shape :' + str(vds_5.shape))

        vds_5 = vds_5.replace(to_replace=0, value=1)

        vds_5 = vds_5.pct_change(periods=1, fill_method='ffill')

        vds_5 = vds_5.fillna(method='ffill')
        vds_5 = vds_5.fillna(method='bfill')

        return vds_5
    elif model == 'perm':

        test = test.resample('H').mean()
        # test = test.rolling(window=20).mean()

        print('Test shape :' + str(test.shape))

        # test = clip_data(test)
        return test


def preprocess_test(test, model=None):
    test = test.dropna(axis=1, how='all', inplace=False)

    print('Length of test : ' + str(test.shape))

    if model == 'xgb':
        test = clip_data(test)
        test = test.resample('H').mean()
        test = test.rolling(window=50).mean()

        test = get_mean_of_cyl_values(test)

        return test

    elif model == 'lstm':

        test = clip_data(test)
        test = get_mean_of_cyl_values(test)

        test_max = test.resample('H').max().add_suffix('_max')
        test_min = test.resample('H').min().add_suffix('_min')
        test_std = test.resample('H').std().add_suffix('_std')
        test = test.resample('H').mean()

        test = pd.concat([test, test_max], axis=1, sort=False)
        test = pd.concat([test, test_min], axis=1, sort=False)
        test = pd.concat([test, test_std], axis=1, sort=False)
        del test_max, test_min,
        gc.collect()

        test = test.rolling(window=150).mean()

        test = test.replace(to_replace=0, value=1)

        test = test.pct_change(periods=1, fill_method='ffill')
        #
        test = test.fillna(method='ffill')
        test = test.fillna(method='bfill')

        vds_55 = normalize(test)
        vds_55 = scale(vds_55)

        n_components_ = 50

        pca = load('pca.joblib')
        pca2_results = pca.fit_transform(vds_55)

        n_comp = n_components_
        print('Number of componeds : ' + str(n_comp))
        # print(pca2_results)
        # print (len(pca2_results[:, 1]))

        for i in range(0, n_comp):
            test['pca_' + str(i)] = 0
            test['pca_' + str(i)] = pca2_results[:, i]

        pca_columns = [x for x in test.columns if x.startswith('pca')]
        test = test[pca_columns]

        return test

    elif model == 'svm':
        test = test.dropna(axis=1, how='all', inplace=False)

        print('Length of test : ' + str(test.shape))

        print('Length of test : ' + str(test.shape))

        for column in test.columns:
            if column in list(Configs.CLIP_DICT.keys()):
                # print(test[column].head())
                # print(Configs.CLIP_DICT[column])
                test[column] = test[column].clip(lower=Configs.CLIP_DICT[column][0],
                                                 upper=Configs.CLIP_DICT[column][1])
                # print(test[column].head())

        test_max = test.resample('H').max().add_suffix('_max')
        test_min = test.resample('H').min().add_suffix('_min')
        test_std = test.resample('H').std().add_suffix('_std')
        test = test.resample('H').mean()

        test = pd.concat([test, test_max], axis=1, sort=False)
        test = pd.concat([test, test_min], axis=1, sort=False)
        test = pd.concat([test, test_std], axis=1, sort=False)
        del test_max, test_min,
        gc.collect()

        test = get_mean_of_cyl_values(test)

        vds_5 = test
        print('VDS_5 shape :' + str(vds_5.shape))

        vds_5 = vds_5.replace(to_replace=0, value=1)

        vds_5 = vds_5.pct_change(periods=1, fill_method='ffill')

        vds_5 = vds_5.fillna(method='ffill')
        vds_5 = vds_5.fillna(method='bfill')

        return vds_5
    elif model == 'perm':

        test = test.resample('H').mean()
        test = test.rolling(window=20).mean()

        print('Test shape :' + str(test.shape))

        test = clip_data(test)
        return test
